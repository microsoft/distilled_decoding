# Modified from:
#   DiT:  https://github.com/facebookresearch/DiT/blob/main/sample.py
import sys
sys.path.append(".")
sys.path.append("autoregressive")
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
from torchvision.utils import save_image

import os
import random
import argparse
from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.dd import GPT_models
from autoregressive.models.gpt import GPT_models as teacher_GPT
from autoregressive.models.generate import generate, intermediate_generate
import torch.nn.functional as F

def perturb_data(data, t, noise=None):
    B, L, C = data.shape
    mask = (torch.arange(L).unsqueeze(0).expand(B, L) >= t.unsqueeze(1)).float().to(data.device)
    if noise is None:
        noise = torch.randn_like(data)
    x_t = data * (1 - mask[:, :, None]) + noise * mask[:, :, None]
    return x_t

@torch.no_grad()
def eval(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"], strict=True)
    codebook = vq_model.quantize.embedding.weight.data
    if vq_model.quantize.l2_norm:
        codebook = F.normalize(codebook, p=2, dim=-1)
    del checkpoint
    print(f"image tokenizer is loaded")

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        codebook_dim=args.codebook_embed_dim,
        codebook=codebook,
    ).to(device=device)
    
    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    if args.ema:
        model_weight = checkpoint["ema"]
    else:
        if args.from_fsdp: # fspd
            model_weight = checkpoint
        elif "model" in checkpoint:  # ddp
            model_weight = checkpoint["model"]
        elif "module" in checkpoint: # deepspeed
            model_weight = checkpoint["module"]
        elif "state_dict" in checkpoint:
            model_weight = checkpoint["state_dict"]
        else:
            raise Exception("please check model weight, maybe add --from-fsdp to run command")
    # if 'freqs_cis' in model_weight:
    #     model_weight.pop('freqs_cis')
    m, u = gpt_model.load_state_dict(model_weight, strict=False)
    print(f"Missing keys for model loading: {m}")
    print(f"Unexpected keys for model loading: {u}")
    gpt_model.eval()
    del checkpoint
    print(f"gpt model is loaded")
    
    # create and load teacher model
    teacher_gpt_model = teacher_GPT[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)
    checkpoint = torch.load(args.teacher_ckpt, map_location="cpu")
    if args.from_fsdp: # fspd
        model_weight = checkpoint
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight, maybe add --from-fsdp to run command")
    teacher_gpt_model.load_state_dict(model_weight, strict=True)
    teacher_gpt_model.eval()
    del checkpoint
    print(f"teacher gpt model is loaded")
    
    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
        teacher_gpt_model = torch.compile(
            teacher_gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no need to compile model in demo") 
    
    sampling_kwargs = {}
            
    t_list = [int(d) for d in args.t_list.split(",")]
    
    save_path = os.path.join(args.save_path, "images")
    os.makedirs(save_path, exist_ok=True)
    img_id = 0
    
    while(1):
        
        if img_id >= args.image_num:
            break
        cur_bs = min(args.bs, args.image_num - img_id)
        
        class_labels = [random.randint(0, 999) for _ in range(cur_bs)]
        cur_noise = torch.randn([cur_bs, latent_size ** 2, args.codebook_embed_dim]).to(device)
            
        c_indices = torch.tensor(class_labels, device=device)
        x = cur_noise
        
        # first step
        if args.teacher_start_t > 0:
            t = torch.zeros([cur_bs]) + t_list[0]
            x_t = perturb_data(x, t, cur_noise)
            gt_idx = (codebook[:, None, None, :] - x_t).square().sum(dim=-1).argmin(dim=0)
            final_predict_results = gpt_model.final_predict(x_BLC=x_t, cond_idx=c_indices, timestep=t, idx=gt_idx)
            final_predict = gpt_model.get_final_x_BLC(final_predict_results, args.split_t)
            x = final_predict
            # quant the output
            idx = (codebook[:, None, None, :] - x).square().sum(dim=-1).argmin(dim=0)
            idx = idx[:, :args.teacher_start_t]
        
        # teacher steps
        if args.teacher_start_t > 0:
            index_sample = intermediate_generate(
                teacher_gpt_model, c_indices, prefix=idx,
                t_start=args.teacher_start_t, max_new_tokens=t_list[1]-args.teacher_start_t,
                cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
                temperature=args.temperature, top_k=args.top_k,
                top_p=args.top_p, sample_logits=True, 
                **sampling_kwargs,
            )
        else:
            index_sample = generate(
                teacher_gpt_model, c_indices, max_new_tokens=t_list[1]-args.teacher_start_t,
                cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
                temperature=args.temperature, top_k=args.top_k,
                top_p=args.top_p, sample_logits=True, 
                **sampling_kwargs,
            )
        
        x = codebook[index_sample]
        final_predict = x
        print(final_predict.shape)
        
        # second step
        if t_list[1] < latent_size ** 2:
            x = torch.cat([x, cur_noise[:, x.shape[1]:]], dim=1)
            t = torch.zeros([cur_bs]) + t_list[1]
            x_t = perturb_data(x, t, cur_noise)
            gt_idx = (codebook[:, None, None, :] - x_t).square().sum(dim=-1).argmin(dim=0)
            final_predict_results = gpt_model.final_predict(x_BLC=x_t, cond_idx=c_indices, timestep=t, idx=gt_idx)
            final_predict = gpt_model.get_final_x_BLC(final_predict_results, args.split_t)
            x = final_predict
  
        qzshape = [cur_bs, args.codebook_embed_dim, latent_size, latent_size]
    
        final_predict = final_predict.reshape(qzshape[0], qzshape[2], qzshape[3], qzshape[1])
        final_predict = final_predict.permute(0, 3, 1, 2).contiguous()
        samples = vq_model.decode(final_predict)
        samples = (samples + 1) / 2

        # Save and images:
        for sample in samples:
            save_image(sample, os.path.join(save_path, f"{img_id}.png"))
            img_id += 1
        print(f"Generated {img_id} images")
        
    # eval fid
    act_path = args.fid_statistic 
    from evaluations.fid_score import calculate_fid_given_paths

    print(f"sample num: {len(os.listdir(save_path))}")
    fid = calculate_fid_given_paths(
        (None, str(save_path)), 
        batch_size=256, 
        device=device, 
        dims=2048, 
        num_workers=8, 
        load_act=[act_path, None],
        save_act=[None, None],
    )
    print(f"fid: {fid}")
    
    with open(os.path.join(args.save_path,"fid.txt"), "w") as f:
        f.write(f"FID: {fid}")

    return fid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1, help="GPU id")
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--teacher-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--dataset", type=str, default='ndpair')
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=384)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--top-k", type=int, default=8000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    # additional argument for DD sampling
    parser.add_argument("--image_num", type=int, default=5000)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--loss_norm", type=str, default="lpips")
    parser.add_argument("--save_path", type=str, default="outputs/debug")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--t_list", type=str, default="0")
    parser.add_argument("--split_t", type=int, default=80)
    parser.add_argument("--teacher_start_t", type=int, default=60)
    parser.add_argument("--fid_statistic", type=str, default="./statistic/imagenet256")
    args = parser.parse_args()
    eval(args)