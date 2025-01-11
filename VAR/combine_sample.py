import os
import random

import numpy as np
import torch
from torchvision.utils import save_image

import dist
from utils import arg_util
from models import build_vae_dd

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor

@torch.no_grad()
def transfer_BLCv_to_B3HW(x, dd_model, vae):
    patch_nums = dd_model.patch_nums[1:]
    cur_idx = 0
    B = x.shape[0]
    f_hat = x.new_zeros(B, dd_model.Cvae, patch_nums[-1], patch_nums[-1])
    
    for i in range(len(patch_nums)):
        pn = patch_nums[i]
        h_level_i = x[:, cur_idx: cur_idx +  pn ** 2]
        h_BChw = h_level_i.transpose_(1, 2).reshape(B, dd_model.Cvae, pn, pn)
        f_hat, _ = vae.quantize.get_next_autoregressive_input(i, len(patch_nums), f_hat, h_BChw)
        cur_idx += pn ** 2
    
    result = vae.fhat_to_img(f_hat).add_(1).mul_(0.5)   

    return result

@torch.no_grad()
def sample():
    
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    
    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    t_list = [int(d) for d in args.t_list.split(",")]
    
    vae, dd_model = build_vae_dd(
        device=dist.get_device(), patch_nums=args.patch_nums,
        # VQVAE args
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        num_classes=1000, depth=args.depth, shared_aln=args.saln, attn_l2_norm=args.anorm,
        flash_if_available=args.fuse, fused_if_available=args.fuse,
        init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini,
        n_d_embed_enable=args.n_d_embed_enable,
        layerwise_n_d_embed=args.layerwise_n_d_embed,
    )
    
    from models import build_vae_var
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4, # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=args.depth, shared_aln=False,
        vae_local=vae,
    )
    
    # load weights for dd model and vae
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location='cpu'), strict=True)
    dd_state_dict = torch.load(args.ckpt_path, map_location='cpu')
    var.load_state_dict(torch.load(args.teacher_path, map_location="cpu"), strict=True)
    if args.ema:
        m, u = dd_model.load_state_dict(dd_state_dict['trainer']["ema_parameters"], strict=False)
        print(f"Missing keys for EMA: {m}")
        print(f"Unexpected keys for EMA: {u}")
    else:
        m, u = dd_model.load_state_dict(dd_state_dict['trainer']["var_wo_ddp"], strict=False)
        print(f"Missing keys for Model: {m}")
        print(f"Unexpected keys for Model: {u}")
    
    # save original image
    save_path = os.path.join(args.sample_dir, "images")
    os.makedirs(save_path, exist_ok=True)
    device = args.device
    img_id = 0
    
    vae.eval()
    dd_model.eval()
    
    labels = torch.randint(0, 1000, (args.sample_num, ))

    while(1):
        
        if img_id >= args.sample_num:
            break

        cur_bs = min(args.bs, args.sample_num - img_id)
        
        label_B: torch.LongTensor = labels[img_id:img_id + cur_bs].to(device)
        noise = torch.randn([cur_bs, dd_model.L - 1, dd_model.Cvae]).to(device)
        x = noise

        # first DD step inference
        if args.t_teacher_start > 0:
            t = t_list[0]
            print(f"Timestep {t} | ckpt: {args.ckpt_path}")
            tensor_t = torch.zeros([cur_bs]) + t
            x = dd_model.perturb_data(x, tensor_t, noise)
            
            final_predict_t = dd_model.final_predict(label_B, x, tensor_t)
            final_predict_conti, final_predict_logits = final_predict_t
            final_predict_t = dd_model.get_final_sequence(final_predict_conti, final_predict_logits, args.loss_split_t)
            x = final_predict_t 
            
            # quant x
            codebook = dd_model.vae_quant_proxy[0].embedding.weight.data
            idx = (codebook[:, None, None, :] - x).square().sum(dim=-1).argmin(dim=0)
            x = codebook[idx]

        # intermediate teacher step
        _, sequence = var.intermediate_sample(
                args.t_teacher_start,
                label_B, 
                x,
                end_point=t_list[1],
                cfg=args.cfg, top_k=900, top_p=0.95,
            )
        sequence = torch.cat([sequence.to(device), noise[:, sequence.shape[1]:]], dim=1)
        final_predict_t = sequence

        # second DD step inference
        x = sequence
        t = t_list[1]
        if t < 10:
            print(f"Timestep {t} | ckpt: {args.ckpt_path}")
            tensor_t = torch.zeros([cur_bs]) + t
            x = dd_model.perturb_data(x, tensor_t, noise)
            final_predict_t = dd_model.final_predict(label_B, x, tensor_t)
            final_predict_conti, final_predict_logits = final_predict_t
            final_predict_t = dd_model.get_final_sequence(final_predict_conti, final_predict_logits, args.loss_split_t)
                
        # from token sequence to image                
        recon_B3HW = transfer_BLCv_to_B3HW(final_predict_t, dd_model, vae)

        # save image
        for i in range(len(recon_B3HW)):
            save_image(recon_B3HW[i], os.path.join(save_path, f"{img_id}.png"))
            img_id += 1
            
        print(f"generate {img_id} images")

    act_path = args.fid_statistic 
    from evaluation.fid_score import calculate_fid_given_paths
    fid = calculate_fid_given_paths(
        (None, str(save_path)), 
        batch_size=256, 
        device=dist.get_device(), 
        dims=2048, 
        num_workers=8, 
        load_act=[act_path, None],
        save_act=[None, None],
    )
    print(f"fid: {fid}")
    
    with open(os.path.join(args.sample_dir,"fid.txt"), "w") as f:
        f.write(f"FID: {fid}")
    
    return fid

if __name__=="__main__":
    sample()

