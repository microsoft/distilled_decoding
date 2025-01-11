import math
from functools import partial
from typing import Optional, Tuple, Union

import os

import torch
import torch.nn as nn
from torchvision.utils import save_image
from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from models.vqvae import VQVAE, VectorQuantizer2
from models.var import VAR

def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift

class FinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size**0.5)

    def forward(self, x, cond_BD, t=None):
        if t is not None:
            shift, scale = (self.scale_shift_table[None] + t[:, None] + cond_BD[:, None]).chunk(2, dim=1)
        else:
            shift, scale = (self.scale_shift_table[None] + cond_BD[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class DD_Model(VAR):
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
        # for data and noise embedding
        n_d_embed_enable=True,
        layerwise_n_d_embed=True,
    ):
        patch_nums = tuple([1] + list(patch_nums))
        super().__init__(
            vae_local,
            num_classes, depth, embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop_rate, drop_path_rate,
            norm_eps, shared_aln, cond_drop_rate,
            attn_l2_norm,
            patch_nums,   # 10 steps by default
            flash_if_available, fused_if_available,
        )
        
        self.accum_token_length = [torch.tensor(self.patch_nums[i:]).square().sum().int().item() for i in range(len(self.patch_nums))]
        self.noise_mask = torch.zeros([len(self.accum_token_length[1:]), self.L-1])
        for i in range(len(self.noise_mask)):
            self.noise_mask[i, -self.accum_token_length[1:][i]:] = 1
        self.accum_token_length.append(0)
        self.accum_token_length = torch.tensor(self.accum_token_length)
        
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        attn_bias_for_masking = torch.where(d >= dT + 1, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        attn_bias_for_masking[0, 0].fill_diagonal_(0)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        self.n_d_embed_enable = n_d_embed_enable
        self.layerwise_n_d_embed = layerwise_n_d_embed
        if n_d_embed_enable:
            self.n_d_embed = nn.Embedding(2, self.C)
            if layerwise_n_d_embed:
                self.layerwise_n_d_embed = nn.ModuleList([
                    nn.Linear(self.C, self.C) for _ in range(depth + 1)
                ])
            
        self.final_layer_conti = FinalLayer(self.C, self.Cvae)
    
    def save_fhat(self, f_hat, path):
        # for debug
        img = self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)
        save_image(img[0], path)
    
    def forward(self, label_B: torch.LongTensor, x_BLCv_input: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B (B,)
        :param x_BLCv_input: intermediate value on the trajectory (B, L + 1, self.Cvae)
        :param t: the timestep of the trajectory (B,), can only be an integer between 0 and len(self.patch_nums) - 1
        :return: x_BLCv_output: the output image (B, L, self.Cvae)
        """
        noise_idx = self.L - self.accum_token_length[t.int().cpu()+1]
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_input.shape[0]
        
        # get global embeddings
        with torch.cuda.amp.autocast(enabled=False):
            sos = cond_BD = self.class_emb(label_B)
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
            x_BLCv_embed = self.word_embed(x_BLCv_input.float())
            if self.prog_si == 0: x_BLC = sos
            else: x_BLC = torch.cat((sos, x_BLCv_embed), dim=1)
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # lvl: BLC;  pos: 1LC
            temp_mask = torch.arange(self.L).unsqueeze(0).expand(B, self.L) < noise_idx.unsqueeze(1)
            n_d_emb = self.n_d_embed(temp_mask.int().to(label_B.device))
            x_BLC += n_d_emb
                
        # get attn mask
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]

        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
                
        AdaLNSelfAttn.forward
        for i, b in enumerate(self.blocks):
                
            if self.layerwise_n_d_embed:
                cur_n_d_emb = self.layerwise_n_d_embed[i](n_d_emb)
            else:
                cur_n_d_emb = n_d_emb
                    
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias, n_d_emb=cur_n_d_emb)
                    
        if not self.n_d_embed_enable:
            n_d_emb = None

        # get final x_BLV
        x_BLV = self.get_logits(x_BLC.float(), cond_BD, n_d_emb=n_d_emb)
        
        # get final x_BLCv
        x_BLCv = self.get_x_BLCv(x_BLC, cond_BD, n_d_emb=n_d_emb)
        
        return x_BLCv, x_BLV
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], 
                   cond_BD: Optional[torch.Tensor], n_d_emb=None):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
            
        # for n d emb
        if self.layerwise_n_d_embed:
            n_d_emb = self.layerwise_n_d_embed[-1](n_d_emb)
        if n_d_emb is not None:
            h = h.float() + n_d_emb
            
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()[:, 1:]
    
    def get_x_BLCv(self, h_or_h_and_residual, cond_BD, n_d_emb=None):
        '''
        :param h_or_h_and_residual: h is the output of the decoder, (B, L, C). Residual has the same shape
        :param cond_BD: embedding of the condition
        :return: x_BLCv: output of the whole model
        '''
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
            
        # for n d emb
        if self.layerwise_n_d_embed and n_d_emb is not None:
            n_d_emb = self.layerwise_n_d_embed[-1](n_d_emb)
        if n_d_emb is not None:
            h = h+ n_d_emb
            
        return self.final_layer_conti(h, cond_BD)[:, 1:]
    
    def final_predict(self, label_B: torch.LongTensor, x_BLCv_input: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        '''
        :param label_B: label_B (B,)
        :param x_BLCv_input: intermediate value on the trajectory (B, L + 1, self.Cvae)
        :param t: the timestep of the trajectory (B,), can only be an integer between 0 and len(self.patch_nums) - 1
        :return: x_BLV_output: the final logits (B, L, V), x_BLCv_output: the final predicted token (B, L, C)
        '''
        B, L, C = x_BLCv_input.shape
        
        # get the model output
        # shape: B, L-1 (exclude the first class token), V 
        final_output = self.forward(label_B, x_BLCv_input, t) 
        
        final_conti, final_logits = final_output
        
        # logits process
        codebook = self.vae_quant_proxy[0].embedding.weight.data
        idx = (codebook[:, None, None, :] - x_BLCv_input).square().sum(dim=-1).argmin(dim=0)
        input_logits = torch.full((B, L, codebook.shape[0]), float('-inf')).to(label_B.device).scatter_(2, idx.unsqueeze(2), 0)
        
        # replace the ungenerated part of input with model output
        temp_mask = (torch.arange(L).expand(B, L) >= (L - self.accum_token_length[1:][t.int()]).unsqueeze(1))
        final_predict_logits = torch.zeros_like(final_logits)
        final_predict_logits[temp_mask] = final_logits[temp_mask]
        final_predict_logits[~temp_mask] = input_logits[~temp_mask]   
        
        # replace the ungenerated part of input with model output
        temp_mask = (torch.arange(L).expand(B, L) >= (L - self.accum_token_length[1:][t.int()]).unsqueeze(1)).float().unsqueeze(-1).to(final_conti)
        final_predict_conti = final_conti * temp_mask + x_BLCv_input * (1 - temp_mask)
        
        return final_predict_conti, final_predict_logits
    
    @torch.no_grad()
    def init_with_teacher(self, teacher_var: VAR, init_std: float=0.02):
        '''
        We have to deal with the following model parameters:
        1. self.word_emb: load from the teacher
        2. self.class_emb: load from the teacher
        3. self.pos_start: load from the teacher
        4. self.lvl_embed: add an extra row for the level of 1*1 pixel and load the rest from the teacher
        5. self.pos_1LC: add an extra row for the 1*1 pixel and load the rest from the teacher
        6. self.blocks: load from the teacher
        7. self.shared_ada_lin: load from the teacher
        8. self.head_nm, self.head: load from the teacher
        9. self.n_d_embed: randomly initialized
        '''
        
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5
        
        # 1. self.word_emb
        self.word_embed.weight.data.copy_(teacher_var.word_embed.weight.data)
        self.word_embed.bias.data.copy_(teacher_var.word_embed.bias.data)
        
        # 2. self.class_emb
        self.class_emb.weight.data.copy_(teacher_var.class_emb.weight.data)
        
        # 3. self.pos_start
        self.pos_start.data.copy_(teacher_var.pos_start.data)
        
        # 4. self.lvl_embed
        assert self.lvl_embed.weight.data.shape[0] == teacher_var.lvl_embed.weight.data.shape[0] + 1
        self.lvl_embed.weight.data[0].copy_(teacher_var.lvl_embed.weight.data[0])
        self.lvl_embed.weight.data[2:].copy_(teacher_var.lvl_embed.weight.data[1:])
        nn.init.trunc_normal_(self.class_emb.weight.data[1], mean=0, std=init_std)
        
        # 5. self.pos_1LC
        assert self.pos_1LC.data.shape[1] == teacher_var.pos_1LC.data.shape[1] + 1
        self.pos_1LC.data[:, 0].copy_(teacher_var.pos_1LC.data[:, 0])
        self.pos_1LC.data[:, 2:].copy_(teacher_var.pos_1LC.data[:, 1:])
        nn.init.trunc_normal_(self.pos_1LC.data[:, 1], mean=0, std=init_std)
        
        # 6. self.blocks
        for dst, src in zip(self.blocks.parameters(), teacher_var.blocks.parameters()):
            dst.data.copy_(src.data)
            
        # 7. self.shared_ada_lin
        for dst, src in zip(self.shared_ada_lin.parameters(), teacher_var.shared_ada_lin.parameters()):
            dst.data.copy_(src.data)
            
        # 8. classification head
        for dst, src in zip(self.head_nm.parameters(), teacher_var.head_nm.parameters()):
            dst.data.copy_(src.data)
        for dst, src in zip(self.head.parameters(), teacher_var.head.parameters()):
            dst.data.copy_(src.data)
            
        # 9. self.n_d_embed
        nn.init.trunc_normal_(self.n_d_embed.weight.data, mean=0, std=init_std)
        
    def perturb_data(self, x: torch.Tensor, t: torch.Tensor, noise=None):
        if noise is None:
            noise = torch.randn_like(x)
        mask = self.noise_mask[t.int()].to(x)
        x_t = x * (1 - mask[:, :, None]) + noise * mask[:, :, None]
        return x_t
        
    def greedy_sampling(self, logits):
        """
        Param logits: a tensor with shape [B, L, V]
        Return result: a sequence with each space being the token with the highest probability
        """
        codebook = self.vae_quant_proxy[0].embedding.weight.data
        max_idx = torch.argmax(logits, dim=-1)
        sequence = codebook[max_idx]
        
        return sequence
    
    def get_final_sequence(self, tokens, logits, split_t):
        """
        Param tokens: token prediction results with shape [B, L, C]
        Param logits: logits prediction results with shape [B, L, V]
        Param split_t: the position of concatenation for the two prediction results.
        """
        with torch.no_grad():
            logits_conti = self.greedy_sampling(logits)
        concat_pos = self.L - torch.tensor(self.accum_token_length[1:])[split_t] - 1
        final = torch.cat([logits_conti[:, :concat_pos], tokens[:, concat_pos:]], dim=1)
        return final
        
        
        
        