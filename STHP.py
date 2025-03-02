import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from spike_pos import PosEmbeds
from GPD import GPD
from spikingjelly.activation_based import functional
import pytorch_lightning as pl
import numpy as np
from typing import Dict, Tuple
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis  

all = ['STHP']


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., embed_dims=384):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_linear = nn.Linear(in_features, hidden_features)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = neuron.LIFNode(tau=2.0, detach_reset=True, step_mode='m')


        self.fc2_linear = nn.Linear(hidden_features, out_features)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = neuron.LIFNode(tau=2.0, detach_reset=True, step_mode='m')
        self.c_hidden = hidden_features
        self.c_output = out_features

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = neuron.LIFNode(tau=2.0, detach_reset=True, step_mode='m')

    def forward(self, x):
        T,B,N,C = x.shape
        x_ = x.flatten(0, 1)
        x = self.fc1_linear(x_)
        x = self.fc1_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.c_hidden).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_linear(x.flatten(0,1))
        x = self.fc2_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        x = self.fc2_lif(x)

        x = self.rpe_conv(x)
        x = self.rpe_bn(x)
        x = self.rpe_lif(x)
        return x
    
class SSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.q_linear = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = neuron.LIFNode(tau=2.0, detach_reset=True, step_mode='m')


        self.k_linear = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = neuron.LIFNode(tau=2.0, detach_reset=True, step_mode='m')

        self.v_linear = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = neuron.LIFNode(tau=2.0, detach_reset=True, step_mode='m')
        self.attn_lif = neuron.LIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, step_mode='m')

        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = neuron.LIFNode(tau=2.0, detach_reset=True, step_mode='m')

    def forward(self, x):
        T,B,N,C = x.shape
        x_for_qkv = x.flatten(0, 1)
        q_linear_out = self.q_linear(x_for_qkv)
        q_linear_out = self.q_bn(q_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        q_linear_out = self.q_lif(q_linear_out)
        q = q_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_linear_out = self.k_linear(x_for_qkv)
        k_linear_out = self.k_bn(k_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        k_linear_out = self.k_lif(k_linear_out)
        k = k_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = self.v_bn(v_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        v_linear_out = self.v_lif(v_linear_out)
        v = v_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = attn @ v
        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_lif(self.proj_bn(self.proj_linear(x).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C))
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
    drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1, embed_dims=384):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
        attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, embed_dims=embed_dims)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class CML(nn.Module):
    def __init__(self, img_size_h=64, img_size_w=64, patch_size=2, in_channels=2, embed_dims=384):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.block1_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1_bn = nn.BatchNorm2d(embed_dims // 8)
        self.block1_lif = neuron.LIFNode(tau=2.0, detach_reset=True, step_mode='m')

        self.block2_conv = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.block2_bn = nn.BatchNorm2d(embed_dims // 4)
        self.block2_lif = neuron.LIFNode(tau=2.0, detach_reset=True, step_mode='m')

        self.block3_conv = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.block3_bn = nn.BatchNorm2d(embed_dims // 2)
        self.block3_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.block3_lif = neuron.LIFNode(tau=2.0, detach_reset=True, step_mode='m')

        self.block4_conv = nn.Conv2d(embed_dims // 2, embed_dims // 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.block4_bn = nn.BatchNorm2d(embed_dims // 1)
        self.block4_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.block4_lif = neuron.LIFNode(tau=2.0, detach_reset=True, step_mode='m')

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = neuron.LIFNode(tau=2.0, detach_reset=True, step_mode='m')
        
    def forward(self, x):
        T, B, C, H, W = x.shape
 
        x = self.block1_conv(x.flatten(0, 1))
        x = self.block1_bn(x).reshape(T, B, -1, H, W)
        x = self.block1_lif(x).flatten(0, 1)

        x = self.block2_conv(x)
        x = self.block2_bn(x).reshape(T, B, -1, H, W)
        x = self.block2_lif(x).flatten(0, 1)

        x = self.block3_conv(x)
        x = self.block3_bn(x)
        x = self.block3_mp(x).reshape(T, B, -1, int(H / 2), int(W / 2))
        x = self.block3_lif(x).flatten(0, 1)

        x = self.block4_conv(x)
        x = self.block4_bn(x)
        x = self.block4_mp(x).reshape(T, B, -1, int(H / 4), int(W / 4))
        x = self.block4_lif(x).flatten(0, 1)

        x_feat = x.reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.rpe_conv(x)
        x = self.rpe_bn(x).reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.rpe_lif(x)
        x = x + x_feat

        x = x.flatten(-2).transpose(-1, -2)
        return x


class STHP(nn.Module):
    def __init__(self,
        img_size_h=64, img_size_w=64, patch_size=8, in_channels=2, out_channels=1,
        embed_dims=384, num_heads=8, mlp_ratios=4, qkv_bias=False, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
        depths=4, sr_ratios=[8, 4, 2], T = 1, pretrained_cfg=None, cnn_dropout=0.15, upsample_step=2,downsample_step=2
    ):
        super().__init__()
        self.T = T 
        self.out_channels = out_channels
        self.depths = depths
        self.embed_dims = embed_dims
        self.automatic_optimization = False


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]

        patch_embed = CML(img_size_h=img_size_h,
                                img_size_w=img_size_w,
                                patch_size=patch_size,
                                in_channels=in_channels,
                                embed_dims=embed_dims)
        

        block = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios, embed_dims=embed_dims)
            for j in range(depths)])
        
        self.pos2 = PosEmbeds(embed_dims, (img_size_h // 2 ** downsample_step, img_size_w // 2 ** downsample_step))


        self.decoder = GPD(embed_dims, out_channels, upsample_step, cnn_dropout)

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

        self.apply(self._init_weights)
    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        print("_get_pos_embed start")
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):

        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")
        x = patch_embed(x)
        for blk in block:
            x = blk(x)
        return x

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[3], 16, 16)
        x = self.decoder(x)
        return x
@register_model
def STHP(pretrained=False, **kwargs):
    model = STHP(
    **kwargs
    )
    model.default_cfg = _cfg()
    return model


