"""
PromptIR: Prompting for All-in-One Blind Image Restoration (NeurIPS 2023)
Adapted for Rain/Snow restoration task.

Modifications from baseline:
1. Channel Attention (SE block) in each Transformer block
2. FFT frequency loss support (used in train.py)
3. Prompt length increased from 5 to 10
4. Learnable residual scaling per block
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(normalized_shape))
    def forward(self, x):
        mu    = x.mean(-1, keepdim=True)
        sigma = x.var(-1,  keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super().__init__()
        self.body = BiasFree_LayerNorm(dim) if LayerNorm_type == 'BiasFree' else WithBias_LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in  = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv      = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads   = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv         = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv  = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out  = rearrange(attn @ v, 'b head c (h w) -> b (head c) h w', h=h, w=w)
        return self.project_out(out)

class ChannelAttention(nn.Module):
    """Modification #1: SE-block style channel attention."""
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, max(1, dim // reduction), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, dim // reduction), dim, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return x * self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.66, bias=False,
                 LayerNorm_type='WithBias', use_ca=True):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn  = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn   = FeedForward(dim, ffn_expansion_factor, bias)
        self.ca    = ChannelAttention(dim) if use_ca else nn.Identity()
        self.scale = nn.Parameter(torch.ones(1) * 0.1)  # Modification #4
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.scale * self.ca(self.ffn(self.norm2(x)))
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 3, 1, 1, bias=bias)
    def forward(self, x):
        return self.proj(x)

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1, bias=False),
            nn.PixelUnshuffle(2),
        )
    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, 3, 1, 1, bias=False),
            nn.PixelShuffle(2),
        )
    def forward(self, x):
        return self.body(x)

class PromptBlock(nn.Module):
    """Modification #3: prompt_len=10 for richer degradation encoding."""
    def __init__(self, prompt_dim=128, prompt_len=10, prompt_size=96, lin_dim=192):
        super().__init__()
        self.prompt_param = nn.Parameter(
            torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size)
        )
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3      = nn.Conv2d(prompt_dim, prompt_dim, 3, 1, 1, bias=False)
    def forward(self, x):
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        w   = F.softmax(self.linear_layer(emb), dim=1)  # (B, prompt_len)
        p   = (w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param).sum(1)
        p   = F.interpolate(p, (H, W), mode='bilinear', align_corners=False)
        return self.conv3x3(p)


class PromptIR(nn.Module):
    """
    PromptIR with 4 modifications:
      #1 Channel Attention per block
      #2 FFT loss (external, in train.py)
      #3 prompt_len = 10
      #4 learnable residual scale
    """
    def __init__(self, inp_channels=3, out_channels=3, dim=48,
                 num_blocks=(4, 6, 6, 8), num_refinement_blocks=4,
                 heads=(1, 2, 4, 8), ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type='WithBias',
                 prompt=True, prompt_len=10):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # Encoder
        self.encoder_l1 = nn.Sequential(*[TransformerBlock(dim,   heads[0], ffn_expansion_factor, bias, LayerNorm_type) for _ in range(num_blocks[0])])
        self.down12      = Downsample(dim)
        self.encoder_l2 = nn.Sequential(*[TransformerBlock(dim*2, heads[1], ffn_expansion_factor, bias, LayerNorm_type) for _ in range(num_blocks[1])])
        self.down23      = Downsample(dim*2)
        self.encoder_l3 = nn.Sequential(*[TransformerBlock(dim*4, heads[2], ffn_expansion_factor, bias, LayerNorm_type) for _ in range(num_blocks[2])])
        self.down34      = Downsample(dim*4)
        self.latent      = nn.Sequential(*[TransformerBlock(dim*8, heads[3], ffn_expansion_factor, bias, LayerNorm_type) for _ in range(num_blocks[3])])

        # Prompt injection modules
        self.prompt = prompt
        if prompt:
            self.prompt3 = PromptBlock(dim*4, prompt_len, 96, dim*4)
            self.prompt2 = PromptBlock(dim*2, prompt_len, 96, dim*2)
            self.prompt1 = PromptBlock(dim,   prompt_len, 96, dim)
            # Reduce fused (latent + upsampled_prompt) → same dim as latent
            self.fuse3   = nn.Conv2d(dim*8 + dim*4, dim*8, 1, bias=False)
            self.fuse2   = nn.Conv2d(dim*4 + dim*2, dim*4, 1, bias=False)
            self.fuse1   = nn.Conv2d(dim*2 + dim,   dim*2, 1, bias=False)
            self.nl3     = TransformerBlock(dim*8, heads[3], ffn_expansion_factor, bias, LayerNorm_type)
            self.nl2     = TransformerBlock(dim*4, heads[2], ffn_expansion_factor, bias, LayerNorm_type)
            self.nl1     = TransformerBlock(dim*2, heads[1], ffn_expansion_factor, bias, LayerNorm_type)

        # Decoder
        self.up43           = Upsample(dim*8)
        self.reduce_chan_l3 = nn.Conv2d(dim*8, dim*4, 1, bias=False)
        self.decoder_l3     = nn.Sequential(*[TransformerBlock(dim*4, heads[2], ffn_expansion_factor, bias, LayerNorm_type) for _ in range(num_blocks[2])])
        self.up32           = Upsample(dim*4)
        self.reduce_chan_l2 = nn.Conv2d(dim*4, dim*2, 1, bias=False)
        self.decoder_l2     = nn.Sequential(*[TransformerBlock(dim*2, heads[1], ffn_expansion_factor, bias, LayerNorm_type) for _ in range(num_blocks[1])])
        self.up21           = Upsample(dim*2)
        self.reduce_chan_l1 = nn.Conv2d(dim*2, dim,   1, bias=False)
        self.decoder_l1     = nn.Sequential(*[TransformerBlock(dim,   heads[0], ffn_expansion_factor, bias, LayerNorm_type) for _ in range(num_blocks[0])])
        self.refinement     = nn.Sequential(*[TransformerBlock(dim,   heads[0], ffn_expansion_factor, bias, LayerNorm_type) for _ in range(num_refinement_blocks)])
        self.output         = nn.Conv2d(dim, out_channels, 3, 1, 1, bias=False)

    def forward(self, inp_img):
        # Encode
        f1 = self.encoder_l1(self.patch_embed(inp_img))
        f2 = self.encoder_l2(self.down12(f1))
        f3 = self.encoder_l3(self.down23(f2))
        lat = self.latent(self.down34(f3))

        if self.prompt:
            # --- Prompt injection at each decoder level ---
            # Level 3 (bottleneck)
            p3  = F.interpolate(self.prompt3(f3), size=lat.shape[-2:], mode='bilinear', align_corners=False)
            lat = self.nl3(self.fuse3(torch.cat([lat, p3], dim=1)))

            lat  = self.up43(lat)
            d3   = self.decoder_l3(self.reduce_chan_l3(torch.cat([lat, f3], dim=1)) + self.prompt3(f3))

            # Level 2
            p2  = F.interpolate(self.prompt2(f2), size=d3.shape[-2:], mode='bilinear', align_corners=False)
            d3  = self.nl2(self.fuse2(torch.cat([d3, p2], dim=1)))

            d3   = self.up32(d3)
            d2   = self.decoder_l2(self.reduce_chan_l2(torch.cat([d3, f2], dim=1)) + self.prompt2(f2))

            # Level 1
            p1  = F.interpolate(self.prompt1(f1), size=d2.shape[-2:], mode='bilinear', align_corners=False)
            d2  = self.nl1(self.fuse1(torch.cat([d2, p1], dim=1)))

            d2   = self.up21(d2)
            d1   = self.decoder_l1(self.reduce_chan_l1(torch.cat([d2, f1], dim=1)) + self.prompt1(f1))
        else:
            lat = self.up43(lat)
            d3  = self.decoder_l3(self.reduce_chan_l3(torch.cat([lat, f3], dim=1)))
            d3  = self.up32(d3)
            d2  = self.decoder_l2(self.reduce_chan_l2(torch.cat([d3,  f2], dim=1)))
            d2  = self.up21(d2)
            d1  = self.decoder_l1(self.reduce_chan_l1(torch.cat([d2,  f1], dim=1)))

        return self.output(self.refinement(d1)) + inp_img  # global residual
