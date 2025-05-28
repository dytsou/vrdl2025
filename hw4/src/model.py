# PromptIR: Prompting for All-in-One Blind Image Restoration
# Vaishnav Potlapalli, Syed Waqas Zamir, Salman Khan, and Fahad Shahbaz Khan
# https://arxiv.org/abs/2306.13090


import kornia.filters as filters  # for spatial_gradient
import torch
# print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from einops.layers.torch import Rearrange
import time

# Import configuration
from config import MODEL_CONFIG, PROMPT_CONFIG, HOG_CONFIG

# from skimage.feature import hog # Removed unused import


##########################################################################
# Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
# Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
# Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class FastHOGAwareAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 bias,
                 orientations: int = HOG_CONFIG['orientations'],
                 eps: float = HOG_CONFIG['eps']):
        """
        dim          : number of channels in QKV
        num_heads    : number of attention heads
        bias         : whether QKV convs have bias
        orientations : number of orientation bins
        """
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.orientations = orientations
        self.eps = eps

        # QKV + depth?wise conv
        self.qkv = nn.Conv2d(dim, dim*3, 1, bias=bias)
        self.qkv_dw = nn.Conv2d(dim*3, dim*3, 3, padding=1,
                                groups=dim*3, bias=bias)
        self.project_o = nn.Conv2d(dim, dim, 1, bias=bias)

        # project our approximate HOG into `dim` channels
        self.hog_proj = nn.Conv2d(orientations, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # 1) compute gradients on GPU
        #    returns shape (B, C, H, W, 2) for (dy, dx)
        grad = filters.spatial_gradient(x, mode='sobel', order=1)
        dy, dx = grad[..., 0], grad[..., 1]  # each (B, C, H, W)

        # 2) magnitude & angle per pixel
        #    collapse channel dim by summing magnitudes
        mag = torch.sqrt(dx.pow(2) + dy.pow(2) + self.eps)  # (B,C,H,W)
        mag = mag.mean(dim=1, keepdim=True)                 # (B,1,H,W)
        ang = torch.atan2(dy, dx + self.eps)                 # (B,C,H,W)
        ang = ang.mean(dim=1, keepdim=True)                  # (B,1,H,W)

        # 3) map angles from [-�k, �k] to [0, �k]
        ang = (ang % torch.pi)

        # 4) compute soft?assignment to orientation bins
        #    for each bin center �c_k, weight = max(0, 1 ? |ang?�c_k|/(�k/orientations))
        bin_width = torch.pi / self.orientations
        # shape (orientations,)
        centers = torch.arange(
            self.orientations, device=x.device).float() * bin_width + bin_width/2
        # reshape for broadcasting: (1, K, 1, 1)
        centers = centers.view(1, -1, 1, 1)
        # ang: (B,1,H,W) �� (B,K,H,W) for difference to each center
        ang = ang.expand(-1, self.orientations, -1, -1)
        diff = torch.abs(ang - centers)
        weights = F.relu(1 - diff / bin_width)  # (B,K,H,W)

        # 5) weight by magnitude
        hog_feat = weights * mag.expand(-1, self.orientations, -1, -1)

        # 6) project to dim channels
        hog_feat = self.hog_proj(hog_feat)  # (B, dim, H, W)
        hog_feat = F.interpolate(hog_feat,
                                 size=(h, w),
                                 mode='bilinear',
                                 align_corners=False)

        # --- now the usual attention path ---
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # inject our HOG into the value stream
        v = v + hog_feat

        # reshape to (B, heads, C_head, HW)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        # scaled dot?prod
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # attend
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        # final projection
        return self.project_o(out)


class resblock(nn.Module):
    def __init__(self, dim):

        super(resblock, self).__init__()
        # self.norm = LayerNorm(dim, LayerNorm_type='BiasFree')

        self.body = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PReLU(),
                                  nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        res = self.body((x))
        res += x
        return res


##########################################################################
# Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        # self.attn = Attention(dim, num_heads, bias)
        self.attn = FastHOGAwareAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
# Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
# ---------- Prompt Gen Module -----------------------
class PromptGenBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=PROMPT_CONFIG['prompt_len'], prompt_size=96, lin_dim=192):
        super(PromptGenBlock, self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(
            1, prompt_len, prompt_dim, prompt_size, prompt_size))
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(
            prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * \
            self.prompt_param.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1)
        prompt = torch.sum(prompt, dim=1)
        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt


##########################################################################
# ---------- PromptIR -----------------------

class PromptIR(nn.Module):
    def __init__(self,
                 inp_channels=MODEL_CONFIG['inp_channels'],
                 out_channels=MODEL_CONFIG['out_channels'],
                 dim=MODEL_CONFIG['base_channels'],
                 num_blocks=MODEL_CONFIG['num_blocks'],
                 num_refinement_blocks=MODEL_CONFIG['num_refinement_blocks'],
                 heads=MODEL_CONFIG['heads'],
                 ffn_expansion_factor=MODEL_CONFIG['ffn_expansion_factor'],
                 bias=MODEL_CONFIG['bias'],
                 LayerNorm_type=MODEL_CONFIG['LayerNorm_type'],
                 decoder=MODEL_CONFIG['decoder'],
                 base_channels=None,  # Added parameter for compatibility
                 prompt_dim=MODEL_CONFIG['prompt_dim'],
                 ):

        super(PromptIR, self).__init__()

        # Handle compatibility with training/visualization scripts
        if base_channels is not None:
            dim = base_channels

        # Convert num_blocks from int to list if it's an integer
        if isinstance(num_blocks, int):
            n = num_blocks
            # Distribute blocks across 4 levels: [n//4, n//4, n//4, n-3*(n//4)]
            num_blocks = [n//4, n//4, n//4, n-3*(n//4)]

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.decoder = decoder

        # Set prompt channel dimensions, use prompt_dim if provided
        if prompt_dim is not None:
            # Scale prompt dimensions proportionally
            prompt1_ch = max(1, prompt_dim)  # Level 1 prompt dimension
            prompt2_ch = max(1, prompt_dim * 2)  # Level 2 prompt dimension
            prompt3_ch = max(1, prompt_dim * 4)  # Level 3 prompt dimension
        else:
            prompt1_ch = 64
            prompt2_ch = 128
            prompt3_ch = 256  # Reduced from 320 to avoid dimension mismatch

        if self.decoder:
            self.prompt1 = PromptGenBlock(
                prompt_dim=prompt1_ch, prompt_len=PROMPT_CONFIG['prompt_len'],
                prompt_size=PROMPT_CONFIG['prompt_sizes'][0], lin_dim=int(dim*2**1))
            self.prompt2 = PromptGenBlock(
                prompt_dim=prompt2_ch, prompt_len=PROMPT_CONFIG['prompt_len'],
                prompt_size=PROMPT_CONFIG['prompt_sizes'][1], lin_dim=int(dim*2**2))
            self.prompt3 = PromptGenBlock(
                prompt_dim=prompt3_ch, prompt_len=PROMPT_CONFIG['prompt_len'],
                prompt_size=PROMPT_CONFIG['prompt_sizes'][2], lin_dim=int(dim*2**3))

        self.chnl_reduce1 = nn.Conv2d(
            prompt1_ch, prompt1_ch, kernel_size=1, bias=bias)
        self.chnl_reduce2 = nn.Conv2d(
            prompt2_ch, prompt2_ch, kernel_size=1, bias=bias)
        self.chnl_reduce3 = nn.Conv2d(
            prompt3_ch, prompt3_ch, kernel_size=1, bias=bias)

        self.reduce_noise_channel_1 = nn.Conv2d(
            dim + prompt1_ch, dim, kernel_size=1, bias=bias)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  # From Level 1 to Level 2

        self.reduce_noise_channel_2 = nn.Conv2d(
            int(dim*2**1) + prompt2_ch, int(dim*2**1), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1],
                                            ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim*2**1))  # From Level 2 to Level 3

        self.reduce_noise_channel_3 = nn.Conv2d(
            int(dim*2**2) + prompt3_ch, int(dim*2**2), kernel_size=1, bias=bias)
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2],
                                            ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2))  # From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3],
                                    ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        # Takes dim*2**2, outputs dim*2**1
        self.up4_3 = Upsample(int(dim*2**2))
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim*2**1) + int(dim*2**2), int(dim*2**2), kernel_size=1, bias=bias)

        if self.decoder:
            self.noise_level3 = TransformerBlock(dim=int(
                dim*2**3) + prompt3_ch, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            self.reduce_noise_level3 = nn.Conv2d(
                int(dim*2**3) + prompt3_ch, int(dim*2**2), kernel_size=1, bias=bias)

            self.noise_level2 = TransformerBlock(dim=int(
                dim*2**2) + prompt2_ch, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            self.reduce_noise_level2 = nn.Conv2d(
                int(dim*2**2) + prompt2_ch, int(dim*2**2), kernel_size=1, bias=bias)

            self.noise_level1 = TransformerBlock(dim=int(
                dim*2**1) + prompt1_ch, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            self.reduce_noise_level1 = nn.Conv2d(
                int(dim*2**1) + prompt1_ch, int(dim*2**1), kernel_size=1, bias=bias)
        else:
            self.noise_level3 = nn.Identity()
            self.reduce_noise_level3 = nn.Identity()
            self.noise_level2 = nn.Identity()
            self.reduce_noise_level2 = nn.Identity()
            self.noise_level1 = nn.Identity()
            self.reduce_noise_level1 = nn.Identity()

        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2],
                                            ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        # Takes dim*2**2, outputs dim*2**1
        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)

        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1],
                                            ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim*2**1))

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0],
                                            ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                        bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(
            int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, degradation_type=None):
        """
        Forward pass for PromptIR

        Args:
            inp_img: Input degraded image
            degradation_type: String indicating degradation type ('rain' or 'snow')
                              Used instead of noise_emb for API compatibility
        """
        # For backwards compatibility, allow both noise_emb and degradation_type
        # degradation_type is ignored in this implementation since the model
        # doesn't need it explicitly - the prompt generator blocks handle the features

        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)
        if self.decoder:
            dec3_param = self.prompt3(latent)

            latent = torch.cat([latent, dec3_param], 1)
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)

        inp_dec_level3 = self.up4_3(latent)

        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        if self.decoder:
            dec2_param = self.prompt2(out_dec_level3)
            out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        if self.decoder:

            dec1_param = self.prompt1(out_dec_level2)
            out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)

        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1
