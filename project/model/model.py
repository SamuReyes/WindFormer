import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from functools import reduce
from operator import mul


class PatchEmbedding3D(nn.Module):
    """
        Patch embedding for upper data

        Input: (B, T, Z, W, H, C)
        Output: (B, T, N, D)

        B: batch size
        T: number of time steps
        Z: number of vertical levels
        W: longitude
        H: latitude
        C: number of channels (meteorological variables)

        N: number of patches
        D: dimension of the embedding
    """
    def __init__(self, image_size, patch_size, dim):
        super().__init__()
        Z, W, H, C = image_size
        patch_z, patch_w, patch_h = patch_size

        assert Z % patch_z == 0 and W % patch_w == 0 and H % patch_h == 0, 'Image dimensions must be divisible by the patch size'

        num_patches = (Z // patch_z) * (W // patch_w) * (H // patch_h)
        patch_dim = C * patch_z * patch_w * patch_h

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t (z pz) (w pw) (h ph) c -> b t (z w h) (pz pw ph c)', 
                      pz = patch_z, pw = patch_w, ph = patch_h),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim)) #TODO: implement RoPe embedding

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, t, n, _ = x.shape

        x += self.pos_embedding[:, :(n)]

        return x
    

class PatchEmbedding2D(nn.Module):
    """
        Patch embedding for surface data

        Input: (B, T, W, H, C)
        Output: (B, T, N, D)

        B: batch size
        T: number of time steps
        W: width
        H: height
        C: number of channels

        N: number of patches
        D: dimension of the embedding
    """
    def __init__(self, image_size, patch_size, dim):
        super().__init__()
        W, H, C = image_size
        patch_w, patch_h = patch_size

        assert W % patch_w == 0 and H % patch_h == 0, 'Image dimensions must be divisible by the patch size'

        num_patches = (W // patch_w) * (H // patch_h)
        patch_dim = C * patch_w * patch_h

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t (w pw) (h ph) c -> b t (w h) (pw ph c)', 
                      pw = patch_w, ph = patch_h),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim)) #TODO: implement RoPe embedding

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, t, n, _ = x.shape

        x += self.pos_embedding[:, :(n)]

        return x

class PreNorm(nn.Module):
    """
        Pre-normalization layer for the transformer
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, mask=None):
        if isinstance(self.fn, Attention):
            return self.fn(self.norm(x), mask=mask)
        else:
            return self.fn(self.norm(x))

class FeedForward(nn.Module):
    """
        Feed forward layer for the transformer:
        Consists of two linear layers with GELU activation and dropout
    """
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """
        Multi-head attention layer for the transformer.
        It can apply a mask to the attention matrix or self-attention.
    """
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # Apply mask if not None, else apply self-attention
        if mask is not None:
            dots.masked_fill_(mask == 0, float('-1e20'))

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class Transformer(nn.Module):
    """
        Transformer layer for the ViViT model.
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask) + x
            x = ff(x) + x
        return self.norm(x)

class ViViT(nn.Module):
    """
        ViViT model
    """
    def __init__(self, image_size_3d, patch_size_3d, image_size_2d, patch_size_2d, output_dim, dim=192, depth=4, heads=3, dim_head=64, dropout=0., emb_dropout=0., scale_dim=4):
        super().__init__()

        self.to_patch_embedding_3d = PatchEmbedding3D(image_size_3d, patch_size_3d, dim)
        self.to_patch_embedding_2d = PatchEmbedding2D(image_size_2d, patch_size_2d, dim)

        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)

        self.output_dim = output_dim
        self.linear = nn.Linear(dim, reduce(mul, self.output_dim))

        self.aggregation = nn.AdaptiveAvgPool1d(1)

    
    def create_temporal_attention_mask(self, t, n, device):
        """
            Creates a mask for the temporal transformer. 
            The mask is a matrix with ones in the lower triangular part (staircase form).

            Example:

            111000000
            111111000
            111111111
        """
        mask = torch.ones((t * n, t * n), device=device)
        for i in range(t):
            mask[i * n:(i + 1) * n, (i + 1) * n:] = 0

        return mask.bool()

    def forward(self, x_3d, x_2d):

        x_3d = self.to_patch_embedding_3d(x_3d) # [B, T, n, D]
        x_2d = self.to_patch_embedding_2d(x_2d) # [B, T, n, D]

        # Concatenate upper and surface data
        x = torch.cat((x_3d, x_2d), dim=2) # [B, T, N, D]

        b, t, n, _ = x.shape

        x = self.dropout(x)

        # Process the patches through the spatial transformer
        x = rearrange(x, 'b t n d -> (b t) n d') # [B * T, N, D]
        x = self.space_transformer(x) # [B, T, N, D]
        x = rearrange(x, '(b t) n d -> b t n d', b=b) # [B, T, N, D]

        # Create mask for temporal transformer
        mask = self.create_temporal_attention_mask(t, n, device=x.device) # [T * N, T * N]

        # Process through the temporal transformer
        x = rearrange(x, 'b t n d -> b (t n) d') # [B, T * N, D]
        x = self.temporal_transformer(x, mask) # [B, T * N, D]

        # Reconstruction
        x = self.linear(x) # [B, T * N, output_dim]
        x = x.transpose(1, 2) # [B, output_dim, T * N]
        x = self.aggregation(x).squeeze(2) # [B, prod(output_dim)]
        x = x.view(-1, *self.output_dim) # [B, output_dim]

        return x
