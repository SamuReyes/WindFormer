import torch
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from torch.nn import MultiheadAttention
import numpy as np
from torch import nn, einsum
from einops.layers.torch import Rearrange
from einops import rearrange
from functools import reduce
from operator import mul


class ViViT(nn.Module):
    """
    ViViT model, a transformer-based model designed for processing both 
    3D and 2D data in a unified framework.

    The model consists of separate patch embeddings for 3D and 2D data, 
    followed by spatial and temporal transformers for feature extraction 
    and aggregation.
    """

    def __init__(self, image_size_3d, patch_size_3d, image_size_2d, patch_size_2d, seq_len, dim=192, depth=4, heads=3, dim_head=64, dropout=0., emb_dropout=0., reconstr_dropout=0., scale_dim=4):
        super().__init__()

        self.image_size_3d = image_size_3d
        self.image_size_2d = image_size_2d
        self.surface_size = reduce(mul, self.image_size_2d)
        self.upper_size = reduce(mul, self.image_size_3d)

        # Patch embedding layers for 3D and 2D data
        self.to_patch_embedding_3d = PatchEmbedding(self.image_size_3d, self.image_size_2d, patch_size_3d, dim)

        # Temporal embedding
        self.temporal_embedding = nn.Parameter(torch.randn(1, seq_len, 1, dim))

        # Dropout layer
        self.dropout = nn.Dropout(emb_dropout)

        # Spatial and temporal transformer layers
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        # Reconstruction head
        self.patch_recovery = PatchRecovery(image_size_3d, image_size_2d, patch_size_3d, patch_size_2d, dim)

    def create_temporal_attention_mask(self, t, n, device):
        """
        Creates a mask for the temporal transformer to enable selective attention.

        The mask is a lower triangular matrix, which allows each time step to 
        only attend to previous and current time steps, not future ones.
        """
        mask = torch.ones((t * n, t * n), device=device)
        for i in range(t):
            mask[i * n:(i + 1) * n, (i + 1) * n:] = 0  # Mask future time steps

        return mask.bool()

    def forward(self, x_3d, x_2d):
        # Embedding 3D and 2D data
        x = self.to_patch_embedding_3d(x_3d, x_2d)  # [B, T, N, D]

        b, t, n, _ = x.shape

        # Add temporal embedding
        temp_emb = self.temporal_embedding[:, :t, :, :]
        x += temp_emb

        x = self.dropout(x)

        # Spatial transformer processing
        x = rearrange(x, 'b t n d -> (b t) n d')  # [B * T, N, D]
        x = self.space_transformer(x)  # [B, T, N, D]
        x = rearrange(x, '(b t) n d -> b t n d', b=b)  # [B, T, N, D]

        # Temporal transformer processing
        attn_mask = self.create_temporal_attention_mask(t, n, device=x.device)  # [T * N, T * N]
        x = rearrange(x, 'b t n d -> b (t n) d')  # [B, T * N, D]
        x = self.temporal_transformer(x, attn_mask)  # [B, T * N, D]

        x = rearrange(x, 'b (t n) d -> b t n d', t=t)  # [B, T, N, D]

        # Patch recovery
        upper_output, surface_output = self.patch_recovery(x, b, t, self.to_patch_embedding_3d.num_patches_3d)

        return upper_output, surface_output


class PatchEmbedding(nn.Module):
    """
    Implements a patch embedding module for surface and upper data.

    This module takes an input tensor with dimensions (B, T, (,Z), W, H, C) and 
    transforms it into a tensor of shape (B, T, N, D), where:
    - B is the batch size,
    - T is the number of time steps,
    - Z, W, H are the dimensions of the data (depth, width, height),
    - C is the number of channels,
    - N is the number of patches, and
    - D is the dimension of the embedding.
    """

    def __init__(self, image_size_3d, image_size_2d, patch_size, dim):
        super().__init__()

        Z, W, H, C_3d = image_size_3d
        W, H, C_2d = image_size_2d
        patch_z, patch_w, patch_h = patch_size

        # Calculate the number of patches
        self.num_patches_3d = (Z // patch_z) * (W // patch_w) * (H // patch_h)
        self.num_patches_2d = (W // patch_w) * (H // patch_h)
        self.num_patches = self.num_patches_3d + self.num_patches_2d

        self.conv_upper = nn.Conv3d(in_channels=C_3d, out_channels=dim, kernel_size=patch_size, stride=patch_size)
        self.conv_surface = nn.Conv2d(in_channels=C_2d, out_channels=dim, kernel_size=patch_size[1:], stride=patch_size[1:])

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(
            1, self.num_patches, dim))  # TODO implement RoPE embedding

    def forward(self, x_3d, x_2d):

        b, t, z, w, h, c = x_3d.shape

        x_3d = rearrange(x_3d, 'b t z w h c -> (b t) c z w h')  # [B*T, C, Z, W, H]
        x_2d = rearrange(x_2d, 'b t w h c -> (b t) c w h')  # [B*T, C, W, H]

        x_3d = self.conv_upper(x_3d)  # [B*T, D, Z//patchZ, W//patchW, H//patchH]
        x_2d = self.conv_surface(x_2d)  # [B*T, D, W//patchW, H//patchH]

        x_3d = rearrange(x_3d, '(b t) d z w h -> b t (z w h) d', b=b, t=t)  # [B, T, N, D]
        x_2d = rearrange(x_2d, '(b t) d w h -> b t (w h) d', b=b, t=t)  # [B, T, N, D]

        x = torch.cat((x_3d, x_2d), dim=2)  # [B, T, N, D]

        # Add the positional embedding to the patch embeddings
        x += self.pos_embedding

        return x

class PatchRecovery(nn.Module):
    def __init__(self, image_size_3d, image_size_2d, patch_size_3d, patch_size_2d, dim):
        super().__init__()
        self.dim = dim

        self.patch_shape_3d = [i//j for i, j in zip(image_size_3d[:3], patch_size_3d)]
        self.patch_shape_2d = [i//j for i, j in zip(image_size_2d[:2], patch_size_2d)]

        intermediate_shape_3d = [i*2 for i in patch_size_3d]
        intermediate_shape_2d = [i*2 for i in patch_size_2d]
        intermediate_dim_3d = image_size_3d[3]*2
        intermediate_dim_2d = image_size_2d[2]*2

        self.tconv_upper = nn.ConvTranspose3d(
            dim, intermediate_dim_3d, kernel_size=intermediate_shape_3d, stride=intermediate_shape_3d)
        self.tconv_surface = nn.ConvTranspose2d(
            dim, intermediate_dim_2d, kernel_size=intermediate_shape_2d, stride=intermediate_shape_2d)

        # TODO: try dilated convolutions
        self.conv_refine_upper = nn.Conv3d(intermediate_dim_3d, image_size_3d[3], kernel_size=3, stride=2, padding=1)
        self.conv_refine_surface = nn.Conv2d(intermediate_dim_2d, image_size_2d[2], kernel_size=3, stride=2, padding=1)

    def forward(self, x, b, t, n_upper):

        x_upper = x[:, :, :n_upper, :]  # [B, T, n_upper, D]
        x_surface = x[:, :, n_upper:, :]  # [B, T, n_surface, D]

        x_upper = rearrange(x_upper, 'b t n d -> (b t) n d')  # [B*T, n_upper, D]
        x_surface = rearrange(x_surface, 'b t n d -> (b t) n d')  # [B*T, n_surface, D]

        x_upper = x_upper.permute(0, 2, 1)  # [B*T, D, n_upper]
        x_surface = x_surface.permute(0, 2, 1)  # [B*T, D, n_surface]

        x_upper = x_upper.view(b*t, self.dim, *self.patch_shape_3d)  # [B*T, D, Z//patchZ, W//patchW, H//patchH]
        x_surface = x_surface.view(b*t, self.dim, *self.patch_shape_2d)  # [B*T, D, W//patchW, H//patchH]

        output_upper = self.tconv_upper(x_upper)  # [B*T, C, Z, W, H]
        output_surface = self.tconv_surface(x_surface)   # [B*T, C, W, H]

        output_upper = self.conv_refine_upper(output_upper)  # [B*T, C, Z, W, H]
        output_surface = self.conv_refine_surface(output_surface)  # [B*T, C, W, H]

        output_upper = rearrange(output_upper, '(b t) c z w h -> b t z w h c', b=b, t=t)  # [B, T, Z, W, H, C]
        output_surface = rearrange(output_surface, '(b t) c w h -> b t w h c', b=b, t=t)  # [B, T, W, H, C]

        return output_upper, output_surface


class PreNorm(nn.Module):
    """
    Implements a pre-normalization layer for a transformer.

    This module wraps another transformer module (like Attention or FeedForward)
    and applies Layer Normalization before passing the input to the wrapped module.
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # Layer normalization
        # Wrapped transformer module (e.g., Attention or FeedForward)
        self.fn = fn

    def forward(self, x, attn_mask=None):
        # Apply normalization before the transformer module
        # If the module is Attention, it also handles the mask
        if isinstance(self.fn, Attention):
            return self.fn(self.norm(x), attn_mask=attn_mask)
        else:
            return self.fn(self.norm(x))


class FeedForward(nn.Module):
    """
    Implements a feed-forward layer for a transformer.

    Consists of two linear layers separated by a GELU activation function and 
    includes dropout for regularization.
    """

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),  # First linear layer
            nn.GELU(),                  # GELU activation
            nn.Dropout(dropout),        # Dropout for regularization
            nn.Linear(hidden_dim, dim),  # Second linear layer
            nn.Dropout(dropout)         # Another dropout layer
        )

    def forward(self, x):
        return self.net(x)  # Process input through the feed-forward network


class Attention(nn.Module):
    """
    Implements a multi-head attention layer for a transformer.

    This module can apply a mask to the attention matrix, allowing for 
    operations like masked attention in transformer models.
    """
    # TODO: use pytorch implementation of multihead attention

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5  # Scale factor for dot products

        self.attend = nn.Softmax(dim=-1)  # Softmax for attention weights
        # Linear layer to compute Q, K, V
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # Output projection layer, if needed
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, attn_mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # Apply mask if provided, for selective attention
        if attn_mask is not None:
            dots.masked_fill_(attn_mask == 0, np.float16(np.NINF))

        attn = self.attend(dots)  # Compute attention weights

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)  # Return the final output


class Transformer(nn.Module):
    """
    Implements a transformer layer for the ViViT model.

    This layer consists of multiple alternating blocks of attention and 
    feed-forward neural network modules, each preceded by pre-normalization.
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)  # Final normalization layer

        # Create transformer layers (depth number of times)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # PreNorm(dim, MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout)),
                PreNorm(dim, Attention(dim, heads=heads,
                        dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, attn_mask=None):
        for attn, ff in self.layers:
            # Apply attention and add residual
            x = attn(x, attn_mask=attn_mask) + x
            x = ff(x) + x          # Apply feed-forward and add residual
        return self.norm(x)        # Apply final layer normalization
