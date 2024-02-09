import torch
import torch.nn.functional as F
import numpy as np
from torch import nn, einsum
from einops.layers.torch import Rearrange
from einops import rearrange
from functools import reduce
from operator import mul


class PatchEmbedding3D(nn.Module):
    """
    Implements a 3D patch embedding module for volumetric data.

    This module takes an input tensor with dimensions (B, T, Z, W, H, C) and 
    transforms it into a tensor of shape (B, T, N, D), where:
    - B is the batch size,
    - T is the number of time steps,
    - Z, W, H are the dimensions of the 3D data (depth, width, height),
    - C is the number of channels,
    - N is the number of patches, and
    - D is the dimension of the embedding.
    """

    def __init__(self, image_size, patch_size, dim):
        super().__init__()
        Z, W, H, C = image_size
        patch_z, patch_w, patch_h = patch_size

        # Ensure that the image dimensions are divisible by the patch size
        assert Z % patch_z == 0 and W % patch_w == 0 and H % patch_h == 0, \
            'Image dimensions must be divisible by the patch size'

        # Calculate the number of patches and the dimension of each patch
        num_patches = (Z // patch_z) * (W // patch_w) * (H // patch_h)
        patch_dim = C * patch_z * patch_w * patch_h

        # Define the transformation to convert input data to patch embeddings
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t (z pz) (w pw) (h ph) c -> b t (z w h) (pz pw ph c)',
                      pz=patch_z, pw=patch_w, ph=patch_h),
            nn.Linear(patch_dim, dim),
        )

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(
            1, num_patches, dim))  # TODO implement RoPE embedding

    def forward(self, img):
        # Apply the patch embedding transformation
        x = self.to_patch_embedding(img)
        b, t, n, _ = x.shape

        # Add the positional embedding to the patch embeddings
        x += self.pos_embedding[:, :n]

        return x


class PatchEmbedding2D(nn.Module):
    """
    Implements a 2D patch embedding module for surface data.

    This module takes an input tensor with dimensions (B, T, W, H, C) and 
    transforms it into a tensor of shape (B, T, N, D), where:
    - B is the batch size,
    - T is the number of time steps,
    - W, H are the width and height of the image,
    - C is the number of channels,
    - N is the number of patches, and
    - D is the dimension of the embedding.
    """

    def __init__(self, image_size, patch_size, dim):
        super().__init__()
        W, H, C = image_size
        patch_w, patch_h = patch_size

        # Ensure that the image dimensions are divisible by the patch size
        assert W % patch_w == 0 and H % patch_h == 0, \
            'Image dimensions must be divisible by the patch size'

        # Calculate the number of patches and the dimension of each patch
        num_patches = (W // patch_w) * (H // patch_h)
        patch_dim = C * patch_w * patch_h

        # Define the transformation to convert input data to patch embeddings
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t (w pw) (h ph) c -> b t (w h) (pw ph c)',
                      pw=patch_w, ph=patch_h),
            nn.Linear(patch_dim, dim),
        )

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(
            1, num_patches, dim))  # TODO implement RoPE embedding

    def forward(self, img):
        # Apply the patch embedding transformation
        x = self.to_patch_embedding(img)
        b, t, n, _ = x.shape

        # Add the positional embedding to the patch embeddings
        x += self.pos_embedding[:, :n]

        return x


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

    def forward(self, x, mask=None):
        # Apply normalization before the transformer module
        # If the module is Attention, it also handles the mask
        if isinstance(self.fn, Attention):
            return self.fn(self.norm(x), mask=mask)
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

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # Apply mask if provided, for selective attention
        if mask is not None:
            dots.masked_fill_(mask == 0, np.float16(np.NINF))

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
                PreNorm(dim, Attention(dim, heads=heads,
                        dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask) + x  # Apply attention and add residual
            x = ff(x) + x          # Apply feed-forward and add residual
        return self.norm(x)        # Apply final layer normalization


class ViViT(nn.Module):
    """
    ViViT model, a transformer-based model designed for processing both 
    3D and 2D data in a unified framework.

    The model consists of separate patch embeddings for 3D and 2D data, 
    followed by spatial and temporal transformers for feature extraction 
    and aggregation.
    """

    def __init__(self, image_size_3d, patch_size_3d, image_size_2d, patch_size_2d, seq_len, output_dim, dim=192, depth=4, heads=3, dim_head=64, dropout=0., emb_dropout=0., reconstr_dropout=0., scale_dim=4):
        super().__init__()

        # Patch embedding layers for 3D and 2D data
        self.to_patch_embedding_3d = PatchEmbedding3D(
            image_size_3d, patch_size_3d, dim)
        self.to_patch_embedding_2d = PatchEmbedding2D(
            image_size_2d, patch_size_2d, dim)

        # Temporal embedding
        self.temporal_embedding = nn.Parameter(
            torch.randn(1, seq_len, 1, dim))

        # Dropout layer
        self.dropout = nn.Dropout(emb_dropout)

        # Spatial and temporal transformer layers
        self.space_transformer = Transformer(
            dim, depth, heads, dim_head, dim*scale_dim, dropout)
        self.temporal_transformer = Transformer(
            dim, depth, heads, dim_head, dim*scale_dim, dropout)

        # Calculate the number of patches and the reconstruction head input dimension
        n_patches_3d = (image_size_3d[0] // patch_size_3d[0]) * (
            image_size_3d[1] // patch_size_3d[1]) * (image_size_3d[2] // patch_size_3d[2])
        n_patches_2d = (
            image_size_2d[0] // patch_size_2d[0]) * (image_size_2d[1] // patch_size_2d[1])
        total_patches = n_patches_3d + n_patches_2d
        recon_head_input = total_patches * dim

        # Reconstruction head
        intermediate_dim = recon_head_input * 2
        self.output_dim = output_dim
        self.reconstruction_head = nn.Sequential(
            nn.Linear(recon_head_input, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(reconstr_dropout),
            nn.Linear(intermediate_dim, reduce(mul, output_dim))
        )

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
        x_3d = self.to_patch_embedding_3d(x_3d)  # [B, T, n, D]
        x_2d = self.to_patch_embedding_2d(x_2d)  # [B, T, n, D]

        # Concatenate embedded 3D and 2D data
        x = torch.cat((x_3d, x_2d), dim=2)  # [B, T, N, D]

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
        mask = self.create_temporal_attention_mask(
            t, n, device=x.device)  # [T * N, T * N]
        x = rearrange(x, 'b t n d -> b (t n) d')  # [B, T * N, D]
        x = self.temporal_transformer(x, mask)  # [B, T * N, D]

        # Reconstruction to output dimensions
        x = rearrange(x, 'b (t n) d -> b t (n d)', t=t, n=n)  # [B, T, N * D]
        x = self.reconstruction_head(x)  # [B, T, output_dim]
        x = x.view(-1, t, *self.output_dim)  # [B, T, output_dim]
        # TODO implement patch recovery
        # TODO try classifying instead of regression
        return x
