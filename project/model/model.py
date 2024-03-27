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


class PatchRecovery(nn.Module):
    def __init__(self, patch_size_2d, patch_size_3d, dim):
        super().__init__()
        #! 8 y 9 son el número de variables meterológicas
        self.tconv_upper = nn.ConvTranspose3d(dim, 8, patch_size_3d, patch_size_3d) 
        self.tconv_surface = nn.ConvTranspose2d(dim, 9, patch_size_2d, patch_size_2d)

    def forward(self, x):
        # input shape: (B, 8, 360, 181, 2C)

        # torch conv layers take inputs of shape (B, C (, Z), H, W), therefore permute:
        #   (B, 8, 360, 181, 2C) -> (B, 2C, 8, 360, 181)
        x = torch.permute(x, (0,4,1,2,3))

        # Recover the air variables from [1:] slice of Z dimension:
        #   (B, 2C, 7, 360, 181) -> (B, 5, 14, 1440, 724)
        output_air = self.tconv_upper(x[:, :, 1:, :, :])

        # Recover the surface variables from [0] slice of the Z dimension:
        #   (B, 2C, 360, 181) -> (B, 4, 1440, 724)
        output_surface = self.tconv_surface(x[:, :, 0, :, :])

        # Crop the padding added in patch embedding:
        #   (B, 5, 14, 1440, 724) -> (B, 5, 13, 1440, 721)
        output_air = output_air[:, :, 1:, :, 2:-1]
        #   (B, 4, 1440, 724) -> (B, 4, 1440, 721)
        output_surface = output_surface[:, :, :, 2:-1]

        # Restore the original shape:
        #   (B, 5, 13, 1440, 721) -> (B, 13, 1440, 721, 5)
        output_air = output_air.permute(0, 2, 3, 4, 1)
        #   (B, 4, 1440, 721) -> (B, 1440, 721, 4)
        output_surface = output_surface.permute(0, 2, 3, 1)
        return output_air, output_surface


class PatchRecovery(nn.Module):
    def __init__(self, image_size_3d, image_size_2d, patch_size_3d, patch_size_2d, dim):
        super().__init__()
        self.dim = dim
        self.patch_shape_3d = [i//j for i, j in zip(image_size_3d[:3], patch_size_3d)]
        self.patch_shape_2d = [i//j for i, j in zip(image_size_2d[:2], patch_size_2d)]
        self.tconv_upper = nn.ConvTranspose3d(dim, image_size_3d[3], kernel_size=patch_size_3d, stride=patch_size_3d)
        self.tconv_surface = nn.ConvTranspose2d(dim, image_size_2d[2], kernel_size=patch_size_2d, stride=patch_size_2d)

    def forward(self, x, b, t, n_upper, n_surface):
        
        # Separa los datos de upper y surface
        x_upper = x[:, :, :n_upper, :]  # [B, T, n_upper, D]
        x_surface = x[:, :, n_upper:, :]  # [B, T, n_surface, D]

        x_upper = rearrange(x_upper, 'b t n d -> (b t) n d') # [B*T, n_upper, D]
        x_surface = rearrange(x_surface, 'b t n d -> (b t) n d') # [B*T, n_surface, D]

        x_upper = x_upper.permute(0, 2, 1)  # [B*T, D, n_upper]
        x_surface = x_surface.permute(0, 2, 1) # [B*T, D, n_surface]

        x_upper = x_upper.view(b*t, self.dim, *self.patch_shape_3d) # [B*T, D, Z//patchZ, W//patchW, H//patchH]
        x_surface = x_surface.view(b*t, self.dim, *self.patch_shape_2d) # [B*T, D, W//patchW, H//patchH]
        
        output_upper = self.tconv_upper(x_upper) # [B*T, C, Z, W, H]
        output_surface = self.tconv_surface(x_surface) # [B*T, C, W, H]

        output_upper = rearrange(output_upper, '(b t) c z w h -> b t z w h c', b=b, t=t)  # [B, T, Z, W, H, C]
        output_surface = rearrange(output_surface, '(b t) c w h -> b t w h c', b=b, t=t)  # [B, T, W, H, C]

        return output_upper, output_surface


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
        self.to_patch_embedding_3d = PatchEmbedding3D(self.image_size_3d, patch_size_3d, dim)
        self.to_patch_embedding_2d = PatchEmbedding2D(self.image_size_2d, patch_size_2d, dim)

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
        attn_mask = self.create_temporal_attention_mask(t, n, device=x.device)  # [T * N, T * N]
        x = rearrange(x, 'b t n d -> b (t n) d')  # [B, T * N, D]
        x = self.temporal_transformer(x, attn_mask)  # [B, T * N, D]

        x = rearrange(x, 'b (t n) d -> b t n d', t=t) # [B, T, N, D]

        upper_output, surface_output = self.patch_recovery(x, b, t, x_3d.shape[2], x_2d.shape[2])

        return upper_output, surface_output