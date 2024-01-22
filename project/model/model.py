import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn
from einops.layers.torch import Rearrange


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
