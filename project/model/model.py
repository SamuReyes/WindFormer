import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
#from timm.models.layers import DropPath, trunc_normal_

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, dim, batch_size): # patch_size = (2,4,4)
        super().__init__()

        self.conv_air = nn.Conv3d(in_channels=8, out_channels=dim, kernel_size=patch_size, stride=patch_size)
        self.conv_surface = nn.Conv2d(in_channels=7, out_channels=dim, kernel_size=patch_size[1:], stride=patch_size[1:])

    def forward(self, input_air, input_surface):
        """
        B: batch size
        Z: number of vertical levels
        H: longitude
        W: latitude
        C: number of channels (variables)

        upper-air variables:    (B, Z=9, W=20, H=20, C=8)
        surface variables:      (B, W=20, H=20, C=9)
        """

        # torch conv layers take inputs of shape (B, C, (Z), H, W), therefore permute:
        input_air = input_air.permute(0, 4, 1, 2, 3)
        input_surface = input_surface.permute(0, 3, 1, 2)
        
        # Add padding to the data
        #   (B, 8, 9, 20, 20) -> (B, 8, 11, 22, 22)
        input_air = F.pad(input_air, (1,1,1,1,1,1))
        #   (B, 7, 20, 20) -> (B, 9, 22, 22)
        input_surface = F.pad(input_surface, (1,1,1,1))

        #! AQUI NO SE HACE PATCH EMBEDDING; SE HACE UNA CONVOLUCION 3D Y 2D
        #TODO: MIRAR LA IMPLEMENTACION DE PATCH EMBEDDING DE ClimaX
        # Apply a linear projection for patches of size patch_size[0]*patch_size[1]*patch_size[2]
        #   (B, 5, 14, 1440, 724) -> (B, C, 9, 360, 181)
        input_air = self.conv_air(input_air)

        # Apply a linear projection for patches of size patch_size[1]*patch_size[2]
        #   (B, 7, 1440, 721) -> (B, C, 360, 181)
        input_surface = self.conv_surface(input_surface)

        # Concat the air and surface data in Z dimension -> (B, C, Z=8, H=360, W=181)
        x = torch.cat((input_air, torch.unsqueeze(input_surface, 2)), 2)

        # torch.premute back to shape familiar from the paper: 
        #   (B, C, Z, H, W) -> (B, Z, H, W, C)
        x = torch.permute(x, (0,2,3,4,1))

        return x