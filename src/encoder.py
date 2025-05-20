#%%
"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""
from datetime import  datetime

import torch
from torch import nn

from aurora.batch import Batch
from aurora.model.fourier import (
    absolute_time_expansion,
    pos_expansion,
    scale_expansion,
)
# from aurora.model.patchembed import LevelPatchEmbed
from aurora.model.perceiver import MLP
from aurora.model.posencoding import pos_scale_enc
from aurora.model.util import (
    check_lat_lon_dtype,
    init_weights,
)

import math
from typing import Optional
import torch.nn.functional as F
from timm.models.layers.helpers import to_2tuple

__all__ = ["Weather3DEncoder","LevelPatchEmbed"]


class LevelPatchEmbed(nn.Module):
    """Maps all variables into a single embedding (no temporal dimension)."""

    def __init__(
        self,
        var_names: tuple[str, ...],
        patch_size: int,
        embed_dim: int,
        norm_layer: Optional[nn.Module] = None,
        flatten: bool = True,
    ) -> None:
        """Initialise.

        Args:
            var_names (tuple[str, ...]): Variables to embed.
            patch_size (int): Patch size.
            embed_dim (int): Embedding dimensionality.
            norm_layer (torch.nn.Module, optional): Normalisation layer. Defaults to no normalisation layer.
            flatten (bool): If True, flattens the spatial dimensions at the end.
        """
        super().__init__()

        self.var_names = var_names
        self.kernel_size = to_2tuple(patch_size)
        self.flatten = flatten
        self.embed_dim = embed_dim

        self.weights = nn.ParameterDict({
            # Shape (C_out, C_in=1, H, W)
            name: nn.Parameter(torch.empty(embed_dim, 1, *self.kernel_size))
            for name in var_names
        })
        self.bias = nn.Parameter(torch.empty(embed_dim))
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.init_weights()

    def init_weights(self) -> None:
        """Initialise weights and bias."""
        for weight in self.weights.values():
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(next(iter(self.weights.values())))
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, var_names: tuple[str, ...]) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Shape (B, V, H, W)
            var_names (tuple[str, ...]): Names of the variables in x (length V)

        Returns:
            torch.Tensor: If flatten=True, shape (B, L, D). Else, shape (B, D, H', W')
        """
        B, V, H, W = x.shape
        assert len(var_names) == V, f"{V} != {len(var_names)}"
        assert H % self.kernel_size[0] == 0, f"{H} % {self.kernel_size[0]} != 0"
        assert W % self.kernel_size[1] == 0, f"{W} % {self.kernel_size[1]} != 0"
        assert len(set(var_names)) == len(var_names), f"{var_names} contains duplicates"

        weight = torch.cat(
            [self.weights[name] for name in var_names], dim=1  # concat along channel dim (C_in)
        )
        
        stride = self.kernel_size

        # (B, V, H, W) → (B, D, H', W')
        proj = F.conv2d(x, weight, self.bias, stride=stride)

        if self.flatten:
            proj = proj.reshape(B, self.embed_dim, -1)  # (B, D, L)
            proj = proj.transpose(1, 2)  # (B, L, D)

        x = self.norm(proj)
        return x


class Weather3DEncoder(nn.Module):
    """Multi-scale multi-source multi-variable encoder based on the Perceiver architecture."""

    def __init__(
        self,
        surf_vars: tuple[str, ...],
        patch_size: int = 4,
        embed_dim: int = 1024,
        drop_rate: float = 0.1,
        mlp_ratio: float = 4.0,
    ) -> None:
        """Initialise.

        Args:
            surf_vars (tuple[str, ...]): All supported surface-level variables.
            static_vars (tuple[str, ...], optional): All supported static variables.
            patch_size (int, optional): Stride of embedding. Defaults to `4`.
            embed_dim (int, optional): Embedding dim. used in the aggregation blocks. Defaults
                to `1024`.
            drop_rate (float, optional): Drop out rate for input patches. Defaults to `0.1`.
            mlp_ratio (float, optional): Ratio of hidden dimensionality to embedding dimensionality
                for MLPs. Defaults to `4.0`.

        """
        super().__init__()

        self.drop_rate = drop_rate
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # We treat the static variables as surface variables in the model.
        # surf_vars = surf_vars + static_vars if static_vars is not None else surf_vars
        self.surf_vars = surf_vars[:-3] # Exclude lat, lon, time

        # Learnable embedding to encode the surface level.
        self.surf_level_encoding = nn.Parameter(torch.randn(embed_dim))
        self.surf_mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout=drop_rate)
        self.surf_norm = nn.LayerNorm(embed_dim)

        # Position, scale, and time embeddings
        self.pos_embed = nn.Linear(embed_dim, embed_dim)
        self.scale_embed = nn.Linear(embed_dim, embed_dim)
        self.absolute_time_embed = nn.Linear(embed_dim, embed_dim)

        # Patch embeddings

        self.surf_token_embeds = LevelPatchEmbed(
            self.surf_vars,  
            patch_size,
            embed_dim,
        )
        
        # Drop patches after encoding.
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.apply(init_weights)

        # Initialize the latents like in the Huggingface implementation of the Perceiver:
        #
        #   https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/models/perceiver/modeling_perceiver.py#L628
        #
        torch.nn.init.trunc_normal_(self.surf_level_encoding, std=0.02)


    def forward(self, x_surf, lat=None, lon=None) -> torch.Tensor:
        """Peform encoding.

        Args:
            batch (:class:`.Batch`): Batch to encode.

        Returns:
            torch.Tensor: Encoding of shape `(B, L, D)`.
        """

        B, _, H, W = x_surf.shape

        if x_surf.shape[1] == 8:
            lat = x_surf[:,5,:,:].transpose(1,2)[:,0]
            lon = x_surf[:,6,0]
            time = x_surf[:,7,0,0]
            x_surf = x_surf[:,:5,:,:] 
        else:
            lat, lon, time = None, None, None

        # Patch embed the surface level.
        x_surf = self.surf_token_embeds(x_surf, self.surf_vars)  # (B, L, D)
        dtype = x_surf.dtype  # When using mixed precision, we need to keep track of the dtype.

        # Add surface level encoding. 
        # Since the surface level is not aggregated, we add a Perceiver-like MLP only.
        x_surf = x_surf + self.surf_level_encoding[None, None, :].to(dtype=dtype)
        x_surf = x_surf + self.surf_norm(self.surf_mlp(x_surf))

        x = x_surf.unsqueeze(1) # (B, 1, L, D) 


        # Add position and scale embeddings to the 3D tensor.
        # Compute pos_encode and scale_encode for each sample in the batch
        if lat is not None or lon is not None:
            pos_encodes = []
            scale_encodes = []

            for b in range(B):

                pos_encode, scale_encode = pos_scale_enc(
                self.embed_dim,
                lat[b],
                lon[b],
                self.patch_size,
                pos_expansion=pos_expansion,
                scale_expansion=scale_expansion,
                )
                pos_encodes.append(pos_encode)
                scale_encodes.append(scale_encode)

            pos_encode = torch.stack(pos_encodes, dim=0)  # (B, L, D)
            scale_encode = torch.stack(scale_encodes, dim=0)  # (B, L, D)

            # Encodings are (L, D).
            pos_encode = self.pos_embed(pos_encode[:, None, :].to(dtype=dtype))
            scale_encode = self.scale_embed(scale_encode[:, None, :].to(dtype=dtype))
            x = x + pos_encode + scale_encode

        # Flatten the tokens.
        x = x.reshape(B, -1, self.embed_dim)  # (B, C + 1, L, D) to (B, L', D)

        if time is not None:

            # Add absolute time embedding.
            # absolute_times_list = [t.item().astype('datetime64[s]').astype(datetime).timestamp() / 3600 for t in time]  # Times in hours
            absolute_times = torch.tensor(time, dtype=torch.float32, device=x.device)
            absolute_time_encode = absolute_time_expansion(absolute_times, self.embed_dim)
            absolute_time_embed = self.absolute_time_embed(absolute_time_encode.to(dtype=dtype))
            x = x + absolute_time_embed.unsqueeze(1)  # (B, L, D) + (B, 1, D)

        x = self.pos_drop(x)

        #temporary fix for 3d encoder (SWAV)
        x= x.squeeze(1)\
            .permute(0, 2, 1)
        
        x = nn.AdaptiveAvgPool1d(1)(x) 
        x = x.squeeze(-1)
        
        return x


