"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""
from datetime import timedelta, datetime

import torch
from einops import rearrange
from torch import nn

from aurora.batch import Batch
from aurora.model.fourier import (
    absolute_time_expansion,
    lead_time_expansion,
    levels_expansion,
    pos_expansion,
    scale_expansion,
)
# from aurora.model.patchembed import LevelPatchEmbed
from aurora.model.perceiver import MLP, PerceiverResampler
from aurora.model.posencoding import pos_scale_enc
from aurora.model.util import (
    check_lat_lon_dtype,
    init_weights,
)

import math
from typing import Optional
import torch.nn.functional as F
from timm.models.layers.helpers import to_2tuple
__all__ = ["Perceiver3DEncoder","LevelPatchEmbed"]

class LevelPatchEmbed(nn.Module):
    """At either the surface or at a single pressure level, maps all variables into a single
    embedding."""

    def __init__(
        self,
        var_names: tuple[str, ...],
        patch_size: int,
        embed_dim: int,
        # history_size: int = 1,
        norm_layer: Optional[nn.Module] = None,
        flatten: bool = True,
    ) -> None:
        """Initialise.

        Args:
            var_names (tuple[str, ...]): Variables to embed.
            patch_size (int): Patch size.
            embed_dim (int): Embedding dimensionality.
            history_size (int, optional): Number of history dimensions. Defaults to `1`.
            norm_layer (torch.nn.Module, optional): Normalisation layer to be applied at the very
                end. Defaults to no normalisation layer.
            flatten (bool): At the end of the forward pass, flatten the two spatial dimensions
                into a single dimension. See :meth:`LevelPatchEmbed.forward` for more details.
        """
        super().__init__()

        self.var_names = var_names
        self.kernel_size =  to_2tuple(patch_size)
        self.flatten = flatten
        self.embed_dim = embed_dim

        self.weights = nn.ParameterDict(
            {
                # Shape (C_out, C_in, H, W). `C_in = 1` here because we're embedding every
                # variable separately.
                name: nn.Parameter(torch.empty(embed_dim, 1, *self.kernel_size))
                for name in var_names
            }
        )
        self.bias = nn.Parameter(torch.empty(embed_dim))
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.init_weights()

    def init_weights(self) -> None:
        """Initialise weights."""
        # Setting `a = sqrt(5)` in kaiming_uniform is the same as initialising with
        # `uniform(-1/sqrt(k), 1/sqrt(k))`, where `k = weight.size(1) * prod(*kernel_size)`.
        # For more details, see
        #
        #   https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        #
        for weight in self.weights.values():
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        # The following initialisation is taken from
        #
        #   https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv3d
        #
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(next(iter(self.weights.values())))
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, var_names: tuple[str, ...]) -> torch.Tensor:
        """Run the embedding.

        Args:
            x (:class:`torch.Tensor`): Tensor to embed of a shape of `(B, V, H, W)`.
            var_names (tuple[str, ...]): Names of the variables in `x`. The length should be equal
                to `V`.

        Returns:
            :class:`torch.Tensor`: Embedded tensor a shape of `(B, L, D]) if flattened,
                where `L = H * W / P^2`. Otherwise, the shape is `(B, D, H', W')`.

        """
        
        B, V, H, W = x.shape
        # assert len(var_names) == V, f"{V} != {len(var_names)}."
        # # assert self.kernel_size[0] >= T, f"{T} > {self.kernel_size[0]}."
        # assert H % self.kernel_size[1] == 0, f"{H} % {self.kernel_size[0]} != 0."
        # assert W % self.kernel_size[2] == 0, f"{W} % {self.kernel_size[1]} != 0."
        # assert len(set(var_names)) == len(var_names), f"{var_names} contains duplicates."

        # Select the weights of the variables and history dimensions that are present in the batch.
        weight = torch.cat(
            [
                # (C_out, C_in, H, W)
                self.weights[name]
                for name in var_names
            ],
            dim=1,
        )
        # Adjust the stride if history is smaller than maximum.
        # stride = (T,) + self.kernel_size[1:]

        # The convolution maps (B, V, T, H, W) to (B, D, 1, H/P, W/P)
        
        proj = F.conv3d(x, weight, self.bias, stride=(23,23))
        if self.flatten:
            proj = proj.reshape(B, self.embed_dim, -1)  # (B, D, L)
            proj = proj.transpose(1, 2)  # (B, L, D)

        x = self.norm(proj)
        return x


class Perceiver3DEncoder(nn.Module):
    """Multi-scale multi-source multi-variable encoder based on the Perceiver architecture."""

    def __init__(
        self,
        surf_vars: tuple[str, ...],
        static_vars: tuple[str, ...] | None,
        # atmos_vars: tuple[str, ...] | None,
        patch_size: int = 4,
        latent_levels: int = 8,
        embed_dim: int = 1024,
        # num_heads: int = 16,
        # head_dim: int = 64,
        drop_rate: float = 0.1,
        # depth: int = 2,
        mlp_ratio: float = 4.0,
        # max_history_size: int = 1,
        # perceiver_ln_eps: float = 1e-5,
    ) -> None:
        """Initialise.

        Args:
            surf_vars (tuple[str, ...]): All supported surface-level variables.
            static_vars (tuple[str, ...], optional): All supported static variables.
            atmos_vars (tuple[str, ...], optional): All supported atmospheric variables.
            patch_size (int, optional): Patch size. Defaults to `4`.
            latent_levels (int): Number of latent pressure levels. Defaults to `8`.
            embed_dim (int, optional): Embedding dim. used in the aggregation blocks. Defaults
                to `1024`.
            num_heads (int, optional): Number of attention heads used in aggregation blocks.
                Defaults to `16`.
            head_dim (int, optional): Dimension of attention heads used in aggregation blocks.
                Defaults to `64`.
            drop_rate (float, optional): Drop out rate for input patches. Defaults to `0.1`.
            depth (int, optional): Number of Perceiver cross-attention and feed-forward blocks.
                Defaults to `2`.
            mlp_ratio (float, optional): Ratio of hidden dimensionality to embedding dimensionality
                for MLPs. Defaults to `4.0`.
            max_history_size (int, optional): Maximum number of history steps to consider. Defaults
                to `2`.
            perceiver_ln_eps (float, optional): Epsilon value for layer normalisation in the
                Perceiver. Defaults to 1e-5.
        """
        super().__init__()

        self.drop_rate = drop_rate
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # We treat the static variables as surface variables in the model.
        surf_vars = surf_vars + static_vars if static_vars is not None else surf_vars

        # Latent tokens
        assert latent_levels > 1, "At least two latent levels are required."
        self.latent_levels = latent_levels
        
        # One latent level will be used by the surface level.
        # self.atmos_latents = nn.Parameter(torch.randn(latent_levels - 1, embed_dim))

        # Learnable embedding to encode the surface level.
        self.surf_level_encoding = nn.Parameter(torch.randn(embed_dim))
        self.surf_mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout=drop_rate)
        self.surf_norm = nn.LayerNorm(embed_dim)

        # Position, scale, and time embeddings
        self.pos_embed = nn.Linear(embed_dim, embed_dim)
        self.scale_embed = nn.Linear(embed_dim, embed_dim)
        # self.lead_time_embed = nn.Linear(embed_dim, embed_dim)
        self.absolute_time_embed = nn.Linear(embed_dim, embed_dim)
        # self.atmos_levels_embed = nn.Linear(embed_dim, embed_dim)

        # Patch embeddings
        # assert max_history_size > 0, "At least one history step is required."
        
        self.surf_token_embeds = LevelPatchEmbed(
            surf_vars,
            patch_size,
            embed_dim,
        )
        
        # self.atmos_token_embeds = LevelPatchEmbed(
        #     atmos_vars,
        #     patch_size,
        #     embed_dim,
        #     max_history_size,
        # )

        # # Learnable pressure level aggregation
        # self.level_agg = PerceiverResampler(
        #     latent_dim=embed_dim,
        #     context_dim=embed_dim,
        #     depth=depth,
        #     head_dim=head_dim,
        #     num_heads=num_heads,
        #     drop=drop_rate,
        #     mlp_ratio=mlp_ratio,
        #     ln_eps=perceiver_ln_eps,
        # )

        # Drop patches after encoding.
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.apply(init_weights)

        # Initialize the latents like in the Huggingface implementation of the Perceiver:
        #
        #   https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/models/perceiver/modeling_perceiver.py#L628
        #
        # torch.nn.init.trunc_normal_(self.atmos_latents, std=0.02)
        torch.nn.init.trunc_normal_(self.surf_level_encoding, std=0.02)

    def aggregate_levels(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate pressure level information.

        Args:
            x (torch.Tensor): Tensor of shape `(B, C_A, L, D)` where `C_A` refers to the number
                of pressure levels.

        Returns:
            torch.Tensor: Tensor of shape `(B, C, L, D)` where `C` is the number of
                aggregated pressure levels.
        """
        B, _, L, _ = x.shape
        latents = self.atmos_latents.to(dtype=x.dtype)
        latents = latents.unsqueeze(1).expand(B, -1, L, -1)  # (C_A, D) to (B, C_A, L, D)

        x = torch.einsum("bcld->blcd", x)
        x = x.flatten(0, 1)  # (B * L, C_A, D)
        latents = torch.einsum("bcld->blcd", latents)
        latents = latents.flatten(0, 1)  # (B * L, C_A, D)

        x = self.level_agg(latents, x)  # (B * L, C, D)
        x = x.unflatten(dim=0, sizes=(B, L))  # (B, L, C, D)
        x = torch.einsum("blcd->bcld", x)  # (B, C, L, D)
        return x

    def forward(self, batch: Batch) -> torch.Tensor:
        """Peform encoding.

        Args:
            batch (:class:`.Batch`): Batch to encode.

        Returns:
            torch.Tensor: Encoding of shape `(B, L, D)`.
        """
        surf_vars = tuple(batch.surf_vars.keys())
        static_vars = tuple(batch.static_vars.keys())
        
    #     atmos_vars = tuple(batch.atmos_vars.keys())
    #     atmos_levels = batch.metadata.atmos_levels

        x_surf = torch.stack(tuple(batch.surf_vars.values()), dim=1)
        x_static = torch.stack(tuple(batch.static_vars.values()), dim=0)
        # x_atmos = torch.stack(tuple(batch.atmos_vars.values()), dim=2)

        # B, T, _, C, H, W = x_atmos.size()
        # assert x_surf.shape[:2] == (B, T), f"Expected shape {(B, T)}, got {x_surf.shape[:2]}."

        B, _, H, W = x_surf.shape
        
        if static_vars is None:
            assert x_static is None, "Static variables given, but not configured."
        else:
            assert x_static is not None, "Static variables not given."
            x_static = x_static.expand((B, -1, -1, -1))
            x_surf = torch.cat((x_surf, x_static), dim=1)  # (B, T, V_S + V_Static, H, W)
            surf_vars = surf_vars + static_vars

        lat, lon = batch.metadata.lat, batch.metadata.lon
        check_lat_lon_dtype(lat, lon)
        lat, lon = lat.to(dtype=torch.float32), lon.to(dtype=torch.float32)
        assert lat.shape[0] == H and lon.shape[-1] == W

        # Patch embed the surface level.
        # x_surf = rearrange(x_surf, "b v h w -> b v h w")]
        x_surf = self.surf_token_embeds(x_surf, surf_vars)  # (B, L, D)
        dtype = x_surf.dtype  # When using mixed precision, we need to keep track of the dtype.

        # Patch embed the atmospheric levels.
        # x_atmos = rearrange(x_atmos, "b t v c h w -> (b c) v t h w")
        # x_atmos = self.atmos_token_embeds(x_atmos, atmos_vars)
        # x_atmos = rearrange(x_atmos, "(b c) l d -> b c l d", b=B, c=C)

        # Add surface level encoding. This helps the model distinguish between surface and
        # atmospheric levels.
        x_surf = x_surf + self.surf_level_encoding[None, None, :].to(dtype=dtype)
        # Since the surface level is not aggregated, we add a Perceiver-like MLP only.
        x_surf = x_surf + self.surf_norm(self.surf_mlp(x_surf))

        # # Add atmospheric pressure encoding of shape (C_A, D) and subsequent embedding.
        # atmos_levels_tensor = torch.tensor(atmos_levels, device=x_atmos.device)
        # atmos_levels_encode = levels_expansion(atmos_levels_tensor, self.embed_dim).to(dtype=dtype)
        # atmos_levels_embed = self.atmos_levels_embed(atmos_levels_encode)[None, :, None, :]
        # x_atmos = x_atmos + atmos_levels_embed  # (B, C_A, L, D)

        # Aggregate over pressure levels.
        # x_atmos = self.aggregate_levels(x_atmos)  # (B, C_A, L, D) to (B, C, L, D)

        # Concatenate the surface level with the amospheric levels.
        # x = torch.cat((x_surf.unsqueeze(1), x_atmos), dim=1)
        
        x = x_surf.unsqueeze(1) # (B, 1, L, D) 

        # Add position and scale embeddings to the 3D tensor.
        pos_encode, scale_encode = pos_scale_enc(
            self.embed_dim,
            lat,
            lon,
            self.patch_size,
            pos_expansion=pos_expansion,
            scale_expansion=scale_expansion,
        )
        # Encodings are (L, D).
        pos_encode = self.pos_embed(pos_encode[None, None, :].to(dtype=dtype))
        scale_encode = self.scale_embed(scale_encode[None, None, :].to(dtype=dtype))
        x = x + pos_encode + scale_encode

        # Flatten the tokens.
        x = x.reshape(B, -1, self.embed_dim)  # (B, C + 1, L, D) to (B, L', D)

        # Add absolute time embedding.
        absolute_times_list = [t.astype('datetime64[s]').astype(datetime.datetime).timestamp() / 3600 for t in batch.metadata.time]  # Times in hours
        absolute_times = torch.tensor(absolute_times_list, dtype=torch.float32, device=x.device)
        absolute_time_encode = absolute_time_expansion(absolute_times, self.embed_dim)
        absolute_time_embed = self.absolute_time_embed(absolute_time_encode.to(dtype=dtype))
        x = x + absolute_time_embed.unsqueeze(1)  # (B, L, D) + (B, 1, D)

        x = self.pos_drop(x)
        return x


