#%%
import numpy as np
import torch
from datetime import  datetime
from torch.utils.data import Dataset
from aurora import Batch, Metadata

from src.load_variables import load_brasil_surf_var, load_era5_static_variables

class SurfWeatherDataset(Dataset):
    def __init__(self, surf_vars, static_vars, surf_stats, patch_size=40, patch_stride=20,n_samples=100):

        surf_vars_values, time, mask = load_brasil_surf_var(surf_vars,n_samples=n_samples)
        static_vars_values, lat, lon  = load_era5_static_variables(static_vars,mask=mask)
        
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        #I will change that later
        self.batch = Batch(
            surf_vars=surf_vars_values,
            static_vars=static_vars_values,
            atmos_vars={},
            metadata=Metadata(
                lat=lat,
                lon=lon,
                time=[t.astype('datetime64[s]').astype(datetime).timestamp() / 3600 for t in time] ,
                atmos_levels=(0,),
            ),
        ).type(torch.float32)\
        .normalise(surf_stats=surf_stats)
        
        surf_vars  = tuple(self.batch.surf_vars.keys())
        static_vars = tuple(self.batch.static_vars.keys())
        self.surf_vars = surf_vars + static_vars + ('lat', 'lon', 'time')

        # self.data, self.lat, self.lon = self._extract_patches() 
        self.data = self._extract_patches()  
        

    def _extract_patches(self, exclude_nans=True):
        """
        Extrai patches usando unfold para divisões com ou sem sobreposição.
        """

        surf_vars = tuple(self.batch.surf_vars.keys())
        static_vars = tuple(self.batch.static_vars.keys())

        x_surf = torch.stack(tuple(self.batch.surf_vars.values()), dim=1)
        x_static = torch.stack(tuple(self.batch.static_vars.values()), dim=0)

        B, _, H, W = x_surf.shape

        x_static = x_static.expand((B, -1, -1, -1))
        x_surf = torch.cat((x_surf, x_static), dim=1)  # (B, T, V_S + V_Static, H, W)
        surf_vars = surf_vars + static_vars

        #lat_long array
        lat_array = self.batch.metadata.lat.repeat(self.batch.metadata.lon.shape[0],1).T #each column is a lat vector
        lon_array = self.batch.metadata.lon.repeat(self.batch.metadata.lat.shape[0],1) #each row is a lon vector
        lat_long = torch.stack((lat_array,lon_array), dim=0).expand((B, -1, -1, -1))

        time_array = torch.Tensor(self.batch.metadata.time).view(-1, 1, 1).expand(B, x_surf.shape[-2], x_surf.shape[-1]).unsqueeze(1)
        
        x_surf = torch.cat((x_surf, lat_long, time_array), dim=1) 

        _, V, _, _ = x_surf.shape

        patches = x_surf.unfold(2, self.patch_size, self.patch_stride).unfold(3, self.patch_size, self.patch_stride)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, V, self.patch_size, self.patch_size)

        if exclude_nans:
            patches = patches[~torch.isnan(patches).any(dim=(1, 2, 3))] 

        # lat_patches = patches[:,5,:,:].transpose(1,2)[:,0]
        # lon_patches = patches[:,6,0]
        # time_patches = patches[:,7,0,0]
        # patches = patches[:,:5,:,:] 

        return patches

    
    def __getitem__(self, idx) -> torch.Tensor:
        return self.data[idx]
    
    def __len__(self):
        return self.data.shape[0]


# %%
