#%%

from src.load_variables import load_brasil_surf_var, load_era5_static_variables
from aurora import Batch, Metadata
# from aurora import Aurora, AuroraHighRes
import torch

from src.encoder import Perceiver3DEncoder

surf_vars, metadata_br, mask = load_brasil_surf_var(['Tmin','Tmax','pr'])
static_vars, metadata_era5  = load_era5_static_variables(['slt','geo'],mask=mask)
   
#3 min pra rodar
# surf_stats = {
#     'Tmin': (surf_vars['Tmin'].nanmean().item(),np.nanstd(surf_vars['Tmin'])),
#     'Tmax': (surf_vars['Tmax'].nanmean().item(), np.nanstd(surf_vars['Tmax'])),
#     'pr': (surf_vars['pr'].nanmean().item(), np.nanstd(surf_vars['pr'])),
# }

surf_stats =  {'Tmin': (19.884305000538596, 3.5784525624804515),
 'Tmax': (30.66388006868657, 3.613170686518841),
 'pr': (4.803210990399181, 8.075623155365308)}

batch = Batch(
    surf_vars=surf_vars,
    static_vars=static_vars,
    atmos_vars={},
    metadata=Metadata(
        lat=metadata_era5['lat'],
        lon=metadata_era5['lon'],
        time=metadata_br['time'],
        atmos_levels=(0,),
    ),
)

PATCH_SIZE = 23

batch = batch.type(torch.float32)
batch = batch.normalise(surf_stats=surf_stats)
batch = batch.crop(patch_size=PATCH_SIZE)


encoder = Perceiver3DEncoder(
    surf_vars=list(surf_vars.keys()),
    static_vars=list(static_vars.keys()),
    patch_size = PATCH_SIZE
)

encoder.forward(batch)


# %%
