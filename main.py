#%%

from src.load_variables import load_brasil_surf_var, load_era5_static_variables

surf_vars, metadata, mask = load_brasil_surf_var(['Tmin','Tmax','pr'])
# static_vars = load_era5_static_variables(['slt','geo'],mask=mask)

tmax = surf_vars['Tmax'][0][:1000]

# np.save('data/tmax.npy',tmax)

   
# %%
