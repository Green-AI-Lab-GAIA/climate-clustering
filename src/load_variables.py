#%%
import xarray as xr
import numpy as np 
import torch 

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from pathlib import Path
import os

project_root = Path(__file__).resolve().parent.parent
current_working_dir = Path.cwd().resolve()

if current_working_dir != project_root:
    os.chdir(project_root)

def load_brasil_surf_var(variables, start_end_dates):
    """
    Extracts and concatenates climate data for specified variables and date ranges.

    Parameters:
    - variables (str or list of str): Variable(s) to extract (e.g., 'Tmin', 'Tmax', 'pr').
    - start_end_dates (list of lists): List of [start_date, end_date] pairs in YYYYMMDD format.

    Returns:
    - torch.Tensor: Concatenated data for the variable if a single variable is provided.

    Raises:
    - FileNotFoundError: If a required file is not found in the specified path.
    """
    if not isinstance(variables, (str, list)):
        raise ValueError("`variables` must be a string or a list of strings.")
    
    if isinstance(variables, str):
        variables = [variables]
        
    vars = {}
    metadata = {}
    time_list = []
    append_time = True
    
    for var in variables:
        
        data = []
        
        for start, end in start_end_dates:
            
            file_path = f"data/raw/{var}_{start}_{end}_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc"
            
            try:
                cur_df = xr.open_dataset(file_path, engine="netcdf4")
            except FileNotFoundError:
                raise FileNotFoundError(f"File not found: {file_path}")
            
            var_name = next(iter(cur_df.data_vars))
            data.append(cur_df[var_name].values)
            
            if not len(metadata):
                metadata["lat"] = torch.from_numpy(cur_df['latitude'].values)
                metadata["lon"] = torch.from_numpy(cur_df['longitude'].values)
            
            if append_time:
                time_list.append(cur_df['time'].values)

        append_time = False
        metadata["time"] = tuple(np.concatenate(time_list, axis=0))
        
        data = np.concatenate(data, axis=0)
        
        data = data[None] # Insert a batch dimension.
       
        data =  torch.from_numpy(data)
        data = torch.flip(data,[2]) # Flip the vertical axis.
        
        vars[var_name] = data

    mask = vars[variables[0]][0][0]
    mask = torch.where(~torch.isnan(mask), 1,0)
            
    return vars, metadata
            
        

def load_era5_static_variables(variables, area=[5.3, -73.9, -33.9, -34.9],mask=None):
    """
    Loads ERA5 variables from the downloaded NetCDF files.
    Parameters:
    area (list): A list specifying the geographical bounding box in the format 
                 [north, west, south, east].
    variables (list): A list of variable names to load (e.g., ['geo', 'lsm', 'slt']).

    Returns:
    dict: A dictionary where keys are variable names and values are xarray.Dataset objects.
    """
    
    vars = {}
    for var in variables:
        
        file_path = f"data/raw/{var}.area-subset.{area[0]}.{area[3]}.{area[2]}.{area[1]}.nc"
        cur_var = xr.open_dataset(file_path, engine="netcdf4")
        
        var_name = next(iter(cur_var.variables)) #get the variable name
        cur_var =torch.from_numpy(cur_var[var_name].values[0])
         
        if mask is not None:
            cur_var = torch.where(mask == 1, cur_var, torch.nan)
           
        vars[str(var_name)] =  cur_var
        
    return vars


#%%

# Plotting the static variables
def plot_static(static_vars,lat,lon):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Define colormap for 7 classes (0–6)
    cat_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0']
    cat_cmap = ListedColormap(cat_colors)
    cat_cmap.set_bad(color='white')  # NaNs as white

    # Categorical labels for classes 0–6
    soil_labels = [
        'Coarse', 'Medium', 'Medium fine',
        'Fine', 'Very fine', 'Organic', 'Tropical organic'
    ]
    
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    # Plot 'slt' (categorical)
    masked_slt = np.ma.masked_invalid(static_vars['slt'])
    im1 = axes[0].imshow(masked_slt, cmap=cat_cmap, vmin=0, vmax=6,extent=extent)
    axes[0].set_title('Soil Type')
    cbar1 = fig.colorbar(im1, ax=axes[0], orientation='vertical', ticks=np.arange(7))
    cbar1.set_label('Soil Class')
    cbar1.ax.set_yticklabels(soil_labels)

    # Continuous colormap for 'geo'
    cont_cmap = plt.cm.viridis.copy()
    cont_cmap.set_bad(color='white')

    # Plot 'geo' (continuous)
    masked_geo = np.ma.masked_invalid(static_vars['z'])
    im2 = axes[1].imshow(masked_geo, cmap=cont_cmap,extent=extent)
    axes[1].set_title('Geopotential')
    fig.colorbar(im2, ax=axes[1], orientation='vertical').set_label('Geopotential')

    plt.tight_layout()
    plt.show()
    

def plot_random_samples(vars, metadata, num_samples=3):
    """
    Plots random samples of variables at random times.

    Parameters:
    - vars (dict): Dictionary where keys are variable names and values are tensors.
    - metadata (dict): Dictionary containing metadata such as 'lat', 'lon', and 'time'.
    - num_samples (int): Number of random samples to plot.
    """
    lat = metadata['lat']
    lon = metadata['lon']
    time = metadata['time']
    
    fig, axes = plt.subplots(len(vars), num_samples, figsize=(5 * num_samples, 5 * len(vars)))

    if len(vars) == 1:
        axes = np.expand_dims(axes, axis=0)

    random_times = np.random.choice(vars[next(iter(vars))].shape[1], num_samples, replace=False)
    for i, (var_name, data) in enumerate(vars.items()):
        
        # Calculate vmin and vmax for the variable
        vmin, vmax =np.nanmin(data),np.nanmax(data) 
        
        for j, t in enumerate(random_times):
            
            ax = axes[i, j]
            sample_data = data[0, t].numpy()
            masked_data = np.ma.masked_invalid(sample_data)

            im = ax.imshow(masked_data,
                           extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                           cmap='seismic', vmin=vmin, vmax=vmax)
            
            ax.set_title(f"{var_name} at time {str(time[t])[:10]}")
            fig.colorbar(im, ax=ax, orientation='vertical')

    plt.tight_layout()
    plt.show()

# %%

# start_end = [[19610101,19801231], 
#             [19810101,20001231],  
#             [20010101,20240320]] 

# surf_vars, metadata, mask = load_brasil_surf_var(['Tmin','Tmax','pr'], start_end)
# static_vars = load_era5_static_variables(['slt','geo'],mask=mask)

# plot_random_samples( surf_vars, metadata)
# plot_static(static_vars, metadata['lat'], metadata['lon'])