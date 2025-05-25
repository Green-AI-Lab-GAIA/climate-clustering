#%%
import xarray as xr
import numpy as np 
import torch 

from pathlib import Path
import os

project_root = Path(__file__).resolve().parent.parent.parent
current_working_dir = Path.cwd().resolve()

if current_working_dir != project_root:
    os.chdir(project_root)

def load_brasil_surf_var(variables, 
                        start_end_dates= [[19610101,19801231], [19810101,20001231],  [20010101,20240320]],
                        lat_lim=None,lon_lim=None,
                        n_samples = None):
    """
    Extracts and concatenates climate data for specified variables and date ranges.

    Parameters:
    - variables (str or list of str): Variable(s) to extract (e.g., 'Tmin', 'Tmax', 'pr').
    - start_end_dates (list of lists): List of [start_date, end_date] pairs in YYYYMMDD format.
    - n_samples (int): Number of time steps to load for each variable.

    Returns:
    - torch.Tensor: Concatenated data for the variable if a single variable is provided.

    Raises:
    - FileNotFoundError: If a required file is not found in the specified path.
    """
    if not isinstance(variables, (str, list)):
        raise ValueError("`variables` must be a string or a list of strings.")
    
    if isinstance(variables, str):
        variables = [variables]

     # Number of time steps to load for each variable

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
            
            if lat_lim is not None and lon_lim is not None:
                cur_df = cur_df.sel(latitude=slice(lat_lim[0], lat_lim[1]), longitude=slice(lon_lim[0], lon_lim[1]))

            if n_samples is not None:
                total_samples = cur_df[next(iter(cur_df.data_vars))].shape[0]
                samples = np.linspace(0, total_samples - 1, n_samples, dtype=int)
                cur_var = cur_df[var_name][samples].values
            else:
                cur_var = cur_df[var_name].values
            data.append(cur_var)
            
            # if not len(metadata):
            #     metadata["lat"] = torch.from_numpy(cur_df['latitude'].values)
            #     metadata["lon"] = torch.from_numpy(cur_df['longitude'].values)
            
            if append_time:
                if n_samples is not None:
                    cur_time = cur_df['time'][samples].values
                else:
                    cur_time = cur_df['time'].values
                time_list.append(cur_time)

        append_time = False
        metadata["time"] = tuple(np.concatenate(time_list, axis=0))
        
        data = np.concatenate(data, axis=0)
        
        # data = data[None] # Insert a batch dimension.
       
        data =  torch.from_numpy(data)
        data = torch.flip(data,[1]) # Flip the vertical axis.

        vars[var_name] = data

        
    mask = vars[variables[0]][0]
    mask = torch.where(~torch.isnan(mask), 1,0)
            
    return vars, metadata["time"], mask
            

def load_era5_static_variables(variables, 
                               area=[5.3, -73.9, -33.9, -34.9],
                               mask=None,
                               lat_lim=None, lon_lim=None):
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
    metadata = {}
    
    for var in variables:
        
        file_path = f"data/raw/{var}.area-subset.{area[0]}.{area[3]}.{area[2]}.{area[1]}.nc"
        cur_var = xr.open_dataset(file_path, engine="netcdf4")
        
        if not len(metadata): 
            metadata['lon'] = torch.from_numpy(cur_var['longitude'].values)
            metadata['lat'] = torch.from_numpy(cur_var['latitude'].values)
            
        var_name = next(iter(cur_var.variables)) #get the variable name
        
        if lat_lim is not None and lon_lim is not None:
            cur_var = cur_var.sel(latitude=slice(lat_lim[0], lat_lim[1]), longitude=slice(lon_lim[0], lon_lim[1]))

        cur_var =torch.from_numpy(cur_var[var_name][0].values)
            
        if mask is not None:
            cur_var = torch.where(mask == 1, cur_var, torch.nan)
           
        vars[str(var_name)] =  cur_var
        

    return vars, metadata['lat'],metadata['lon']


# %%
