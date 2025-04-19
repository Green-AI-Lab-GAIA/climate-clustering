
#%%
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np 

from load_variables import load_brasil_surf_var, load_era5_static_variables

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
    
if __name__ == "__main__":
        
    surf_vars, metadata, mask = load_brasil_surf_var(['Tmin','Tmax','pr'])
    static_vars = load_era5_static_variables(['slt','geo'],mask=mask)

    plot_random_samples( surf_vars, metadata)
    plot_static(static_vars, metadata['lat'], metadata['lon'])

