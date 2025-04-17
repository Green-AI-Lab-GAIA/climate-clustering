
#%%
import cdsapi
import os 

from src.sharepoint_connection import SharePointConnection

from pathlib import Path
import os

project_root = Path(__file__).resolve().parent.parent
current_working_dir = Path.cwd().resolve()

if current_working_dir != project_root:
    os.chdir(project_root)


def download_era5_data(area, path="data/raw/"):
    """
    Downloads ERA5 land data for a specified geographical area.

    Parameters:
    area (list): A list specifying the geographical bounding box in the format 
                 [north, west, south, east].
    path (str): The directory path where the downloaded data will be stored.

    Returns:
    None: The function downloads and extracts the data into the specified directory.
    """
    dataset = "reanalysis-era5-land"
    request = {
        "variable": [
            "geopotential",
            "soil_type"
        ],
        "data_format": "netcdf",
        "download_format": "zip",
        "area": area
    }

    # Initialize the CDS API client
    client = cdsapi.Client(
        url="https://cds.climate.copernicus.eu/api",
        key="b0474ce3-c4ed-4747-b666-66dcca14d8ea"
    )

    # Retrieve and download the dataset
    file_name = client.retrieve(dataset, request).download()

    # Unzip the downloaded file and remove the zip file
    os.system(f"unzip {file_name} -d {path}")
    os.system(f"rm {file_name}")


def download_brasil_data_from_onedrive(variables,start_end_dates):

    password = open("secrets/password.txt", "r").read()
    login = open("secrets/login.txt", "r").read()
    
    sp = SharePointConnection(
        login,
        password,
        sharepoint_site="https://gvmail.sharepoint.com/sites/GreenAILab",
        sharepoint_site_name="GreenAILab",
        sharepoint_doc="Green AI Lab Data",
    )
    
    for var in variables:
        
        for start, end in start_end_dates:
            
            file =  f"{var}_{start}_{end}_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc"
    
            sp.download_file(remote_path=f"Clima/Climate Clustering/Dados Entrada/{file}",
                            local_path="data/raw/")
            
    
# # download params 
# start_end = [[19610101,19801231], 
#             [19810101,20001231],  
#             [20010101,20240320]] 

# variables = ["pr","Tmin","Tmax"]

# area = [5.3, -73.9, -33.9, -34.9]


#%%
# import matplotlib.pyplot as plt

# # check if lat/lon are aligned

# test_br = xr.open_dataset("data/raw/Tmin_19810101_20001231_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc", engine="netcdf4")
# test_br = np.flipud(test_br['Tmin'].values[0])

# test_era5 = vars['slt']['slt'].values[0].copy()
# test_era5 = np.where(test_era5!=0,test_era5,np.nan)

# print("Brazil data shape: ",test_br.shape)
# print("ERA 5 data shape: ",test_era5.shape)

# br_mask = np.isnan(test_br).astype(int)
# era_mask = np.isnan(test_era5).astype(int)

# # Criando uma figura com 3 subplots
# fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# # Mostrar A
# im0 = axs[0].imshow(np.nan_to_num(era_mask, nan=-1), cmap='viridis')
# axs[0].set_title("Dado ERA5")

# # Mostrar B
# im1 = axs[1].imshow(np.nan_to_num(br_mask, nan=-1), cmap='viridis')
# axs[1].set_title("Dado Brazil")

# # Diferença (ignorando nans)
# diff = np.where(np.isnan(era_mask) & np.isnan(br_mask), np.nan, era_mask - br_mask)
# im2 = axs[2].imshow(np.nan_to_num(diff, nan=-1), cmap='coolwarm')
# axs[2].set_title("Diferença (ERA - Brazil)")
# plt.colorbar(im2, ax=axs[2])

# plt.tight_layout()
# plt.show()