
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

