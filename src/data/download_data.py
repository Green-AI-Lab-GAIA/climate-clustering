
#%%
import cdsapi
import os 

from sharepoint_connection import SharePointConnection

from pathlib import Path
import os

import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
current_working_dir = Path.cwd().resolve()

if current_working_dir != project_root:
    os.chdir(project_root)


def download_era5_data(dataset_name, area, path="data/raw/era5/",years_to_download=["2023"]):

    # os.mkdir(path, exist_ok=True)
    download_path = Path(path)

    client = cdsapi.Client(
        url="https://cds.climate.copernicus.eu/api",
        key="b0474ce3-c4ed-4747-b666-66dcca14d8ea"
    )
    
    current_dataset = download_path  / f"{dataset_name}-static.nc"
    
    if not (current_dataset).exists():
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "geopotential",
                    "land_sea_mask",
                    "soil_type",
                ],
                "year": "2023",
                "month": "01",
                "day": "01",
                "time": "00:00",
                "format": "netcdf",
                "area": area,
            },
            str(current_dataset),
        )

        print("Static variables downloaded!")


    for year_list in years_to_download:

        start, end = year_list[0], year_list[-1]
        current_dataset = download_path  / f"{dataset_name}-{start}-{end}-surface-level.nc"

        if not (current_dataset).exists():
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": [
                        "2m_temperature",
                        "10m_u_component_of_wind",
                        "10m_v_component_of_wind",
                        "mean_sea_level_pressure",
                    ],
                    "year": year_list,
                    "month": ["01", "02", "03","04", "05", "06",
                                "07", "08", "09","10", "11", "12"],
                    "day":  [ "01", "02", "03","04", "05", "06",
                                "07", "08", "09","10", "11", "12",
                                "13", "14", "15","16", "17", "18",
                                "19", "20", "21","22", "23", "24",
                                "25", "26", "27","28", "29", "30","31"],
                    "time": ["00:00", "06:00", "12:00", "18:00"],
                    "format": "netcdf",
                    "area": area,
                },
                str(current_dataset),
            )

        print("Surface-level variables downloaded for years:", year_list)


        current_dataset = download_path  / f"{dataset_name}-{start}-{end}-atmospheric.nc"

        if not (current_dataset).exists():
            client.retrieve(
                "reanalysis-era5-pressure-levels",
                {
                    "product_type": "reanalysis",
                    "variable": [
                        "temperature",
                        "u_component_of_wind",
                        "v_component_of_wind",
                        "specific_humidity",
                        "geopotential",
                    ],
                    "pressure_level": [
                        "50","100","150","200","250",
                        "300","400","500","600","700",
                        "850","925","1000",
                    ],
                    "year": year_list,
                    "month": ["01", "02", "03","04", "05", "06",
                                "07", "08", "09","10", "11", "12"],
                    "day":  [ "01", "02", "03","04", "05", "06",
                                "07", "08", "09","10", "11", "12",
                                "13", "14", "15","16", "17", "18",
                                "19", "20", "21","22", "23", "24",
                                "25", "26", "27","28", "29", "30","31"],
                    "time": ["00:00", "06:00", "12:00", "18:00"],
                    "format": "netcdf",
                    "area": area,
                },
                str(current_dataset),
            )
                
        print("Atmospheric variables downloaded for years:", year_list)
        print('\n')

def download_era5_static_data(area, path="data/raw/"):
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
            
    
if __name__ == "__main__":
    
#     # Define the area for Brazil
    area = [5.3, -73.9, -33.9, -34.9]  # [north, west, south, east]

    # years_to_download=[str(i) for i in range(1980, 2024)]
    years_to_download = [
                        # ['1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987','1988', '1989'],
                        ['1990','1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998','1999'],
                        ['2000', '2001','2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009',],
                        ['2010','2011', '2012','2013', '2014', '2015', '2016', '2017', '2018', '2019'], 
                        ['2020', '2021', '2022', '2023','2024']]

    download_era5_data(dataset_name="brasil",
                    area=area,
                    path="data/raw/era5/",
                    years_to_download=years_to_download)


#     # Download ERA5 data
#     download_era5_static_data(area)
    
#     # Define the variables and date ranges for Brasil data
#     variables = ["pr", "Tmin", "Tmax"]
#     start_end_dates = [
#         [19610101, 19801231],
#         [19810101, 20001231],
#         [20010101, 20240320]
#     ]
    
#     # Download Brasil data from OneDrive
#     download_brasil_data_from_onedrive(variables, start_end_dates)  


