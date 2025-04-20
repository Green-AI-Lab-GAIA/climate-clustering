"""
Download data from ERA5 reanalysis dataset

API guide: https://cds-beta.climate.copernicus.eu/how-to-api

Notes:
- You can use a website like http://bboxfinder.com to get the coordinates of the desired area,
    but you need to pay attention to the ERA 5 API correct order: [North, West, South, East ]

- You'll need to install:
pip install cdsapi --quiet
pip install cfgrib --quiet
pip install eccodes --quiet
sudo apt-get install libeccodes0  --quiet
pip install netCDF4 scipy
pip install h5netcdf

"""
#%%
import os
import cdsapi
import cfgrib
import xarray as xr
import numpy as np
from time import sleep
import re

def download_era5_single_levels(area_download,years_to_download,
                                times_to_get,variables,dataset_tag='',
                                data_format=  'grib', folder='data/raw'):

    cds_client = cdsapi.Client()

    #download blocks
    block_size = 6
    block_start, block_end = 0, block_size

    datasets = [] #store datasets downloaded 
    if len(dataset_tag): dataset_tag =  "_"+dataset_tag 

    while block_start < len(years_to_download):

        cur_years = years_to_download[block_start:block_end]
        start_y, end_y = cur_years[0],cur_years[-1]

        print('Downloading',cur_years)
        
        ext = data_format if data_format == 'grib' else 'nc'
        dataset_name =f'{folder}/era{dataset_tag}_{start_y}_{end_y}.{ext}'
        datasets.append(dataset_name)

        era_dataset = "reanalysis-era5-single-levels"

        request = {

            'product_type': ['reanalysis'],
            'variable':variables,
            'year': cur_years,

            'month':[ '01', '02', '03','04', '05', '06',
                    '07', '08', '09','10', '11', '12'],

            'day': ['01', '02', '03','04', '05', '06',
                    '07', '08', '09','10', '11', '12',
                    '13', '14', '15','16', '17', '18',
                    '19', '20', '21','22', '23', '24',
                    '25', '26', '27','28', '29', '30','31'],

            'time': times_to_get,
            'area': area_download,
            'data_format': data_format,
            'download_format': 'unarchived',

        }

        print('Request:',request)

        data = cds_client.retrieve(era_dataset, request, dataset_name)
        
        block_start = block_end
        block_end += block_size

        print('-- finished --')

        sleep(10)




def concat_files(folder='data/raw',file_name='era',dataset_tag='',data_format=  'grib'):
    
    #tp timesteps are diff than t2m and msl
    datasets = [i for i in os.listdir(folder) if i[-len(data_format):] == data_format]
    datasets = [i for i in datasets if dataset_tag in i]

    pattern = r'(\d{4})_(\d{4})'

    start_year = min([
        match.group(1)
        for filename in datasets
        if (match := re.search(pattern, filename))  # Walrus operator for assignment and condition
    ])

    end_year = max([
        match.group(2)
        for filename in datasets
        if (match := re.search(pattern, filename))  
    ])

    ds_list = []

    for file in sorted(datasets):

        if file:
    
            if data_format == 'grib':
                ds = cfgrib.open_dataset( os.path.join(folder,file) ) 
            
            elif data_format == 'nc':
                ds = xr.open_dataset(os.path.join(folder,file))

            ds_list.append(ds)
            ds.close()  

    ds = xr.concat(ds_list, dim='time')
    output_file = f'data/{file_name}_{dataset_tag}_{min(start_year)}_{max(end_year)}.nc'
    ds.to_netcdf(output_file)


def open_nc(file_path):

    try:
        ds = xr.open_dataset(file_path)

    except Exception as e:
        print(f"Error opening file: {e}")
        return 

    else:
        return ds


def get_normalized_values(variable, file= 'data/msl_t2m_1983_2023.nc',
               wind = 4, quantile=False,save_file=False):

    ds = open_nc(file)
    df = ds[variable].values

    if quantile:
        max_o = np.quantile(df,0.99)
        min_o = np.quantile(df,0.1)

        df = np.clip(df, min_o, max_o)

    else:
        min_o = np.min(df)
        max_o = np.max(df)

    norm_imgs = (df - min_o)/(max_o -min_o)

    time, lat, long = norm_imgs.shape

    norm_imgs = norm_imgs.reshape(int(time/wind), wind, lat, long)
    norm_imgs = np.transpose(norm_imgs, (0, 2, 3, 1))

    if save_file:
      np.save(f'data/{variable}',norm_imgs,allow_pickle=False)

    return norm_imgs

###################

if __name__ == "__main__":
        
    area_download = [2.46, -71.97, -33.26,-36.64] #brazil area [North, West, South, East ]

    years_to_download = [str(i) for i in range(1983,2024)]

    times_available = [ '00:00', '01:00', '02:00',
                        '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00',
                        '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00',
                        '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00',
                        '21:00', '22:00', '23:00']

    times_to_get = times_available[::3]
     

    # download_era5_single_levels(area_download,years_to_download,times_to_get,
    #                             variables=['2m_temperature', 'mean_sea_level_pressure'],dataset_tag='t2m_msl')

    # concat_files(dataset_tag='t2m_msl')

    #prep data in grib format dont have the same shape // has a lot of nan values
    # download_era5_single_levels(area_download,years_to_download,times_to_get,['total_precipitation'],
    #                             dataset_tag='tp',data_format='netcdf')

    # concat_files(dataset_tag='tp',data_format='nc')

