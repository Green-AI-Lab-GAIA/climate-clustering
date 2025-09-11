# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#%%

from logging import getLogger

import torch
import torchvision.transforms as transforms
import torchvision

from src.data.load_variables import load_brasil_surf_var, load_era5_static_variables

import numpy as np
import pandas as pd

_GLOBAL_SEED = 0
logger = getLogger()

def init_data(
    transform,
    batch_size,
    surf_vars=['Tmax', 'Tmin', 'pr'],
    static_vars=['slt', 'geo'],
    lat_lim=None, lon_lim=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    drop_last=True,
    dataset_samples=None,
    adj_prep_balance=True,
    split_val=False
):
    
    dataset = BrazilWeatherDataset( transform=transform,
                                    surf_vars=surf_vars,
                                    static_vars=static_vars,
                                    lat_lim=lat_lim, lon_lim=lon_lim,
                                    n_samples=dataset_samples,
                                    adj_prep_balance=adj_prep_balance,
                                    split_val=split_val)
    
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers)
    
    logger.info('data loader created')

    return (data_loader, dist_sampler)


def make_transforms(
    rand_size=224,
    focal_size=96,
    rand_crop_scale=(0.5, 1.0),
    focal_crop_scale=(0.3, 0.7),
    color_jitter=1.0,
    rand_views=2,
    focal_views=10,
    norm_means=None,
    norm_stds=None
):

    rand_transform = transforms.Compose([
        transforms.RandomResizedCrop(rand_size, scale=rand_crop_scale),
        transforms.Normalize(
            norm_means,
            norm_stds)
    ])

    focal_transform = transforms.Compose([
        transforms.RandomResizedCrop(focal_size, scale=focal_crop_scale),
        transforms.Normalize(
            norm_means, 
            norm_stds)
    ])


    transform = MultiViewTransform(
        rand_transform=rand_transform,
        focal_transform=focal_transform,
        rand_views=rand_views,
        focal_views=focal_views
    )
    
    return transform


class MultiViewTransform(object):

    def __init__(
        self,
        rand_transform=None,
        focal_transform=None,
        rand_views=1,
        focal_views=1,
    ):
        self.rand_views = rand_views
        self.focal_views = focal_views
        self.rand_transform = rand_transform
        self.focal_transform = focal_transform

    def __call__(self, img):
        img_views = []

        # -- generate random views
        if self.rand_views > 0:
            img_views += [self.rand_transform(img) for i in range(self.rand_views)]

        # -- generate focal views
        if self.focal_views > 0:
            img_views += [self.focal_transform(img) for i in range(self.focal_views)]

        return img_views



class BrazilWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, transform, surf_vars, static_vars=None, return_patches=False,
                 patch_size=40, patch_stride=20, lat_lim=None, lon_lim=None,n_samples=None,adj_prep_balance=True,split_val=False):
        
        self.transform = transform

        self.imgs, self.validation_imgs = self.load_images(surf_vars, static_vars, return_patches=return_patches,
                                     patch_size=patch_size, patch_stride=patch_stride,
                                     lat_lim=lat_lim, lon_lim=lon_lim,n_samples=n_samples,adj_prep_balance=adj_prep_balance,split_val=split_val)


    def load_images(self, surf_vars, static_vars, 
                    return_patches,patch_size, patch_stride,
                    lat_lim, lon_lim,n_samples,
                    adj_prep_balance,split_val):

        surf_vars_values, time, mask = load_brasil_surf_var(surf_vars,lat_lim=lat_lim,lon_lim=lon_lim,n_samples=n_samples) # ['Tmax','Tmin','pr']
    
        x_surf = torch.stack(tuple(surf_vars_values.values()), dim=1)
        
        if static_vars is not None:
            
            static_vars_values, lat, lon  = load_era5_static_variables(static_vars,mask=mask,lat_lim=lat_lim,lon_lim= lon_lim) #['slt','geo']

            x_static = torch.stack(tuple(static_vars_values.values()), dim=0)

            B, _, H, W = x_surf.shape

            x_static = x_static.expand((B, -1, -1, -1))
            x_surf = torch.cat((x_surf, x_static), dim=1)
        
        if return_patches: 
            _, V, _, _ = x_surf.shape

            patches = x_surf.unfold(2, patch_size,patch_stride).unfold(3, patch_size, patch_stride)
            patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, V, patch_size, patch_size)
            patches = patches[~torch.isnan(patches).any(dim=(1, 2, 3))] #exclude nans (ocean)

            return patches.float()
        
        else:
            
            if split_val:
                
                times_index = pd.to_datetime(time)
                val_indices = np.where(times_index.year.isin([1980, 2000]))[0]
                train_indices = np.where(~times_index.year.isin([1980, 2000]))[0]

                validation = x_surf[val_indices].float()
                x_surf = x_surf[train_indices]

                self.time = times_index[train_indices]
                self.val_time = times_index[val_indices]
                
            else:
                validation = None
                self.time = pd.to_datetime(time)
                
                
            if (surf_vars[0] == 'pr') and adj_prep_balance:
                print("Adjusting precipitation balance...")
                x_surf, time = self.adjust_prep_balance(x_surf, time, total_percnt=0.05, val_crop=5)
                self.time = time
                
            return x_surf.float(), validation


    def adjust_prep_balance(self, data, time, total_percnt = 0.05,val_crop = 5):
        
        low_prep_samples = np.where(data.mean(dim=(1,2,3)) < val_crop)[0]
        other_samples = np.where(data.mean(dim=(1,2,3)) >= val_crop)[0]

        n_samples = int((total_percnt*data.shape[0]) // 1)
        samples = np.linspace(0, low_prep_samples.shape[0] - 1, n_samples, dtype=int)

        new_dataset = torch.cat((data[other_samples],data[low_prep_samples[samples]]))
        time_others, time_lowprep = np.array(time)[other_samples] , np.array(time)[low_prep_samples[samples]]
        new_time = np.concatenate((time_others, time_lowprep))

        sorted_time = pd.Series(new_time).sort_values()
        new_dataset = new_dataset[sorted_time.index]
        new_time = pd.to_datetime(sorted_time.values)
        
        return new_dataset, new_time

    def __getitem__(self, index):

        cimg = self.imgs[index]
        timg = self.transform(cimg)

        return timg, torch.tensor(-1)

    def __len__(self):
        return self.imgs.shape[0]
    
