# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#%%
import os
import subprocess
import time

from logging import getLogger

from PIL import ImageFilter

import torch
import torchvision.transforms as transforms
import torchvision

from src.data.load_variables import load_brasil_surf_var, load_era5_static_variables

import numpy as np

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
    root_path=None,
    # image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None,
    dataset_samples=None
):

    # dataset = ImageNet(
    #     root=root_path,
    #     image_folder=image_folder,
    #     transform=transform,
    #     train=training,
    #     copy_data=copy_data)
    
    # if subset_file is not None:
    #     dataset = ImageNetSubset(dataset, subset_file)
    
    # logger.info('ImageNet dataset created')
    

    dataset = BrazilWeatherDataset( transform=transform,
                                    surf_vars=surf_vars,
                                    static_vars=static_vars,
                                    lat_lim=lat_lim, lon_lim=lon_lim,
                                    n_samples=dataset_samples)
    
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=dist_sampler,
        # shuffle=True,
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
    # logger.info('making imagenet data transforms')

    # def get_color_distortion(s=1.0):
    #     # s is the strength of color distortion.
    #     color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    #     rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    #     rnd_gray = transforms.RandomGrayscale(p=0.2)
    #     color_distort = transforms.Compose([
    #         rnd_color_jitter,
    #         rnd_gray])
    #     return color_distort

    rand_transform = transforms.Compose([
        transforms.RandomResizedCrop(rand_size, scale=rand_crop_scale),
        # transforms.RandomHorizontalFlip(),
        # get_color_distortion(s=color_jitter),
        # GaussianBlur(p=0.5),
        # transforms.ToTensor(),
        transforms.Normalize(
            norm_means, #'Tmin', 'Tmax',
            norm_stds)
    ])

    focal_transform = transforms.Compose([
        transforms.RandomResizedCrop(focal_size, scale=focal_crop_scale),
        # transforms.RandomHorizontalFlip(),
        # get_color_distortion(s=color_jitter),
        # GaussianBlur(p=0.5),
        # transforms.ToTensor(),
        transforms.Normalize(
            norm_means, #'Tmin', 'Tmax',
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
                 patch_size=40, patch_stride=20, lat_lim=None, lon_lim=None,n_samples=None):
        
        # self.root = root
        self.transform = transform

        self.imgs = self.load_images(surf_vars, static_vars, return_patches=return_patches,
                                     patch_size=patch_size, patch_stride=patch_stride,
                                     lat_lim=lat_lim, lon_lim=lon_lim,n_samples=n_samples)

    def load_images(self, surf_vars, static_vars=None, return_patches=False,
                    patch_size=None, patch_stride=None,
                    lat_lim=None, lon_lim=None,n_samples=None):

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
            self.time = time
            return x_surf.float()

    def __getitem__(self, index):

        cimg = self.imgs[index]
        timg = self.transform(cimg)

        return timg, torch.tensor(-1)

    def __len__(self):
        return self.imgs.shape[0]
    

class ImageNet(torchvision.datasets.ImageFolder):

    def __init__(
        self,
        root,
        image_folder='imagenet_full_size/061417/',
        tar_folder='imagenet_full_size/',
        tar_file='imagenet_full_size-061417.tar',
        transform=None,
        train=True,
        job_id=None,
        local_rank=None,
        copy_data=True
    ):
        """
        ImageNet

        Dataset wrapper (can copy data locally to machine)

        :param root: root network directory for ImageNet data
        :param image_folder: path to images inside root network directory
        :param tar_file: zipped image_folder inside root network directory
        :param train: whether to load train data (or validation)
        :param job_id: scheduler job-id used to create dir on local machine
        :param copy_data: whether to copy data from network file locally
        """

        suffix = 'train/' if train else 'val/'
        data_path = None
        if copy_data:
            logger.info('copying data locally')
            data_path = copy_imgnt_locally(
                root=root,
                suffix=suffix,
                image_folder=image_folder,
                tar_folder=tar_folder,
                tar_file=tar_file,
                job_id=job_id,
                local_rank=local_rank)
        if (not copy_data) or (data_path is None):
            data_path = os.path.join(root, image_folder, suffix)
        logger.info(f'data-path {data_path}')

        super(ImageNet, self).__init__(root=data_path, transform=transform)
        logger.info('Initialized ImageNet')


class ImageNetSubset(object):

    def __init__(self, dataset, subset_file):
        """
        ImageNetSubset

        :param dataset: ImageNet dataset object
        :param subset_file: '.txt' file containing IDs of IN1K images to keep
        """
        self.dataset = dataset
        self.subset_file = subset_file
        self.filter_dataset_(subset_file)

    def filter_dataset_(self, subset_file):
        """ Filter self.dataset to a subset """
        root = self.dataset.root
        class_to_idx = self.dataset.class_to_idx
        # -- update samples to subset of IN1k targets/samples
        new_samples = []
        logger.info(f'Using {subset_file}')
        with open(subset_file, 'r') as rfile:
            for line in rfile:
                class_name = line.split('_')[0]
                target = class_to_idx[class_name]
                img = line.split('\n')[0]
                new_samples.append(
                    (os.path.join(root, class_name, img), target)
                )
        self.samples = new_samples

    @property
    def classes(self):
        return self.dataset.classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.dataset.loader(path)
        if self.dataset.transform is not None:
            img = self.dataset.transform(img)
        if self.dataset.target_transform is not None:
            target = self.dataset.target_transform(target)
        return img, target


class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


def copy_imgnt_locally(
    root,
    suffix,
    image_folder='imagenet_full_size/061417/',
    tar_folder='imagenet_full_size/',
    tar_file='imagenet_full_size-061417.tar',
    job_id=None,
    local_rank=None
):
    if job_id is None:
        try:
            job_id = os.environ['SLURM_JOBID']
        except Exception:
            logger.info('No job-id, will load directly from network file')
            return None

    if local_rank is None:
        try:
            local_rank = int(os.environ['SLURM_LOCALID'])
        except Exception:
            logger.info('No job-id, will load directly from network file')
            return None

    source_file = os.path.join(root, tar_folder, tar_file)
    target = f'/scratch/slurm_tmpdir/{job_id}/'
    target_file = os.path.join(target, tar_file)
    data_path = os.path.join(target, image_folder, suffix)
    logger.info(f'{source_file}\n{target}\n{target_file}\n{data_path}')

    tmp_sgnl_file = os.path.join(target, 'copy_signal.txt')

    if not os.path.exists(data_path):
        if local_rank == 0:
            commands = [
                ['tar', '-xf', source_file, '-C', target]]
            for cmnd in commands:
                start_time = time.time()
                logger.info(f'Executing {cmnd}')
                subprocess.run(cmnd)
                logger.info(f'Cmnd took {(time.time()-start_time)/60.} min.')
            with open(tmp_sgnl_file, '+w') as f:
                print('Done copying locally.', file=f)
        else:
            while not os.path.exists(tmp_sgnl_file):
                time.sleep(60)
                logger.info(f'{local_rank}: Checking {tmp_sgnl_file}')

    return data_path
