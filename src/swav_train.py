# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.
#%%

import sys
import os
import numpy as np
import PIL
from PIL.Image import Image
from tqdm import tqdm
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as T

from lightly.loss import SwaVLoss
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes
from lightly.models.modules.memory_bank import MemoryBankModule


class CopyTransform(nn.Module):

    def __init__(self, n_copies=5):
        super().__init__()

        self.n_copies = n_copies
        self.copy = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip()
        ])

    def forward(self, img: Tensor):
        width, height = img.shape
        half_width, half_height = int(width // 2), int(height // 2)

        original_img = img.clone()
        for _ in range(self.n_copies):
            img_copy = self.copy(original_img)
            empty_pixels = torch.all(img == 0, dim=0)
            if not torch.any(empty_pixels):
                return img
            
            # Random generate center
            indexes = torch.where(empty_pixels)
            x_ind = torch.randint(0, indexes[0].shape[0], (1,))
            y_ind = torch.randint(0, indexes[1].shape[0], (1,))
            x, y = indexes[0][x_ind], indexes[1][y_ind]
            
            # Find original image boundaries
            x0, y0 = max(0, x - half_width), max(0, y - half_height)
            x1, y1 = min(width, x + half_width), min(height, y + half_height)
            
            # Find copy boundaries
            dx, dy = x1 - x0, y1 - y0
            img_copy = img_copy[
                :,
                half_width - dx//2:half_width + dx//2 + dx%2,
                half_height - dy//2:half_height + dy//2 + dy%2
            ]

            img[:, x0:x1, y0:y1] = torch.where(
                empty_pixels[x0:x1, y0:y1],
                img_copy,
                img[:, x0:x1, y0:y1]
            )

        return img


class MultiViewTransform:
    def __init__(self, transforms: Sequence[T.Compose]):
        self.transforms = transforms

    def __call__(self, image: Union[Tensor, Image]) -> Union[List[Tensor], List[Image]]:
        return [transform(image) for transform in self.transforms]    

class MultiCropTranform(MultiViewTransform):
    def __init__(
        self,
        crop_sizes: Tuple[int, ...],
        crop_counts: Tuple[int, ...],
        crop_min_scales: Tuple[float, ...],
        crop_max_scales: Tuple[float, ...],
        transforms: T.Compose,
    ):
        crop_transforms = []
        for i in range(len(crop_sizes)):
            random_resized_crop = T.RandomResizedCrop(
                crop_sizes[i], scale=(crop_min_scales[i], crop_max_scales[i])
            )

            crop_transforms.extend(
                [
                    T.Compose(
                        [
                            random_resized_crop,
                            transforms,
                        ]
                    )
                ]
                * crop_counts[i]
            )
        super().__init__(crop_transforms)



class SwaVTransform(MultiCropTranform):
    def __init__(
        self,
        crop_sizes: Tuple[int, int] = (11, 8),
        crop_counts: Tuple[int, int] = (2, 3),
        crop_min_scales: Tuple[float, float] = (0.8, 0.2),
        crop_max_scales: Tuple[float, float] = (1.0, 0.5),

        normalize: Union[None, Dict[str, List[float]]] = None,
    ):

        transforms = T.Compose([
            T.Normalize(mean=normalize['mean'], std=normalize['std'])
            ])

        super().__init__(
            crop_sizes=crop_sizes,
            crop_counts=crop_counts,
            crop_min_scales=crop_min_scales,
            crop_max_scales=crop_max_scales,
            transforms=transforms,
        )
    
    
class SwAVDataset(Dataset):
    def __init__(self,imgs_path,patch_size=11, patch_stride=8,
                        low_info_thresh=None,adjust_scale=False,filter_data=None):
        
        self.data =  np.load(imgs_path)

        self.max_value= np.nanmax(self.data) 
        self.corrected_mean = np.nanmean(self.data) /self.max_value
        self.corrected_std = np.nanstd(self.data) /self.max_value

        self.data = torch.tensor(self.data, dtype=torch.float64)

        if filter_data:
            random_dx= np.random.randint(0, self.data.shape[0],size=filter_data)    
            self.data= self.data[random_dx]

        self.imgs = self._extract_patches(self.data, patch_size, patch_stride)  
        
        if adjust_scale: 
            self.imgs*=1000

        if low_info_thresh is not None:
            print("######### Initial datasetsize:",self.imgs.shape[0],"\n")
            self.imgs = self._remove_low_info_data(self.imgs ,low_info_thresh)
            print("Final datasetsize:",self.imgs.shape[0]," #########\n")

    def __getitem__(self, idx) -> torch.Tensor:
        return torch.Tensor(self.imgs[idx]).to(torch.uint8)

    def __len__(self):
        return len(self.imgs)

    def _extract_patches(self, data, patch_size, patch_stride,exclude_nans=True):
        """
        Extrai patches usando unfold para divisões com ou sem sobreposição.
        """
        b, h, w = data.shape
        patches = data.unfold(1, patch_size, patch_stride).unfold(2, patch_size, patch_stride)
        patches = patches.contiguous().view(-1, patch_size, patch_size)

        if exclude_nans:
            patches = patches[~torch.isnan(patches).any(dim=(1, 2))] 

        return patches
    
    def _remove_low_info_data(self, data, quantile_thresh=0.2):

        patch_variance = data.var(dim=(1, 2))
        variance_threshold = patch_variance.quantile(quantile_thresh)

        low_info = (patch_variance <= variance_threshold) 
        informative_indices = torch.nonzero(~low_info).squeeze()

        return data[informative_indices]

def resnet_backbone(backbone_model,pretrained_path=None):

    if backbone_model == 'resnet18':
        backbone = torchvision.models.resnet18()

    elif backbone_model == 'resnet50':
        backbone = torchvision.models.resnet50()

    #change to 1 channel
    original_conv1 = backbone.conv1

    backbone.conv1 = nn.Conv2d(
        in_channels=1,  # Change input channels to 1
        out_channels=original_conv1.out_channels,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=original_conv1.bias
    )

    # Initialize new weights for the modified conv1
    nn.init.kaiming_normal_(backbone.conv1.weight, mode='fan_out', nonlinearity='relu')

    # Remove the fully connected layer (last layer) of ResNet
    backbone = nn.Sequential(*list(backbone.children())[:-1])

    if pretrained_path:
        state_dict = torch.load(pretrained_path)
        backbone.load_state_dict(state_dict, strict=False) 
        print(f"Loaded pretrained weights from {pretrained_path}")

    return backbone


class SwaV(nn.Module):
    def __init__(self, backbone_model='resnet50',
                input_size=11,
                n_hr_views=2, n_prototypes=60,
                n_features_swav=128, batch_size=64):
        super().__init__()

        backbone = resnet_backbone(backbone_model)

        with torch.no_grad():
            n_features_backbone = backbone(torch.randn(batch_size, 1, input_size, input_size)).shape[1]

        self.backbone = backbone

        self.projection_head = SwaVProjectionHead(
            input_dim=n_features_backbone, 
            hidden_dim=n_features_backbone, # arbitrary hidden dim, setting as the same as backbone
            output_dim=n_features_swav
        )
        self.prototypes = SwaVPrototypes(
            input_dim=n_features_swav, 
            n_prototypes=n_prototypes, 
            n_steps_frozen_prototypes=1
        )

        self.start_queue_at_epoch = 2
        self.queues = nn.ModuleList(
            [MemoryBankModule(size=(3840, n_features_swav)) for _ in range(n_hr_views)]
        )

    def forward(self, high_resolution, low_resolution, epoch):
        self.prototypes.normalize()

        high_resolution_features = [self._subforward(x) for x in high_resolution]
        low_resolution_features = [self._subforward(x) for x in low_resolution]

        high_resolution_prototypes = [
            self.prototypes(x, epoch) for x in high_resolution_features
        ]
        low_resolution_prototypes = [
            self.prototypes(x, epoch) for x in low_resolution_features
        ]
        queue_prototypes = self._get_queue_prototypes(high_resolution_features, epoch)

        return high_resolution_prototypes, low_resolution_prototypes, queue_prototypes

    def _subforward(self, input):
        # with torch.no_grad():
        features = self.backbone(input).flatten(start_dim=1)
        features = self.projection_head(features)
        features = nn.functional.normalize(features, dim=1, p=2)
        return features

    @torch.no_grad()
    def _get_queue_prototypes(self, high_resolution_features, epoch):
        if len(high_resolution_features) != len(self.queues):
            raise ValueError(
                f'The number of queues ({len(self.queues)}) should be equal to the number of high '
                f'resolution inputs ({len(high_resolution_features)}). Set `n_queues` accordingly.'
            )

        # Get the queue features
        queue_features = []
        for i in range(len(self.queues)):
            _, features = self.queues[i](high_resolution_features[i], update=True)
            # Queue features are in (num_ftrs X queue_length) shape, while the high res
            # features are in (batch_size X num_ftrs). Swap the axes for interoperability.
            features = torch.permute(features, (1, 0))
            queue_features.append(features)

        # If loss calculation with queue prototypes starts at a later epoch,
        # just queue the features and return None instead of queue prototypes.
        if self.start_queue_at_epoch > 0 and epoch < self.start_queue_at_epoch:
            return None

        # Assign prototypes
        queue_prototypes = [self.prototypes(x, epoch) for x in queue_features]
        return queue_prototypes

def main(dataset:str):

    ########################### config ###########################

    device = 'cuda'
    backbone_model = 'resnet50'  
    n_features_swav = 128
    batch_size = 512
    n_prototypes = 15
    learning_rate = 0.0001
    sinkhorn_epsilon = 0.03
    max_patience = 5
    min_epochs = 20
    
    patch_size = 11
    patch_stride = 8

    # transformation crops
    n_hr_views = 2
    n_lr_views = 6
    crop_sizes = (11, 8)
    min_scales = (0.8, 0.2)
    max_scales = (1.0, 0.5) 

    ########################### 

    imgs_path=f"/u00/arquivos_compartilhados/green_lab/climate_data/reanalysis-era5-land/{dataset}.npy"
    checkpoints_file_prefix = f'/u00/arquivos_compartilhados/green_lab/era_land_parana_checkpoints/{dataset}_{patch_size}patches_{n_prototypes}proto'
    os.makedirs(checkpoints_file_prefix, exist_ok=True)
    
    log_file = open(checkpoints_file_prefix+"/log.txt", "a") 

    log_file.write(f"""Model config:
    
    backbone_model={backbone_model}, 
    n_features_swav={n_features_swav},
    batch_size ={batch_size},
    n_prototypes={n_prototypes},
    learning_rate={learning_rate},
    sinkhorn_epsilon={sinkhorn_epsilon},
    max_patience ={max_patience}, 
    min_epochs ={min_epochs}, 
    patch_size={patch_size},
    patch_stride={patch_stride},
    n_crops={n_hr_views,n_lr_views},
    crop_sizes={crop_sizes},
    min_scales={min_scales},
    max_scales={max_scales} 

    Start training: \n""")

    ###########################
    
    print('Loading images dataset into RAM')

    if 'precipitation' in checkpoints_file_prefix:
        dataset = SwAVDataset(imgs_path,patch_size=patch_size,patch_stride=patch_stride,
                            low_info_thresh=0.3,adjust_scale=True)
        
    else:
        dataset = SwAVDataset(imgs_path,patch_size=patch_size,patch_stride=patch_stride)

    dataset_size = f'Total images: {len(dataset)}'
    print(dataset_size)
    log_file.write(dataset_size+"\n") 

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    mosaic_normalize = {'mean':[dataset.corrected_mean], 'std' :[dataset.corrected_std]}
    transform = SwaVTransform( crop_sizes=crop_sizes, crop_min_scales = min_scales,
                                crop_max_scales = max_scales, crop_counts=(n_hr_views, n_lr_views),
                                normalize=mosaic_normalize)

    ln_proto = np.round(np.log(n_prototypes),5)
    print("############# ln(proto):",ln_proto,"#############")
    log_file.write("ln(proto): "+str(ln_proto)+"\n") 

    model = SwaV(
        backbone_model,
        patch_size,
        n_hr_views,
        n_prototypes,
        n_features_swav,
        batch_size
    ).to(device)

    # try:
    #     model.load_state_dict(torch.load(f"{checkpoints_file_prefix}/minloss.pth", weights_only=True))  
    # except:
    #     print("New model")

    criterion = SwaVLoss(sinkhorn_epsilon = sinkhorn_epsilon)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    patience = 0
    epoch = 0
    min_loss = np.inf

    print(f'Starting Training')
    
    while True:
        total_loss = 0
        for batch in dataloader:
            batch = torch.concat(
                [
                    batch[i].view(1, 1, patch_size, patch_size)
                    for i in range(batch.shape[0])
                ],
                dim=0
            ).to(device)
            views = transform(batch /  dataset.max_value)
            high_resolution, low_resolution = views[:n_hr_views], views[n_hr_views:]

            high_resolution, low_resolution, queue = model(
                high_resolution, low_resolution, epoch
            )

            loss = criterion(high_resolution, low_resolution, queue)
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()

        avg_loss = total_loss / len(dataloader)
        patience += 1
        epoch += 1
        loss_print = f'epoch: {epoch:>04}, loss: {avg_loss:.5f}'

        if avg_loss < min_loss:
            min_loss = avg_loss
            loss_print += ' (*)'
            patience = 0
            torch.save(model.state_dict(), f'{checkpoints_file_prefix}/minloss.pth')
        
        if ((epoch % 100) == 0):
            torch.save(model.state_dict(), f'{checkpoints_file_prefix}/epoch_{epoch}.pth')
        
        print(loss_print)
        log_file.write(loss_print+"\n") 

        if( patience >= max_patience) and (epoch>= min_epochs):
            torch.save(model.state_dict(), f'{checkpoints_file_prefix}/epoch_{epoch}.pth')
            break

    log_file.close()

from argparse import ArgumentParser

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", default="temperature")

    args = parser.parse_args()

    main(dataset=args.dataset)
