
from PIL.Image import Image
from typing import Dict, List, Sequence, Tuple, Union

import torch
from torch import nn, Tensor

import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

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
  