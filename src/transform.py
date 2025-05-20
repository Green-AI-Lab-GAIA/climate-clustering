
from PIL.Image import Image
from typing import  List, Sequence, Tuple, Union

from torch import Tensor

import torchvision.transforms as T


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
                        ]
                    )
                ]
                * crop_counts[i]
            )
        super().__init__(crop_transforms)
