from torchvision.transforms import functional as FT
import albumentations as A  # our data augmentation library
from albumentations.pytorch import ToTensorV2

def get_transforms(train=False):
    if train:
        transform = A.Compose([
            # A.Resize(600, 600), # our input size can be 600px
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.1),
            A.ColorJitter(p=0.1),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    else:
        transform = A.Compose([
            # A.Resize(600, 600), # our input size can be 600px
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    return transform

from collections import defaultdict
descriptions = defaultdict(str)