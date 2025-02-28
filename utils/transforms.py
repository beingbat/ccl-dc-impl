import torch
from torch import nn

from torch import nn
from torchvision.transforms import RandAugment, Compose, ToTensor, ConvertImageDtype
from kornia.augmentation import (
    RandomResizedCrop,
    RandomHorizontalFlip,
    ColorJitter,
    RandomGrayscale,
)
from utils.common import *

LOADER_TRANSFORMS = Compose([ToTensor()])

TRAIN_TRANSFORMS = nn.Sequential(
    RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(MIN_CROP, 1.0)),
    RandomHorizontalFlip(),
    ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
    RandomGrayscale(p=0.2),
)

AUG_TRANSFORMS1 = nn.Sequential(
    RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.6, 1.0)),
    RandomHorizontalFlip(),
)

AUG_TRANSFORMS2 = nn.Sequential(
    ConvertImageDtype(torch.uint8),
    RandAugment(RANDAUG_N, RANDAUG_M),
    ConvertImageDtype(torch.float32),
)

AUG_TRANSFORMS3 = nn.Sequential(
    ConvertImageDtype(torch.uint8),
    RandAugment(RANDAUG_N, RANDAUG_M),
    ConvertImageDtype(torch.float32),
)

TEST_TRANSFORMS = nn.Sequential(nn.Identity())
