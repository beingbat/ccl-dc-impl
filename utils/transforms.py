import torch
from torch import nn

from torchvision.transforms import functional as F
from torch import nn
import torchvision
from torchvision.transforms import RandAugment
from kornia.augmentation import (
    RandomResizedCrop,
    RandomHorizontalFlip,
    ColorJitter,
    RandomGrayscale,
)

IMG_SIZE = 32
MIN_CROP = 0.2
RANDAUG_N = 3
RANDAUG_M = 15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    torchvision.transforms.ConvertImageDtype(torch.uint8),
    RandAugment(RANDAUG_N, RANDAUG_M),
    torchvision.transforms.ConvertImageDtype(torch.float32),
)

AUG_TRANSFORMS3 = nn.Sequential(
    torchvision.transforms.ConvertImageDtype(torch.uint8),
    RandAugment(RANDAUG_N, RANDAUG_M),
    torchvision.transforms.ConvertImageDtype(torch.float32),
)

TEST_TRANSFORMS = nn.Sequential(nn.Identity())


class ToPilImage(nn.Module):
    def __init__(self, mode=None):
        super().__init__()
        self.mode = mode  # Optional: specify conversion mode (e.g., "RGB", "L")

    def forward(self, img):
        return F.to_pil_image(img, mode=self.mode)
