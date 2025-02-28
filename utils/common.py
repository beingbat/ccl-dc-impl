import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXP_REPITIONS = 1
MIN_CROP = 0.2

DATASET = "tiny"
IMG_SIZE = 64
MEM_SIZE = [2000, 5000, 10000]
RANDAUG_N = 1
RANDAUG_M = 11
CLASSES_PER_TASK = 2
TOTAL_CLASSES = 200

# DATASET = "cifar10"
# IMG_SIZE = 32
# MEM_SIZE = [500, 1000]
# MIN_CROP = 0.2
# RANDAUG_N = 3
# RANDAUG_M = 15
# CLASSES_PER_TASK = 2
# TOTAL_CLASSES = 10

NAME = "er"
# NAME = "derpp"
