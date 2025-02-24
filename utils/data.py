from PIL import Image
import torch
import torchvision.datasets as datasets
from torchvision.datasets.cifar import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from utils.transforms import ToPilImage


# class CIFAR10:
#     """CIFAR10 Instance Dataset."""

#     def __init__(self, data_root_dir, train, download, transform):
#         super().__init__(
#             data_root_dir, train=train, download=download, transform=transform
#         )
#         self.targets = torch.Tensor(self.targets)

#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#         Returns:
#             tuple: (image, target) where target is index of the target class.
#         """
#         # if self.train:
#         img, target = self.data[index], self.targets[index]
#         # else:
#         # img, target = self.data[index], self.targets[index]

#         # doing this so that it is consistent with all other datasets
#         # to return a PIL Image
#         img = Image.fromarray(img)

#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return img, target, index


# Image.fromarray


def filter_mask(y, labels):
    """Utility used to create a mask to filter values in a tensor.

    Args:
        y (list, torch.Tensor): tensor where each element is a numeric integer
            representing a label.
        labels (list, torch.Tensor): filter used to generate the mask. For each
            value in ``y`` its mask will be "1" if its value is in ``labels``,
            "0" otherwise".
    Returns:
        mask (torch.ByteTensor): a binary mask, with "1" with the respective value from ``y`` is
        in the ``labels`` filter.
    """
    mask = torch.zeros(y.size(), dtype=torch.uint8)
    for label in labels:
        mask = mask | y.eq(label)
    return mask


class SplitCIFAR10(CIFAR10):
    def __init__(self, root, train, transform, download=False, selected_labels=[0]):
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.selected_labels = selected_labels
        self.targets = torch.Tensor(self.targets)
        self.indexes = torch.nonzero(
            filter_mask(self.targets, self.selected_labels)
        ).flatten()

    def __getitem__(self, index):
        img, target = self.data[self.indexes[index]], self.targets[self.indexes[index]]
        # img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.indexes[index]

    def __len__(self):
        return len(self.indexes)


def get_dataloaders(dataset="cifar10", total_classes=10, classes_per_task=2):
    """
    Creates Task Incremental DataLoaders for given dataset

    Parameters
    =======
     `dataset`: Dataset name to be loaded

     `classes_per_task`: Class count which are contained in a task

    Returns
    =======
     `List[Dict]` : List of dataloaders in form of dict with some meta information
        as well i.e. classes used, task id etc.
    """
    dataloaders = {}

    # shuffle class order in dataloader
    clss = np.arange(total_classes)
    np.random.shuffle(clss)
    clss_order = clss.tolist()

    # Apply normalization and toPil transforms by default when loading dataset
    tf = torch.nn.Sequential(transforms.ToTensor(), ToPilImage())

    # for incremental training dividing data
    dataloaders = get_incremental_dataloaders(
        dataloaders,
        tf,
        total_classes,
        classes_per_task,
        clss_order,
    )

    # dataset_train = CIFAR10("./data/", train=True, download=True, transform=tf)
    # dataset_test = CIFAR10("./data/", train=False, download=True, transform=tf)
    # dataloaders["train"] = DataLoader(
    #     dataset_train,
    #     batch_size=10,
    #     shuffle=True,
    #     num_workers=4,
    # )
    # dataloaders["test"] = DataLoader(
    #     dataset_test,
    #     batch_size=10,
    #     shuffle=False,
    #     num_workers=4,
    # )

    return dataloaders, clss_order


def get_incremental_dataloaders(
    dataloaders, tf, total_classes, classes_per_task, clss_order
):
    step_size = classes_per_task
    dataset = "cifar10"
    data_path = "./data"
    batch_size = 10
    num_workers = 4

    # lg.info("Loading incremental splits with labels :")
    # for i in range(0, args.n_classes, step_size):
    #     lg.info([args.labels_order[j] for j in range(i, i + step_size)])
    # shuffling classes for training
    for i in range(0, total_classes, step_size):
        task_id = int(i / step_size)
        selected_classes = [clss_order[j] for j in range(i, i + step_size)]
        if dataset == "cifar10":
            dataset_train = SplitCIFAR10(
                data_path,
                train=True,
                transform=tf,
                download=True,
                selected_labels=selected_classes,
            )
            dataset_test = SplitCIFAR10(
                data_path,
                train=False,
                transform=tf,
                download=True,
                selected_labels=selected_classes,
            )
        else:
            raise NotImplementedError

        dataloaders[f"{task_id}"]["train"] = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        dataloaders[f"{task_id}"]["test"] = DataLoader(
            dataset_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    return dataloaders
