import torch
import numpy as np

from torchvision.datasets.cifar import CIFAR10
from torch.utils.data import DataLoader

from utils.transforms import LOADER_TRANSFORMS
import logging


class CIFAR10Partial(CIFAR10):
    def __init__(self, root, train, transform, download=False, selected_labels=[0]):
        super().__init__(root=root, train=train, download=download, transform=transform)
        # convert to tensor for filtering
        self.selected_labels = torch.Tensor(selected_labels)
        self.targets = torch.Tensor(self.targets)
        # filtering indices and only keeping ones matching the selected classes
        self.indexes = torch.nonzero(
            torch.isin(self.targets, self.selected_labels), as_tuple=True
        )[0]
        # filtering data
        self.data = self.data[self.indexes]
        self.targets = self.targets[self.indexes]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.indexes)


def get_dataloaders(
    logger: logging,
    dataset="cifar10",
    total_classes=10,
    classes_per_task=2,
):
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
    dataset = "cifar10"
    data_path = "./data"
    batch_size = 10
    num_workers = 0
    logger.info(f"DATASET: {dataset}")

    # shuffle class order in dataloader
    clss = np.arange(total_classes)
    np.random.shuffle(clss)
    clss_order = clss.tolist()

    logger.info("Creating Data with incremental tasks with classes")
    for task_id, i in enumerate(range(0, total_classes, classes_per_task)):
        selected_classes = clss_order[i : i + classes_per_task]
        logger.info(f"Task ID: {task_id} | Classes: {selected_classes}")
        if dataset == "cifar10":
            dataset_train = CIFAR10Partial(
                data_path,
                train=True,
                transform=LOADER_TRANSFORMS,
                download=True,
                selected_labels=selected_classes,
            )
            dataset_test = CIFAR10Partial(
                data_path,
                train=False,
                transform=LOADER_TRANSFORMS,
                download=True,
                selected_labels=selected_classes,
            )
        else:
            raise NotImplementedError

        dataloaders[f"{task_id}"] = {}
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

    return dataloaders, np.array(clss_order).reshape(-1, classes_per_task)
