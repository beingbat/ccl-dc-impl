import os
import logging
from collections import defaultdict

import torch
import numpy as np
from PIL import Image

from torchvision.datasets.cifar import CIFAR10, CIFAR100
from torchvision.datasets.utils import download_and_extract_archive
from torch.utils.data import DataLoader, Dataset
from utils.common import NUM_WORKERS
from utils.transforms import LOADER_TRANSFORMS



class CIFAR100Partial(CIFAR100):
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


class TinyImageNet(Dataset):
    def __init__(self, root: str, train: bool, transform=None, download: bool = True):
        root_dir = os.path.join(root, "tiny-imagenet-200")
        self.train = train
        self.transform = transform
        self.img_shape = (64, 64, 3)

        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        download_and_extract_archive(
            url,
            root,
            filename="tiny-imagenet-200.zip",
            remove_finished=False,
            md5="90528d7ca1a48142e341f4ef8d21d0de",
        )
        self._make_paths(root_dir)

        if self.train:
            self.samples = list(self.paths["train"])
        else:
            self.samples = list(self.paths["val"])

    def _make_paths(self, root_dir):
        train_path = os.path.join(root_dir, "train")
        val_path = os.path.join(root_dir, "val")
        wnids_path = os.path.join(root_dir, "wnids.txt")
        words_path = os.path.join(root_dir, "words.txt")

        self.ids = []
        with open(wnids_path, "r") as idf:
            for nid in idf:
                nid = nid.strip()
                self.ids.append(nid)
        self.nid_to_words = defaultdict(list)
        with open(words_path, "r") as wf:
            for line in wf:
                nid, labels = line.split("\t")
                labels = list(map(lambda x: x.strip(), labels.split(",")))
                self.nid_to_words[nid].extend(labels)

        self.paths = {
            "train": [],  # [img_path, id, nid, box]
            "val": [],  # [img_path, id, nid, box]
        }

        # print(self.paths['test'])
        # Get the validation paths and labels
        with open(os.path.join(val_path, "val_annotations.txt")) as valf:
            for line in valf:
                fname, nid, x0, y0, x1, y1 = line.split()
                fname = os.path.join(val_path, "images", fname)
                bbox = int(x0), int(y0), int(x1), int(y1)
                label_id = self.ids.index(nid)
                self.paths["val"].append((fname, label_id, nid, bbox))

        # Get the training paths
        train_nids = os.listdir(train_path)
        for nid in train_nids:
            anno_path = os.path.join(train_path, nid, nid + "_boxes.txt")
            imgs_path = os.path.join(train_path, nid, "images")
            label_id = self.ids.index(nid)
            with open(anno_path, "r") as annof:
                for line in annof:
                    fname, x0, y0, x1, y1 = line.split()
                    fname = os.path.join(imgs_path, fname)
                    bbox = int(x0), int(y0), int(x1), int(y1)
                    self.paths["train"].append((fname, label_id, nid, bbox))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, target = Image.open(self.samples[idx][0]), self.samples[idx][1]

        if self.transform:
            img = self.transform(img)
        if img.size(0) == 1:
            img = torch.cat([img, img, img], dim=0)
        return img, target


class TinyImageNetPartial(TinyImageNet):
    def __init__(self, root, train, transform, download=False, selected_labels=[0]):
        super().__init__(root=root, train=train, download=download, transform=transform)
        # convert to tensor for filtering
        self.selected_labels = torch.Tensor(selected_labels)
        self.targets = [smpl[1] for smpl in self.samples]
        self.f_paths = [smpl[0] for smpl in self.samples]
        self.targets = torch.Tensor(self.targets)
        # filtering indices and only keeping ones matching the selected classes
        self.indexes = torch.nonzero(
            torch.isin(self.targets, self.selected_labels), as_tuple=True
        )[0]
        # filtering data
        self.f_paths = np.array(self.f_paths)[self.indexes]
        self.targets = self.targets[self.indexes]
        del self.samples

    def __getitem__(self, index):
        img, target = Image.open(self.f_paths[index]), self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if img.size(0) == 1:
            img = torch.cat([img, img, img], dim=0)
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
    data_path = "./data"
    batch_size = 10
    num_workers = NUM_WORKERS
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
        elif dataset == "cifar100":
            dataset_train = CIFAR100Partial(
                data_path,
                train=True,
                transform=LOADER_TRANSFORMS,
                download=True,
                selected_labels=selected_classes,
            )
            dataset_test = CIFAR100Partial(
                data_path,
                train=False,
                transform=LOADER_TRANSFORMS,
                download=True,
                selected_labels=selected_classes,
            )
        elif dataset == "tiny":
            dataset_train = TinyImageNetPartial(
                data_path,
                train=True,
                transform=LOADER_TRANSFORMS,
                download=True,
                selected_labels=selected_classes,
            )
            dataset_test = TinyImageNetPartial(
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
