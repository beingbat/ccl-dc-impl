"""
The backbone for each model is assumed to be ResNet18 in CCL-DC, 
Differences can occur in the network heads and how it trained.
So basically adding seperate train loops here for each method
"""

import torch
from torch import nn, optim, cat, FloatTensor, LongTensor
from utils.transforms import (
    TRAIN_TRANSFORMS,
    AUG_TRANSFORMS1,
    AUG_TRANSFORMS2,
    AUG_TRANSFORMS3,
    TEST_TRANSFORMS,
    DEVICE,
)
import time
import random
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


class Trainer:

    model1: nn.Module
    model2: nn.Module
    optim1: optim.Optimizer
    optim2: optim.Optimizer
    criterion: nn.Module
    device = DEVICE
    results_ens = []
    results_0 = []
    results_1 = []

    def __init__(self, model1, model2, optim1, optim2, criterion):
        self.model1 = model1
        self.model2 = model2
        self.optim1 = optim1
        self.optim2 = optim2
        self.criterion = criterion
        self.data_counter = 0
        self.iter = 0

    def train():
        raise NotImplementedError

    def combine_batch(self, batch1, batch2):
        x1, y1 = batch1
        x2, y2 = batch2
        x = cat([x1.to(self.device), x2.to(self.device)])
        y = cat([y1.to(self.device), y2.to(self.device)])
        return (x, y)

    # def backward_transfer(self):
    #     n_tasks = len(self.results)
    #     bt = 0
    #     for i in range(1, n_tasks):
    #         for j in range(i):
    #             bt += self.results[i][j] - self.results[j][j]
    #     return bt / (n_tasks * (n_tasks - 1) / 2)

    # def learning_accuracy(self):
    #     n_tasks = len(self.results)
    #     la = 0
    #     for i in range(n_tasks):
    #         la += self.results[i][i]
    #     return la / n_tasks

    # def relative_forgetting(self):
    #     n_tasks = len(self.results)
    #     rf = 0
    #     max = np.nanmax(np.array(self.results), axis=0)
    #     for i in range(n_tasks - 1):
    #         if max[i] != 0:
    #             rf += self.results_forgetting[-1][i] / max[i]
    #         else:
    #             rf += 1
    #     return rf / n_tasks

    def evaluate(self, dataloaders, task_id):
        with torch.no_grad():
            self.model1.eval()
            self.model2.eval()

            accs_ens = []
            accs_0 = []
            accs_1 = []

            pred_cls_ens = []
            pred_cls_0 = []
            pred_cls_1 = []
            pred_y = []

            for j in range(task_id + 1):
                task_y = None
                task_pred_ens = None
                task_pred_0 = None
                task_pred_1 = None

                for b_id, batch in enumerate(dataloaders[f"{j}"]["test"]):
                    X, y = batch
                    y = y.cpu().numpy()
                    X = TEST_TRANSFORMS(X.to(self.device))

                    x0 = self.model1.logits(X)
                    x1 = self.model2.logits(X)
                    x_ens = (x0 + x1) / 2.0

                    _b_pred_cls_ens = x_ens.argmax(dim=1).cpu().numpy()
                    _b_pred_cls_0 = x0.argmax(dim=1).cpu().numpy()
                    _b_pred_cls_1 = x1.argmax(dim=1).cpu().numpy()

                    if b_id == 0:
                        task_y = y
                        task_pred_0 = _b_pred_cls_0
                        task_pred_1 = _b_pred_cls_1
                        task_pred_ens = _b_pred_cls_ens
                    else:
                        task_y = np.hstack([task_y, y])
                        task_pred_ens = np.hstack([task_pred_ens, _b_pred_cls_ens])
                        task_pred_0 = np.hstack([task_pred_0, _b_pred_cls_0])
                        task_pred_1 = np.hstack([task_pred_1, _b_pred_cls_1])

                task_acc_ens = accuracy_score(task_y, task_pred_ens)
                task_acc_0 = accuracy_score(task_y, task_pred_0)
                task_acc_1 = accuracy_score(task_y, task_pred_1)

                accs_ens.append(task_acc_ens)
                accs_0.append(task_acc_0)
                accs_1.append(task_acc_1)

                pred_cls_ens = np.concatenate([pred_cls_ens, task_pred_ens])
                pred_cls_0 = np.concatenate([pred_cls_0, task_pred_0])
                pred_cls_1 = np.concatenate([pred_cls_1, task_pred_1])
                pred_y = np.concatenate([pred_y, task_y])

            for task_id in range(task_id + 1):
                print(
                    f"Task {task_id} Accuracy: {accs_ens[task_id]} "
                    f"| {accs_0[task_id]} | {accs_1[task_id]}"
                )
            self.results_ens.append(accs_ens)
            self.results_0.append(accs_0)
            self.results_1.append(accs_1)


class ERTrainer(Trainer):
    def __init__(
        self,
        model1,
        model2,
        optim1,
        optim2,
        criterion,
        mem_iter=1,
        mem_bs=64,
        mem_size=500,
    ):
        super().__init__(model1, model2, optim1, optim2, criterion)
        self.mem_iter = mem_iter
        self.mem_bs = mem_bs
        self.mem_size = mem_size
        self.mem_buffer = MemoryBuffer()

    def train(self, dataloader):
        # task_name = dataloader["task_name"]
        # classes_name = dataloader["classes_name"]
        # loader = dataloader["loader"]
        loader = dataloader
        self.model1.train()
        self.model2.train()
        start_time = time.time()
        for b_id, batch in enumerate(loader):
            # Stream data
            X, y = batch
            self.data_counter += len(X)

            for __ in range(self.mem_iter):
                mbatch = self.mem_buffer.get(size=self.mem_bs)

                # Combined batch
                X, y = self.combine_batch(batch, mbatch)
                y = y.long()

                # Default Loss Aug
                X_aug = TRAIN_TRANSFORMS(X)
                # For CCL
                X_aug1 = AUG_TRANSFORMS1(X)
                X_aug2 = AUG_TRANSFORMS2(X_aug1)
                X_aug3 = AUG_TRANSFORMS3(X_aug2)

                # Baseline infer
                x0 = self.model1.logits(X_aug)
                x1 = self.model2.logits(X_aug)

                # Multi-view infer
                x00 = self.model1.logits(X)
                x10 = self.model2.logits(X)

                x01 = self.model1.logits(X_aug1)
                x11 = self.model2.logits(X_aug1)

                x02 = self.model1.logits(X_aug2)
                x12 = self.model2.logits(X_aug2)

                x03 = self.model1.logits(X_aug3)
                x13 = self.model2.logits(X_aug3)

                l0, l1 = self.criterion(
                    (x0, x00, x01, x02, x03, x1, x10, x11, x12, x13, y)
                )

                self.optim1.zero_grad()
                l0.backward()
                self.optim1.step()

                self.optim2.zero_grad()
                l1.backward()
                self.optim2.step()

                self.iter += 1

            self.mem_buffer.add(batch)

        print(
            f"Batch {b_id}/{len(dataloader)} "
            f"Loss (Model 0) : {l0.item():.4f} "
            f"Loss (Model 1) : {l1.item():.4f} "
            f"time : {time.time() - start_time:.4f}s"
        )


class MemoryBuffer:
    processed_count = 0

    def __init__(self, max_size=200, shape=(3, 32, 32), n_classes=10):
        self.n_classes = n_classes
        self.max_size = max_size
        self.shape = shape
        self.buffer_imgs = FloatTensor(
            self.max_size, self.shape[0], self.shape[1], self.shape[2]
        )
        self.buffer_labels = LongTensor(self.max_size)

    def get(self, size):
        if self.processed_count < size:
            return (
                self.buffer_imgs[: self.processed_count],
                self.buffer_labels[: self.processed_count],
            )

        selected_ids = random.sample(
            np.arange(min(self.processed_count, self.max_size)).tolist(), size
        )
        imgs = self.buffer_imgs[selected_ids]
        lbls = self.buffer_labels[selected_ids]

        return imgs, lbls

    def add(self, batch):
        for img, lbl in batch:
            # in first case if buffer has space then just add data
            # in second case, as total data viewed becomes larger the
            # chance of replacing become smaller as the idx has to lie under
            # max size compared to total probability of processed / seen data
            # which just means that given that we see same amount of data for each task,
            # we will have same amount of data for each task in buffer
            if self.processed_count < self.max_size:
                data_idx = self.processed_count
            else:
                data_idx = int(random.random() * (self.processed_count + 1))
            if data_idx < self.max_size:
                self.buffer_imgs[data_idx] = img
                self.buffer_labels[data_idx] = lbl
            self.processed_count += 1
