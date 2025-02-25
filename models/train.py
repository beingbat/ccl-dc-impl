"""
The backbone for each model is assumed to be ResNet18 in CCL-DC, 
Differences can occur in the network heads and how it trained.
So basically adding seperate train loops here for each method
"""

import logging
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
import random
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm


class Trainer:

    model1: nn.Module
    model2: nn.Module
    optim1: optim.Optimizer
    optim2: optim.Optimizer
    criterion: nn.Module
    device = DEVICE
    accuracies_ensemble = []
    accuracies_0 = []
    accuracies_1 = []

    def __init__(self, model1: nn.Module, model2: nn.Module, optim1, optim2, criterion):
        self.model1 = model1.to(self.device)
        self.model2 = model2.to(self.device)
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
        return (x, y.long())

    def evaluate(self, dataloaders, task_id, logger: logging):
        logger.info("#####################################")
        with torch.no_grad():
            self.model1.eval()
            self.model2.eval()

            accuracy_ensemble = []
            accuracy_0 = []
            accuracy_1 = []

            for j in range(task_id + 1):
                task_y = []
                task_pred_ens = []
                task_pred_0 = []
                task_pred_1 = []
                logger.info(f"Task: {j}")
                with tqdm(dataloaders[f"{j}"]["test"], unit=" batch") as tloader:
                    for __, batch in enumerate(tloader):
                        X, y = batch
                        y = y.cpu().numpy()
                        X = TEST_TRANSFORMS(X.to(self.device))

                        x0 = self.model1(X)
                        x1 = self.model2(X)
                        x_ens = (x0 + x1) / 2.0

                        _b_pred_cls_ens = x_ens.argmax(dim=1).cpu().numpy()
                        _b_pred_cls_0 = x0.argmax(dim=1).cpu().numpy()
                        _b_pred_cls_1 = x1.argmax(dim=1).cpu().numpy()

                        task_y = np.concatenate([task_y, y])
                        task_pred_ens = np.concatenate([task_pred_ens, _b_pred_cls_ens])
                        task_pred_0 = np.concatenate([task_pred_0, _b_pred_cls_0])
                        task_pred_1 = np.concatenate([task_pred_1, _b_pred_cls_1])

                    task_acc_ens = accuracy_score(task_y, task_pred_ens)
                    task_acc_0 = accuracy_score(task_y, task_pred_0)
                    task_acc_1 = accuracy_score(task_y, task_pred_1)

                    accuracy_ensemble.append(task_acc_ens)
                    accuracy_0.append(task_acc_0)
                    accuracy_1.append(task_acc_1)

            self.accuracies_ensemble.append(accuracy_ensemble)
            self.accuracies_0.append(accuracy_0)
            self.accuracies_1.append(accuracy_1)

            for task_id in range(task_id + 1):
                logger.info(
                    f"Task {task_id} Accuracy: {accuracy_ensemble[task_id]} "
                    f"| {accuracy_0[task_id]} | {accuracy_1[task_id]}"
                )

            aa_ensemble = np.mean(accuracy_ensemble)
            aa_0 = np.mean(accuracy_0)
            aa_1 = np.mean(accuracy_1)

            avg_la_ensemble = np.mean(
                [task_accuracies[-1] for task_accuracies in self.accuracies_ensemble]
            )
            avg_la_0 = np.mean(
                [task_accuracies[-1] for task_accuracies in self.accuracies_0]
            )
            avg_la_1 = np.mean(
                [task_accuracies[-1] for task_accuracies in self.accuracies_1]
            )

            task_view_ensemble = []
            task_view_0 = []
            task_view_1 = []
            for i in range(len(self.accuracies_ensemble)):
                task_view_ensemble.append([])
                task_view_0.append([])
                task_view_1.append([])
                for j in range(len(self.accuracies_ensemble[i])):
                    task_view_ensemble[j].append(self.accuracies_ensemble[i][j])
                    task_view_0[j].append(self.accuracies_0[i][j])
                    task_view_1[j].append(self.accuracies_1[i][j])

            fm_ensemble = []
            fm_0 = []
            fm_1 = []

            rf_ensemble = []
            rf_0 = []
            rf_1 = []
            logger.info("Forgetting Measure")
            for tsk_id in range(len(task_view_ensemble)):
                if len(task_view_ensemble[tsk_id]) > 1:
                    fm_ensemble.append(
                        np.max(task_view_ensemble[tsk_id])
                        - task_view_ensemble[tsk_id][-1]
                    )
                    fm_0.append(np.max(task_view_0[tsk_id]) - task_view_0[tsk_id][-1])
                    fm_1.append(np.max(task_view_1[tsk_id]) - task_view_1[tsk_id][-1])

                    rf_ensemble.append(
                        1
                        - task_view_ensemble[tsk_id][-1]
                        / np.max(task_view_ensemble[tsk_id])
                    )
                    rf_0.append(
                        1 - task_view_0[tsk_id][-1] / np.max(task_view_0[tsk_id])
                    )
                    rf_1.append(
                        1 - task_view_1[tsk_id][-1] / np.max(task_view_1[tsk_id])
                    )

                    logger.info(
                        f"FM Task {tsk_id}: {format(fm_ensemble[-1], '.4f')} "
                        f"| {format(fm_0[-1], '.4f')} | {format(fm_1[-1], '.4f')}"
                    )

                    logger.info(
                        f"RF Task {tsk_id}: {format(rf_ensemble[-1], '.4f')} "
                        f"| {format(rf_0[-1], '.4f')} | {format(rf_1[-1], '.4f')}"
                    )
            if len(task_view_ensemble) > 1:
                logger.info(
                    f"FM Mean: {format(np.mean(fm_ensemble), '.4f')} | {format(np.mean(fm_0), '.4f')} | {format(np.mean(fm_1), '.4f')}"
                )
                logger.info(
                    f"RF Mean: {format(np.mean(rf_ensemble), '.4f')} | {np.mean(rf_0)} | {np.mean(rf_1)}"
                )

            logger.info(
                f"Averaging Accuracy: {format(aa_ensemble, '.4f')} "
                f"| {format(aa_0, '.4f')} | {format(aa_1, '.4f')}"
            )

            logger.info(
                f"Averaged Learning Accuracy: {format(avg_la_ensemble, '.4f')} "
                f"| {format(avg_la_0, '.4f')} | {format(avg_la_1, '.4f')}"
            )

            return aa_ensemble, aa_0, aa_1


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

    def train(self, dataloader, task_name="na"):
        self.model1.train()
        self.model2.train()

        train_loss_model0_avg = 0
        train_loss_model1_avg = 0

        # Only single viewing in online continual learning i.e. single epoch
        with tqdm(dataloader, unit=" batch") as tloader:
            for b_id, batch in enumerate(tloader):
                tloader.set_description(f"[train] Task: {task_name}")

                # if b_id > 1:
                #     break

                # Stream data
                X, y = batch
                self.data_counter += len(X)

                for __ in range(self.mem_iter):
                    mbatch = self.mem_buffer.get(size=self.mem_bs)

                    # Combined batch
                    X, y = self.combine_batch(batch, mbatch)

                    # Default Loss Aug
                    X_aug = TRAIN_TRANSFORMS(X)
                    # For CCL
                    X_aug1 = AUG_TRANSFORMS1(X)
                    X_aug2 = AUG_TRANSFORMS2(X_aug1)
                    X_aug3 = AUG_TRANSFORMS3(X_aug2)

                    # Baseline infer
                    x0 = self.model1(X_aug)
                    x1 = self.model2(X_aug)

                    # Multi-view infer
                    x00 = self.model1(X)
                    x10 = self.model2(X)

                    x01 = self.model1(X_aug1)
                    x11 = self.model2(X_aug1)

                    x02 = self.model1(X_aug2)
                    x12 = self.model2(X_aug2)

                    x03 = self.model1(X_aug3)
                    x13 = self.model2(X_aug3)

                    l0, l1 = self.criterion(
                        (x0, x00, x01, x02, x03, x1, x10, x11, x12, x13, y)
                    )
                    train_loss_model0_avg = (
                        (train_loss_model0_avg * self.iter) + l0.item()
                    ) / (self.iter + 1)

                    train_loss_model1_avg = (
                        (train_loss_model1_avg * self.iter) + l1.item()
                    ) / (self.iter + 1)

                    self.optim1.zero_grad()
                    l0.backward()
                    self.optim1.step()

                    self.optim2.zero_grad()
                    l1.backward()
                    self.optim2.step()

                    self.iter += 1

                self.mem_buffer.add(batch)

                tloader.set_postfix(
                    loss_model0=train_loss_model0_avg, loss_model1=train_loss_model1_avg
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
        (imgs, lbls) = batch
        for idx, img in enumerate(imgs):
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
                self.buffer_labels[data_idx] = lbls[idx]
            self.processed_count += 1
