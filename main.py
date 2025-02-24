from utils.data import get_dataloaders
from utils.model import get_model, get_criterion, get_trainer
from models.train import Trainer
import numpy as np


def main():
    # Experience-Replay for Continual Learning:
    # https://papers.nips.cc/paper_files/paper/2019/file/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Paper.pdf
    name = "er"
    dataset = "cifar10"
    classes_per_task = 2
    dataloaders, classes_order = get_dataloaders(
        dataset, total_classes=10, classes_per_task=classes_per_task
    )
    classes_order = np.array(classes_order).reshape(-1, 2)
    model1, optim1 = get_model(name)
    model2, optim2 = get_model(name)
    criterion = get_criterion(name)
    trainer_cls = get_trainer(name)
    trainer: Trainer = trainer_cls(model1, model2, optim1, optim2, criterion)

    for task_idx, task_name in enumerate(dataloaders):
        task_classes = classes_order[task_idx]
        train_dataloader = dataloaders[task_name]["train"]
        print(f"Task: {task_name}; Classes: {task_classes}")
        trainer.train(train_dataloader)
        trainer.evaluate(dataloaders, task_name)


if __name__ == "__main__":
    main()
