from utils.model import get_model, get_criterion, get_trainer
from utils.data import get_dataloaders
from models.train import Trainer
import logging


def setup_logging():
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    file_handler = logging.FileHandler("main.log", mode="a")
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def main():
    # Experience-Replay for Continual Learning:
    # https://papers.nips.cc/paper_files/paper/2019/file/
    # fa7cdfad1a5aaf8370ebeda47a1ff1c3-Paper.pdf

    # name = "er"
    name = "derpp"
    dataset = "cifar10"
    classes_per_task = 2
    total_classes = 10

    logger = setup_logging()
    logger.info(f"Total Classes {total_classes} | Classes Per Task: {classes_per_task}")

    for c_mem_size in [500, 1000]:
        for iteration_id in range(10):
            if iteration_id < 5:
                base = True
            else:
                base = False

            logger.info(f"Iteration no. {iteration_id}")
            logger.info(
                f"{name.upper()}{' + CCL-DC' if not base else ''} | "
                f"Dataset: {dataset.upper()} | Mem Size: {c_mem_size}"
            )
            dataloaders, classes_order = get_dataloaders(
                logger=logger,
                dataset=dataset,
                total_classes=total_classes,
                classes_per_task=classes_per_task,
            )
            model1, optim1 = get_model(name)
            model2, optim2 = get_model(name)
            criterion = get_criterion(name, base, total_classes)
            trainer_cls = get_trainer(name)
            trainer: Trainer = trainer_cls(
                model1, model2, optim1, optim2, criterion, mem_size=c_mem_size
            )

            for task_idx, task_name in enumerate(dataloaders):
                task_classes = classes_order[task_idx]
                train_dataloader = dataloaders[task_name]["train"]
                logger.info(f"Task: {task_name}; Classes: {task_classes}")
                trainer.train(train_dataloader, task_name)
                trainer.evaluate(dataloaders, task_idx, logger=logger)


if __name__ == "__main__":
    main()
