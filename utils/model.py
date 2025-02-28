from torch import nn, optim
from models.resnet import ResNet
from models.loss import Loss, LossDerpp
from models.train import ERTrainer, DerppTrainer


def get_model(name="er", lr=0.0005, weight_decay=1e-4):
    if name == "er" or name == "derpp":
        resnet18 = ResNet()
        adamw_optim = optim.AdamW(
            resnet18.parameters(), lr=lr, weight_decay=weight_decay
        )
        return resnet18, adamw_optim
    raise NotImplementedError


def get_criterion(name="er", only_base=False, classes_per_task=2):
    criteria = nn.CrossEntropyLoss()
    if name == "er":
        return Loss(criteria, criteria, only_base=only_base)
    elif name == "derpp":
        return LossDerpp(
            criteria, criteria, only_base=only_base, class_count=classes_per_task
        )
    raise NotImplementedError


def get_trainer(name="er"):
    if name == "er":
        return ERTrainer
    if name == "derpp":
        return DerppTrainer
    raise NotImplementedError
