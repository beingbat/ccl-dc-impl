from torch import nn, optim

from models.resnet import ResNet, BasicBlock
from models.loss import Loss
from models.train import ERTrainer


def get_model(name="er", no_cls=10, lr=0.0005, weight_decay=1e-4):
    if name == "er":
        resnet18 = ResNet(num_classes=no_cls)
        adamw_optim = optim.AdamW(
            resnet18.parameters(), lr=lr, weight_decay=weight_decay
        )
        return resnet18, adamw_optim
    raise NotImplementedError


def get_criterion(name="er", only_base=False):
    if name == "er":
        criteria = nn.CrossEntropyLoss()
        return Loss(criteria, criteria, only_base=only_base)
    raise NotImplementedError


def get_trainer(name="er"):
    if name == "er":
        return ERTrainer
    raise NotImplementedError
