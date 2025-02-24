from torch import nn, optim

from models.resnet import ResNet, BasicBlock
from models.loss import Loss
from models.train import ERTrainer

def get_model(name="er", no_cls=10, dim_in=512, nf=64, bias=True, lr=0.0005, weight_decay=1e-4):
    if name=="er":
        resnet18 = ResNet(BasicBlock, 
                               [2, 2, 2, 2], 
                               no_cls, nf, bias, 
                               dim_in=dim_in)
        adamw_optim = optim.AdamW(resnet18.parameters(), 
                                lr=lr,
                                weight_decay=weight_decay)
        return resnet18, adamw_optim
    raise NotImplementedError

def get_criterion(name="er"):
    if name=="er":
        criteria = nn.CrossEntropyLoss()
        return Loss(criteria, criteria)
    raise NotImplementedError

def get_trainer(name="er"):
    if name=="er":
        return ERTrainer
    raise NotImplementedError
