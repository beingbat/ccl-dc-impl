from torch import nn
import torch.nn.functional as F

class Loss(nn.Module):
    """
    Calculates the baseline loss, cls loss and collaborative continual learning 
    and distillation chain loss. For both models in peer-learning.
    """
    def __init__(self, base_criterion, cls_criterion, kd_lambda=0.5, baseline_lambda=0.5):
        super().__init__()
        self.base_criterion = base_criterion
        self.cls_criterion = cls_criterion
        self.kdl = kd_lambda
        self.basel = baseline_lambda

    def forward(self, data):
        (x0, x00, x01, x02, x03, 
         x1, x10, x11, x12, x13, 
         y) = data

        # TODO: baseline loss assumed to be class loss; fix this

        # baseline loss
        loss1 = self.base_criterion(x0, y)
        loss2 = self.base_criterion(x1, y)

        # cls loss
        loss1 += (self.cls_criterion(x00, y) + 
                    self.cls_criterion(x01, y) + 
                    self.cls_criterion(x02, y) + 
                    self.cls_criterion(x03, y))

        loss2 += (self.cls_criterion(x10, y) + 
                     self.cls_criterion(x11, y) + 
                     self.cls_criterion(x12, y) + 
                     self.cls_criterion(x13, y))

        # ccl and dc Loss
        loss1_dc = (self.kl_loss(x00, x10.detach()) + # ccl
                    self.kl_loss(x00, x11.detach()) + # dc
                    self.kl_loss(x01, x12.detach()) + 
                    self.kl_loss(x02, x13.detach())) 

        loss2_dc = (self.kl_loss(x10, x00.detach()) + # ccl
                    self.kl_loss(x10, x01.detach()) + # dc
                    self.kl_loss(x11, x02.detach()) + 
                    self.kl_loss(x12, x03.detach()))

        # Total Loss
        l1 = (self.basel * loss1  + 
              self.kdl * loss1_dc)
        l2 = (self.basel * loss2 + 
              self.kdl * loss2_dc)

        self.l1 = l1.item()
        self.l2 = l2.item()

        print(f"Loss (Peer1) : {l1.item():.4f} "
              f"Loss (Peer2) : {l2.item():.4f} ", end="\r")

        return l1, l2

    def kl_loss(self, ls, lt, temperature=4.0):
        """
        Args:
            ls: student logits
            lt: teacher logits
            temperature: temperature
        Returns:
            distillation loss
        """
        pred_t = F.softmax(lt / temperature, dim=1)
        log_pred_s = F.log_softmax(ls / temperature, dim=1)
        lkd = F.kl_div(
            log_pred_s,
            pred_t,
            reduction='none'
        ).sum(1).mean(0) * (temperature ** 2)
        return lkd
