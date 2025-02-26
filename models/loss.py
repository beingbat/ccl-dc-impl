from torch import nn
import torch.nn.functional as F


class Loss(nn.Module):
    """
    Calculates the baseline loss, cls loss and collaborative continual learning
    and distillation chain loss. For both models in peer-learning.
    """

    def __init__(
        self,
        base_criterion,
        cls_criterion,
        only_base=False,
        kd_lambda=2,
        baseline_lambda=0.5,
    ):
        super().__init__()
        self.only_base = only_base
        self.base_criterion = base_criterion
        self.cls_criterion = cls_criterion
        self.kdl = kd_lambda
        self.basel = baseline_lambda

    def forward(self, data):
        (x0, x00, x01, x02, x03, x1, x10, x11, x12, x13, y) = data

        # baseline loss
        loss1 = self.base_criterion(x0, y)
        loss2 = self.base_criterion(x1, y)

        if not self.only_base:
            # cls loss
            loss1 += (
                self.cls_criterion(x00, y)
                + self.cls_criterion(x01, y)
                + self.cls_criterion(x02, y)
                + self.cls_criterion(x03, y)
            )

            loss2 += (
                self.cls_criterion(x10, y)
                + self.cls_criterion(x11, y)
                + self.cls_criterion(x12, y)
                + self.cls_criterion(x13, y)
            )

            # ccl and dc Loss
            loss1_dc = (
                self.kl_loss(x03, x13.detach())  # ccl
                + self.kl_loss(x00, x11.detach())  # dc
                + self.kl_loss(x01, x12.detach())
                + self.kl_loss(x02, x13.detach())
            )

            loss2_dc = (
                self.kl_loss(x13, x03.detach())  # ccl
                + self.kl_loss(x10, x01.detach())  # dc
                + self.kl_loss(x11, x02.detach())
                + self.kl_loss(x12, x03.detach())
            )

        l1 = loss1
        l2 = loss2

        if not self.only_base:
            l1 *= self.basel
            l2 *= self.basel
            l1 += self.kdl * loss1_dc
            l2 += self.kdl * loss2_dc

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
        lkd = F.kl_div(log_pred_s, pred_t, reduction="none").sum(1).mean(0) * (
            temperature**2
        )
        return lkd


class LossDerpp(Loss):
    def __init__(
        self,
        base_criterion,
        cls_criterion,
        only_base=False,
        class_count=2,
        kd_lambda=2.0,
        baseline_lambda=0.5,
        alpha=0.2,
        beta=1.0,
    ):
        super().__init__(
            base_criterion,
            cls_criterion,
            only_base=only_base,
            kd_lambda=kd_lambda,
            baseline_lambda=baseline_lambda,
        )
        self.class_count = class_count
        self.alpha = alpha
        self.beta = beta

    def forward(self, data):
        (
            x0,
            m00,
            m01,
            x00,
            x01,
            x02,
            x03,
            x1,
            m10,
            m11,
            x10,
            x11,
            x12,
            x13,
            y_,
            y,
            mem0_y,
            mem1_y,
        ) = data

        # baseline loss
        loss1 = self.base_criterion(x0, y_.long())
        loss2 = self.base_criterion(x1, y_.long())
        if mem0_y is not None:
            loss1 += self.alpha * F.mse_loss(
                m00, F.one_hot(mem0_y, num_classes=self.class_count).float()
            )
            loss2 += self.alpha * F.mse_loss(
                m10, F.one_hot(mem0_y, num_classes=self.class_count).float()
            )

            loss1 += self.beta * self.base_criterion(m01, mem1_y.long())
            loss2 += self.beta * self.base_criterion(m11, mem1_y.long())

        if not self.only_base:
            y = y.long()
            # cls loss
            loss1 += (
                self.cls_criterion(x00, y)
                + self.cls_criterion(x01, y)
                + self.cls_criterion(x02, y)
                + self.cls_criterion(x03, y)
            ) * self.basel

            loss2 += (
                self.cls_criterion(x10, y)
                + self.cls_criterion(x11, y)
                + self.cls_criterion(x12, y)
                + self.cls_criterion(x13, y)
            ) * self.basel

            # ccl and dc Loss
            loss1_dc = (
                self.kl_loss(x03, x13.detach())  # ccl
                + self.kl_loss(x00, x11.detach())  # dc
                + self.kl_loss(x01, x12.detach())
                + self.kl_loss(x02, x13.detach())
            )

            loss2_dc = (
                self.kl_loss(x13, x03.detach())  # ccl
                + self.kl_loss(x10, x01.detach())  # dc
                + self.kl_loss(x11, x02.detach())
                + self.kl_loss(x12, x03.detach())
            )

        l1 = loss1
        l2 = loss2

        if not self.only_base:
            l1 += self.kdl * loss1_dc
            l2 += self.kdl * loss2_dc

        return l1, l2
