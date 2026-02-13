import torch
import torch.nn as nn
import torch.nn.functional as F



class MMSegCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-1, loss_weight=1.0, avg_non_ignore=True):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self.avg_non_ignore = avg_non_ignore

    def forward(self, logits, target):
        if not self.avg_non_ignore:
            return self.loss_weight * F.cross_entropy(
                logits, target, ignore_index=self.ignore_index, reduction='mean'
            )

        loss = F.cross_entropy(
            logits, target, ignore_index=self.ignore_index, reduction='none'
        )
        valid = (target != self.ignore_index).to(loss.dtype)
        denom = valid.sum().clamp_min(1)
        return self.loss_weight * ((loss * valid).sum() / denom)
      