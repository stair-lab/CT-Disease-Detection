import torch.nn as nn

from models.cc_resnet_old import resnet34
from utils.checkpoints import load_checkpoint

import torch


class BaseModelOld(nn.Module):
    def __init__(self, conditions, checkpoint):

        super().__init__()

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.conditions = conditions
        self.model = resnet34(num_classes=len(conditions), pretrained=False)
        print(conditions)
        load_checkpoint(checkpoint, self.model)

        self.to(self.device)

    def forward(self, x):
        x = self.model(x.repeat(1, 3, 1, 1).to(self.device))
        return x




