import contextlib

import torch
import torch.nn as nn
from models.cc_resnet_old import resnet34
import torchvision.models as models


class BaseModel(nn.Module):
    def __init__(self, conditions, pretrained=True, only_last_layer=False):

        super().__init__()

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.conditions = conditions
        # self.model = resnet34(num_classes=len(conditions), pretrained=False)

        self.model = models.resnet34(pretrained=pretrained)
        if only_last_layer:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, len(conditions))

        self.to(self.device)

    def forward(self, x):
        x = self.model(x.repeat(1, 3, 1, 1).to(self.device))
        return x

