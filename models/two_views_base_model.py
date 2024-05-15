import torch
import torch.nn as nn
from models.cc_resnet_old import resnet34, Head
import torchvision.models as models


class TwoViewsBaseModel(nn.Module):
    def __init__(self, conditions, pretrained=True, only_last_layer=False):

        super().__init__()

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.conditions = conditions
        # self.model_front = resnet34(num_classes=len(self.conditions), pretrained=False, add_head=False)
        # self.model_lat = resnet34(num_classes=len(self.conditions), pretrained=False, add_head=False)
        # self.head = Head(in_planes=1024 * 2, mid_planes=512, out_planes=len(self.conditions))

        self.model_front = models.resnet34(pretrained=pretrained)
        if only_last_layer:
            for param in self.model_front.parameters():
                param.requires_grad = False

        self.model_lat = models.resnet34(pretrained=pretrained)
        if only_last_layer:
            for param in self.model_lat.parameters():
                param.requires_grad = False

        self.head = Head(
            in_planes=self.model_front.fc.in_features + self.model_lat.fc.in_features,
            mid_planes=512,
            out_planes=len(self.conditions)
        )
        self.model_front.fc = nn.Identity()
        self.model_lat.fc = nn.Identity()

        self.to(self.device)

    def forward(self, x):
        x_front, x_lat = x
        x_front = self.model_front(x_front.repeat(1, 3, 1, 1).to(self.device))
        x_lat = self.model_lat(x_lat.repeat(1, 3, 1, 1).to(self.device))
        x = self.head(torch.cat([x_front, x_lat], dim=1))
        return x
