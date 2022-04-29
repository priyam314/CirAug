import torch.nn as nn 
from torchvision import models

class Encoder:
    def __init__(self, name: str = "efficientnet-b0"):
        self.encoder = name
        self.model: nn.Module
    def __call__(self, projector: nn.Sequential):
        if self.encoder == "mobilenet-v2":
            self.model = models.mobilenet_v2()
            self.model.classifier = projector(i_f = self.model.classifier[1].in_features)
        elif self.encoder == "efficientnet-b0":
            self.model = models.efficientnet_b0()
            self.model.classifier = projector(i_f = self.model.classifier[1].in_features)
        return self.model
    def skeleton(self):
        return self.model
    def show(self):
        return self.model.features