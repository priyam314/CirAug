import torch.nn as nn 
from torchvision import models
from config import Model
from abc import ABC, abstractmethod

class Command(ABC):
    def __init__(self):
        self.model = 0
    
    @abstractmethod
    def init(self, projector: nn.Sequential):
        pass
    
class Encoder:
    def __init__(self):
        self.commands = {}
    
    def register(self, command_name:str, command: Command): 
        self.commands[command_name] = command()
    
    def execute(self, command_name: Model, projector: nn.Sequential):
        return self.commands[command_name.encoder].init(projector)

class MobileNet_V2(Command):
        
    def init(self, projector: nn.Sequential)->nn.Module:
        self.model = models.mobilenet_v2()
        self.model.classifier = projector(i_f = self.model.classifier[1].in_features)
        return self.model

class EfficientNet_B0(Command):
        
    def init(self, projector: nn.Sequential)->nn.Module:
        self.model = models.efficientnet_b0()
        self.model.classifier = projector(i_f = self.model.classifier[1].in_features)
        return self.model