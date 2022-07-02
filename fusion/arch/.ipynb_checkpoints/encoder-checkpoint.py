import torch.nn as nn 
from torchvision import models
from config import Model
from abc import ABC, abstractmethod

class Encoder:
    def __init__(self):
        self.commands = {}
    
    def register(self, command_name:str, command: Command):
        self.commands[command_name] = command
    
    def execute(self, command_name: str, projector: nn.Sequential):
        return self.command[command_name].init(self.projector)

class Command(ABC):
    
    def __init__(self):
        self.model = 0
    
    @abstractmethod
    def init(self, projector: nn.Sequential):
        pass
    
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