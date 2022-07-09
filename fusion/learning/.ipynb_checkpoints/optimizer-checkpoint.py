import torch.nn as nn 
import torch.optim as optim
# !pip install adabelief-pytorch==0.2.0
from adabelief_pytorch import AdaBelief as Adb
from config import Optimizer
from abc import ABC, abstractmethod

class Command(ABC):
    @abstractmethod
    def execute(self):
        pass

class Optim:
    def __init__(self):
        self.commands = {}
    
    def register(self, command_name: str, command: Command):
        self.commands[command_name] = command()
    
    def execute(self, command_name: str, model: nn.Module, config: Optimizer):
        return self.commands[command_name].execute(model, config)

class Adam(Command):
    def execute(self, model: nn.Module, config: Optimizer):
        return optim.Adam(model.parameters(), 
                          lr = config.lr, 
                          betas=(config.beta_1, config.beta_2), 
                          weight_decay=config.weight_decay
                        )

class SGD(Command):
    def execute(self, model: nn.Module, config: Optimizer):
        return optim.SGD(model.parameters(), 
                         lr = config.lr, 
                         momentum=config.momentum, 
                         weight_decay=config.weight_decay
                        )

class AdaBelief(Command):
    def execute(self, model: nn.Module, config: Optimizer):
        return Adb(model.parameters(), 
                         lr = config.lr, 
                         betas=(config.beta_1, config.beta_2), 
                         weight_decouple=config.weight_decouple, 
                         eps=config.eps*config.eps, 
                         rectify=config.rectify, 
                         weight_decay=config.weight_decay
                        )

class AdamW(Command):
    def execute(self, model: nn.Module, config: Optimizer):
        return optim.AdamW(model.parameters(),
                          lr = config.lr,
                          betas = (config.beta_1, config.beta_2),
                          eps = config.eps,
                          weight_decay = config.weight_decay,
                          amsgrad = True
                        )

