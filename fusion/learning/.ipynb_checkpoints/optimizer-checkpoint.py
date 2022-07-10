import torch.nn as nn 
import torch.optim as optim
# !pip install adabelief-pytorch==0.2.0
from adabelief_pytorch import AdaBelief as Adb
from shared.config import Optimizer
from abc import ABC, abstractmethod
from shared.log import setter

logger = setter(__name__)

class Command(ABC):
    @abstractmethod
    def execute(self):
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}"

class Optim:
    def __init__(self):
        self.commands = {}
    
    def register(self, command_name: str, command: Command):
        logger.info("Inside Optim register method")
        logger.debug(f"Registering {command_name}: {command()}")
        self.commands[command_name] = command()
    
    def execute(self, model: nn.Module, config: Optimizer):
        logger.info("Inside Optim execute method")
        if not issubclass(type(config), Optimizer): logger.critical(f"config should be an instance of {Optimizer} got {type(config)}")
        logger.debug(f"Executing {config.name} with {model.__class__.__name__}")
        return self.commands[config.name].execute(model, config)

class Adam(Command):
    def execute(self, model: nn.Module, config: Optimizer):
        op = optim.Adam(model.parameters(), 
                          lr = config.lr, 
                          betas=(config.beta_1, config.beta_2), 
                          weight_decay=config.weight_decay
                        )
        logger.debug(f"Returning {op}: {type(op)}")
        return op

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

