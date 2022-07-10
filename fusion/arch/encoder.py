import torch.nn as nn 
from torchvision import models
from shared.config import Model
from abc import ABC, abstractmethod
from shared.log import setter

logger = setter(__name__)

class Command(ABC):
    def __init__(self):
        self.model = 0
    
    @abstractmethod
    def execute(self, projector):
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}"
    
class Encoder:
    def __init__(self):
        self.commands = {}
    
    def register(self, command_name:str, command: Command): 
        logger.info("Inside Encoder register method")
        logger.debug(f"Registering {command_name}: {command()}")
        self.commands[command_name] = command()
    
    def execute(self, command_name: Model, projector):
        logger.info("Inside Encoder execute method")
        if not issubclass(type(command_name), Model): logger.critical(f"command_name should be an instance of {Model} got {type(command_name)}")
        logger.debug(f"Executing {command_name.encoder} with {projector.__class__.__name__}")
        return self.commands[command_name.encoder].execute(projector)
    
    def __repr__(self):
        return f"{type(self).__name__}"

class MobileNet_V2(Command):
        
    def execute(self, projector)->nn.Module:
        logger.info(f"Inside {self.__class__.__name__} execute method")
        self.model = models.mobilenet_v2()
        logger.info(f"Fetched the skeleton of {self.__class__.__name__} from torchvision.models")
        self.model.classifier = projector.execute(i_f = self.model.classifier[1].in_features).projector
        logger.info("Assigned the projector to model.classifier")
        return self.model

class EfficientNet_B0(Command):
        
    def execute(self, projector)->nn.Module:
        logger.info(f"Inside {self.__class__.__name__} execute method")
        self.model = models.efficientnet_b0()
        logger.info(f"Fetched the skeleton of {self.__class__.__name__} from torchvision.models")
        self.model.classifier = projector.execute(i_f = self.model.classifier[1].in_features).projector
        logger.info("Assigned the projector to model.classifier")
        return self.model