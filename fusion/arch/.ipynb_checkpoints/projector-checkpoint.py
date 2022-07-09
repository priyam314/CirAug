from typing import List
import torch.nn as nn
from config import ConfigSeq, Hlayer
from abc import ABC, abstractmethod

cfg = ConfigSeq()

def get_projector(hlayerList, dropoutList, projector, i_f)->nn.Sequential:
    hlayerList = [i_f] + hlayerList
    if len(hlayerList) == 2:
        projector.add_module("Linear_0", nn.Linear(hlayerList[0], hlayerList[1]))
    elif len(hlayerList) > 2:
        i = 0
        for i in range(0, len(hlayerList)-2):
            projector.add_module("Linear_{}".format(i), nn.Linear(hlayerList[i], hlayerList[i+1]))
            projector.add_module("Dropout_{}".format(i), nn.Dropout(dropoutList[i]))
            projector.add_module("Relu_{}".format(i), nn.ReLU())
        projector.add_module("Linear_L", nn.Linear(hlayerList[-2], hlayerList[-1]))
    elif len(hlayerList) == 1:
        projector.add_module("Linear_0", nn.Linear(hlayerList[0], hlayerList[0]))
    return projector

class AbstractCommand(nn.Module):
    pass

class Command(AbstractCommand):
    @abstractmethod
    def execute(self):
        pass

class Projector:
    def __init__(self):
        self.commands = {}
    
    def register(self, command_name: str, command: Command):
        self.commands[command_name] = command(cfg.hlayer)
    
    def execute(self, command_name: str):
        return self.commands[command_name].execute()
    
class TrainProjector(Command):
    def __init__(self, layerList: Hlayer):
        super(TrainProjector, self).__init__()
        self.layerlist = layerList
        self._projector = 0
    
    def execute(self, i_f = 12):
        self._projector = get_projector(self.layerlist.train, self.layerlist.dropout, nn.Sequential(), i_f)
        return self
    
    @property
    def projector(self): return self._projector
    
    @property
    def name(self): return "trainProjector"
        
class TestProjector(nn.Module):
    
    def __init__(self, layerList: Hlayer):
        super(TestProjector, self).__init__()
        self.layerlist = layerList
    
    def execute(self, i_f = 12):
        self._projector = get_projector(self.layerlist.test, self.layerlist.dropout, nn.Sequential(), i_f)
        return self

    @property
    def projector(self): return self._projector
    
    @property
    def name(self): return "testProjector"
        
    