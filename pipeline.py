from fusion.execute import Executioner, Train, Test, FineTune
from config import Exec, ConfigSeq
from typing import List
from fusion.utils.data import TrainLoader, ValLoader, TestLoader, DataLoader
from abc import ABC, abstractmethod
from fusion.arch.model_builder import Command
from fusion.learning.criteria import Criterion
from fusion.learning.optimizer import Optim
from fusion.execute import Executioner

cfg = ConfigSeq()

class PipeLine:
    def __init__(self, typ: str):
        self.object = {}
        self.type = typ
    def add(self, objec: Executioner):
        self.object.update({objec.name:objec})
    def set_on_start(self):
        self.object[self.type].run(cfg.epoch)
        if self.type == "fine-tune":
            self.object[self.type].add_in_pipe(self.object['test']).run(cfg.epoch)

class Pipeline:
    def __init__(self, exc: Exec):
        self.exec = exc
        self.entities = {}
        self.strategies = {}
        
    def addLoaders(self, *loaders: DataLoader):
        for loader in loaders: self.entities[loader.name] = loader
    
    def addModels(self, *models: Command):
        for model in models: self.entities[model.name] = model.model
    
    def addCriterion(self, criterion: Criterion):
        self.entities['criterion'] = criterion
    
    def addOptimizer(self, optimizer: Optim):
        self.entities['optimizer'] = optimizer
    
    def addExecutions(self, *executions: Executioner):
        for execution in executions: self.entities[execution.name] = execution
    
    def register(self, strategy_name: str, strategy: Strategy):
        self.strategies[strategy_name] = strategy
    
    def execute(self, strategy_name: str):
        self.strategies[strategy_name]

## TODO
### build strategies
class Strategy(ABC):
    @astractmethod
    def execute(self):
        pass

class Training(Strategy):
    def execute(self):
        pass
            

            