
from config import Exec, ConfigSeq
from typing import List
from fusion.utils.data import TrainLoader, ValLoader, TestLoader, DataLoader
from abc import ABC, abstractmethod
from fusion.arch.model_builder import Command
from fusion.learning.criterion import Criterion
from fusion.learning.optimizer import Optim
from fusion.execute import Executioner
from ProAug.augSeq import AugSeq
from protrack import ProTrack

cfg = ConfigSeq()

# class PipeLine:
#     def __init__(self, typ: str):
#         self.object = {}
#         self.type = typ
#     def add(self, objec: Executioner):
#         self.object.update({objec.name:objec})
#     def set_on_start(self):
#         self.object[self.type].run(cfg.epoch)
#         if self.type == "fine-tune":
#             self.object[self.type].add_in_pipe(self.object['test']).run(cfg.epoch)

class Strategy(ABC):
    @abstractmethod
    def execute(self):
        pass

class Pipeline:
    def __init__(self, exc: Exec):
        self.exec = exc
        self.entities = {}
        self.strategies = {}
        
    def addLoaders(self, *loaders: DataLoader):
        for loader in loaders: self.entities[loader.name] = loader
    
    def addModels(self, *models: Command):
        for model in models: self.entities[model.name] = model.model.cuda() if cfg.utils.cuda else model.model
    
    def addCriterion(self, criterion: Criterion):
        self.entities['criterion'] = criterion
    
    def addOptimizer(self, optimizer: Optim):
        self.entities['optimizer'] = optimizer
    
    def addExecutions(self, *executions: Executioner):
        for execution in executions: self.entities[execution.name()] = execution
    
    def addAugs(self, augs: AugSeq):
        self.entities['augs'] = augs
    
    def addProtrack(self, pt: ProTrack):
        self.entities['protrack'] = pt
    
    def register(self, strategy_name: str, strategy: Strategy):
        self.strategies[strategy_name] = strategy()
    
    def execute(self, strategy_name: Exec):
        self.strategies[strategy_name.typ].execute(self.entities)

class Training(Strategy):
    def execute(self, entities: dict):
        """execution strategy of training
        trainloader->trainmodel->criterion->optimizer->Augs->TrainExecution
        """
        train = entities['trainExecution'](
                            model= entities['trainModel'],
                            dataloader= entities['trainloader'],
                            Augs= entities['augs'],
                            optim= entities['optimizer'],
                            criterion= entities['criterion'],
                            protrack= entities['protrack'],
                            data= cfg.data
        )
        train.run(cfg.epoch)
        
class FineTuning(Strategy):
    def execute(self, entities: dict):
        """execution strategy of fine tuning
        [valLoader, testLoader]->fine-tuneModel->criterion->optimizer->Augs->[FineTuneExecution, TestExecution]
        """
        fine_tune = entities['fine-tuneExecution'](
                            model= self.entitites['fine-tuneModel'],
                            dataloader= self.entities['valloader'],
                            Augs= self.entities['augs'],
                            optim= self.entities['optimizer'],
                            criterion= self.entities['criterion'],
                            protrack= self.entities['protrack'],
                            data= cfg.data
        )
        test = entities['testExecution'](
                            model= self.entitites['fine-tuneModel'],
                            dataloader= self.entities['testloader'],
                            Augs= self.entities['augs'],
                            optim= self.entities['optimizer'],
                            criterion= self.entities['criterion'],
                            protrack= self.entities['protrack'],
                            data= cfg.data
        )
        fine_tune.compositeRun(test, cfg.epoch)

            