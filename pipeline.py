from shared.log import setter
from shared.config import Exec, ConfigSeq
from typing import List
from fusion.utils.data import TrainLoader, ValLoader, TestLoader, DataLoader
from abc import ABC, abstractmethod
from fusion.arch.model_builder import Command
from fusion.learning.criterion import Criterion
from fusion.learning.optimizer import Optim
from fusion.execute import Executioner
from ProAug.augSeq import AugSeq
from protrack import ProTrack

logger = setter(__name__)
cfg = ConfigSeq()

class Strategy(ABC):
    @abstractmethod
    def execute(self):
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}"

class Pipeline:
    def __init__(self, exc: Exec):
        self.exec = exc
        self.entities = {}
        self.strategies = {}
        logger.info(f"Inside Pipeline Class, as arg: {type(exc)} class")
        
    def addLoaders(self, *loaders: DataLoader):
        logger.info("Inside addLoaders method")
        for loader in loaders: 
            logger.debug(f"adding {loader.name}: {loader} to entities dict")
            self.entities[loader.name] = loader
        
    def addProjectors(self, projectorObj):
        logger.info("Inside addProjectors method")
        for name, command in projectorObj.commands.items(): 
            logger.debug(f"adding {name}: {command} to entities dict")
            self.entities[name] = command
    
    def addModels(self, *models: Command):
        logger.info("Inside addModels method")
        for model in models: 
            logger.debug(f"adding {model.name}: {model.__class__.__name__} to entities dict")
            self.entities[model.name] = model.model.cuda() if cfg.utils.cuda else model.model
    
    def addCriterion(self, criterion: Criterion):
        logger.info("Inside addCriterion method")
        logger.debug(f"adding object of Criterion to entities dict")
        self.entities['criterion'] = criterion
    
    def addOptimizer(self, optimizer: Optim):
        logger.info("Inside addOptimizer method")
        if not optimizer: logger.critical(f"optimizer is {optimizer}")
        logger.debug(f"adding object of Optim to entities dict")
        self.entities['optimizer'] = optimizer
    
    def addExecutions(self, *executions: Executioner):
        logger.info("Inside addExecutions method")        
        for execution in executions: 
            logger.debug(f"adding {execution.name()}: {execution} to entities dict")
            self.entities[execution.name()] = execution
    
    def addAugs(self, augs: AugSeq):
        logger.info("Inside addAugs method")
        logger.debug(f"adding {augs} to entities dict")
        self.entities['augs'] = augs
    
    def addProtrack(self, pt: ProTrack):
        logger.info("Inside addProtrack method")
        logger.debug(f"adding {pt.trackDict.keys()} to entities dict")        
        self.entities['protrack'] = pt
    
    def register(self, strategy_name: str, strategy: Strategy):
        logger.info("Inside register method")
        logger.debug(f"adding {strategy_name}: {strategy()} to entities dict")
        self.strategies[strategy_name] = strategy()
    
    def execute(self, strategy_name: Exec):
        logger.info("Inside execute method")
        logger.debug(f"Executing {strategy_name.typ}")
        self.strategies[strategy_name.typ].execute(self.entities)

class Training(Strategy):
    def execute(self, entities: dict):
        """execution strategy of training
        trainloader->trainmodel->criterion->optimizer->Augs->TrainExecution
        """
        logger.info("Inside Training execute method")
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
        logger.info("Inside FineTuning execute method")
        fine_tune = entities['fine-tuneExecution'](
                        model= entities['fine-tuneModel'],
                        dataloader= entities['valloader'],
                        projector= entities['trainProjector'],
                        Augs= entities['augs'],
                        optim= entities['optimizer'],
                        criterion= entities['criterion'],
                        protrack= entities['protrack'],
                        data= cfg.data
        )
        test = entities['testExecution'](
                    model= entities['fine-tuneModel'],
                    dataloader= entities['testloader'],
                    Augs= entities['augs'],
                    optim= entities['optimizer'],
                    criterion= entities['criterion'],
                    protrack= entities['protrack'],
                    data= cfg.data
        )
        fine_tune.compositeRun(test, cfg.epoch)

            