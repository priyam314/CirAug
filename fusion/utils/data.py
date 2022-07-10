from torchvision import datasets, transforms
import torch
import shared.config as config
from abc import ABC, abstractmethod
from shared.log import setter

logger = setter(__name__)

class Command(ABC):
    def __init__(self):
        self.params = {
            'root': '../data',
            'download': True,
            'transform': transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406),
                                                      (0.229, 0.224, 0.225))
                             ])
            }
        
    @abstractmethod
    def execute(self):
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}"

class Dataset:
    """Applies transforms to desired dataet and downloads it
    
    .......
    
    Attributes:
    ----------
    name: str
        name of the dataset
    
    params: dict
        essential parameters for datasets
    """
    def __init__(self):
        self.commands = {}
    
    def register(self, command_name: str, command: Command):
        logger.info("Inside Dataset Register Method")
        if not command_name: logger.error(f"command_name got as {command_name}")
        if not issubclass(command, Command): logger.critical(f"command should be an instance of {Command} got {command}")
        logger.debug(f"Registering {command_name}: {command()}")
        self.commands[command_name] = command()
    
    def execute(self, command_name: config.Data, split: str):
        logger.info("Inside Dataset Execute Method")
        if not issubclass(type(command_name), config.Data): logger.critical(f"command_name should be an instance of {config.Data} got {command_name}")
        if not split: logger.warning(f"split argument shall be passed as string got {split}")
        logger.debug(f"Executing {command_name.name} with split: {split}")
        return self.commands[command_name.name].execute(split=split)

class SVHN(Command):
    """transforms and downloads the SVHN dataset
        
        .........
        
        Parameters:
        ----------
        split: str
            split -> 'train','test','extra'
            
        Returns:
        -------
        torchvision.dataset
    """
    
    def execute(self, split: str): 
        logger.info("Inside SVHN class")
        logger.debug(f"Executing SVHN with split: {split}")
        return datasets.SVHN(split = split, **self.params)

class STL10(Command):
    """transforms and downloads the STL10 dataset
        
        .........
        
        Parameters:
        ----------
        split: str
            split -> 'unlabeled', 'train', 'test'
        
        Returns:
        -------
        torchvision.dataset
    """
    def execute(self, split: str):
        logger.info("Inside STL10 class")
        logger.debug(f"Executing STL10 with split: {split}")
        return datasets.STL10(split = split, **self.params)

class FOOD101(Command):
    """transforms and downloads the Food101 dataset
        
        .........
        
        Parameters:
        ----------
        split: str
            split -> 'train', 'test'
        
        Returns:
        -------
        torchvision.dataset
    """
    def execute(self, split: str): 
        logger.info("Inside FOOD101 class")
        logger.debug(f"Executing FOOD101 with split: {split}")
        return datasets.FOOD101(split = split, **self.params)

class DataLoader(torch.utils.data.DataLoader):
    """DataLoader base class
    creates a dataloader around the dataset and divides the data in 
    batches. Also assigns the number of workers.
    
    .........
    
    Attributes:
    ----------
    dataset: Dataset
        assigns the dataset that been chosen
        
    batch_size: int
        takes the value of batch size
    
    nw: int
        assigns the number of workers
    
    _name: str
        name of the dataloader to uniquely define it
        
    Methods:
    -------
    execute: torch.utils.data.DataLoader
        returns the dataloader
        
    Property:
    -------
    name: str
        returns the name of the dataloader
    """
    def __init__(self):
        self.dataset = 0
        self.batch_size = 0
        self.nw = 0
        self._name = "dataloader"
        self._dataset = 0

    def execute(self):
        logger.info(f"Inside Execute method of {self.__class__.__name__}")
        self._dataset = torch.utils.data.DataLoader(dataset=self.dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=self.nw,
                                           pin_memory=True)
        logger.debug(f"Returning {self.name}")
        return self
    
    def __repr__(self):
        return f"{type(self).__name__}"
    
    @property
    def data(self): return self._dataset

    @property
    def name(self): return self._name


class TrainLoader(DataLoader):

    def __init__(self,
                 dataset: Dataset,
                 batch: config.Batch,
                 num_workers: int = 8) -> None:
        self.dataset: Dataset = dataset
        self.batch_size: config.Batch = batch.train_size
        self.nw: int = num_workers
        self._name: str = "trainloader"


class ValLoader(DataLoader):

    def __init__(self,
                 dataset: Dataset,
                 batch: config.Batch,
                 num_workers: int = 8) -> None:
        self.dataset: Dataset = dataset
        self.batch_size: config.Batch = batch.val_size
        self.nw: int = num_workers
        self._name: str = "valloader"

class TestLoader(DataLoader):

    def __init__(self,
                 dataset: Dataset,
                 batch: config.Batch,
                 num_workers: int = 8) -> None:
        self.dataset: Dataset = dataset
        self.batch_size: config.Batch = batch.test_size
        self.nw: int = num_workers
        self._name: str = "testloader"

def data_summary(*loaders: torch.utils.data.DataLoader):
    for loader in loaders:
        print()
        print(loader.dataset)
        print()
    return