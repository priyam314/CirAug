from torchvision import datasets, transforms
import torch
from config import Batch
from abc import ABC, abstractmethod

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
    def __init__(self, name: str = "stl10"):
        self.name = name
        self.params = {
            'root': '../data',
            'download': True,
            'transform': transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406),
                                                      (0.229, 0.224, 0.225))
                             ])
            }

    def dataset_SVHN(self, split: str = 'extra'):
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
        return datasets.SVHN(split = split, **self.params)

    def dataset_STL10(self, split: str = 'unlabeled'):
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
        return datasets.STL10(split = split, **self.params)

    def dataset_FOOD101(self, split: str = 'train'):
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
        return datasets.Food101(split = split, **self.params)

    def __call__(self, split: str = "train"):
        if self.name == "stl10": return self.dataset_STL10(split=split)
        elif self.name == "svhn": return self.dataset_SVHN(split=split)
        elif self.name == "food101": return self.dataset_FOOD101(split=split)

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
        
    Methods:
    -------
    execute: torch.utils.data.DataLoader
        returns the dataloader
    """
    def __init__(self):
        self.dataset = 0
        self.batch_size = 0
        self.nw = 0

    def execute(self):
        print(f"{self.__class__.__name__}: setting up training set\n")
        return torch.utils.data.DataLoader(dataset=self.dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=self.nw,
                                           pin_memory=True)


class TrainLoader(DataLoader):

    def __init__(self,
                 dataset: Dataset,
                 batch: Batch,
                 num_workers: int = 8) -> None:
        self.dataset: Dataset = dataset
        self.batch_size: Batch = batch.train_size
        self.nw: int = num_workers


class ValLoader(DataLoader):

    def __init__(self,
                 dataset: Dataset,
                 batch: Batch,
                 num_workers: int = 8) -> None:
        self.dataset: Dataset = dataset
        self.batch_size: Batch = batch.val_size
        self.nw: int = num_workers


class TestLoader(DataLoader):

    def __init__(self,
                 dataset: Dataset,
                 batch: Batch,
                 num_workers: int = 8) -> None:
        self.dataset: Dataset = dataset
        self.batch_size: Batch = batch.test_size
        self.nw: int = num_workers

def data_summary(*loaders: torch.utils.data.DataLoader):
    for loader in loaders:
        print()
        print(loader.dataset)
        print()
    return