from torchvision import datasets, transforms
import torch
from config import Batch
from abc import ABC, abstractmethod

class Dataset:
    
    def __init__(self, which: str = "stl10"):
        self.which = which
        
    def dataset_SVHN(self, split: str = 'extra'):
        """
        >>> split -> 'train','test','extra'
        """
        return datasets.SVHN(root = '../data', split = split, download = True,
                              transform = transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                              ]))
    def dataset_STL10(self, split: str = 'unlabeled'):
        """
        >>> split -> 'unlabeled', 'train', 'test'
        """
        return datasets.STL10(root = '../data', split = split, download = True,
                              transform = transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                              ]))
    def dataset_FOOD101(self, split: str = 'train'):
        """
        >>> split -> 'train', 'test'
        """
        return datasets.Food101(root = '../data', split = split, download = True,
                              transform = transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                              ]))
    
    def __call__(self, split: str = "train"):
        if self.which == "stl10":
            return self.dataset_STL10(split=split)
        elif self.which == "svhn":
            return self.dataset_SVHN(split=split)
        elif self.which == "food101":
            return self.dataset_FOOD101(split=split)
        
def data_summary(*loaders: torch.utils.data.DataLoader):
    for loader in loaders:
        print ()
        print (loader.dataset)
        print ()
    return

class DataLoader(torch.utils.data.DataLoader):
    
    def __init__(self):
        self.dataset = 0
        self.batch_size = 0
        self.nw = 0
    
    
    def execute(self):
        print (f"{self.__class__.__name__}: setting up training set\n")
        return torch.utils.data.DataLoader(
            dataset = self.dataset, batch_size = self.batch_size, shuffle = True, num_workers = self.nw, pin_memory = True
        )
    
class TrainLoader(DataLoader):
    def __init__(self,
                dataset: Dataset,
                batch: Batch,
                num_workers: int = 8)->None:
        self.dataset: Dataset = dataset
        self.batch_size: Batch = batch.train_size
        self.nw: int = num_workers
    
    # def execute(self):
    #     print ("TrainLoader: setting up training set")
    #     return torch.utils.data.DataLoader(
    #         dataset = self.dataset, batch_size = self.batch_size, shuffle = True, num_workers = self.nw, pin_memory = True
    #     )
        
class ValLoader(DataLoader):
    def __init__(self,
                dataset: Dataset,
                batch: Batch,
                num_workers: int = 8)->None:
        self.dataset: Dataset = dataset
        self.batch_size: Batch = batch.val_size
        self.nw: int = num_workers
    
    # def execute(self):
    #     print ("ValLoader: setting up validation set")
    #     return torch.utils.data.DataLoader(
    #         dataset = self.dataset, batch_size = self.batch_size, shuffle = True, num_workers = self.nw, pin_memory = True
    #     )
    
class TestLoader(DataLoader):
    def __init__(self,
                dataset: Dataset,
                batch: Batch,
                num_workers: int = 8)->None:
        self.dataset: Dataset = dataset
        self.batch_size: Batch = batch.test_size
        self.nw: int = num_workers
    
    # def execute(self):
    #     print ("TestLoader: setting up test set")
    #     return torch.utils.data.DataLoader(
    #         dataset = self.dataset, batch_size = self.batch_size, shuffle = True, num_workers = self.nw, pin_memory = True
    #     )
    
