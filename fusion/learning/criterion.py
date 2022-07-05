import torch.nn as nn
import torch
from config import ConfigSeq
from abc import abstractmethod
from config import Criteria
from fusion.utils.util import get_batch_size

cfg = ConfigSeq()

class Command(nn.Module):
    @abstractmethod
    def execute(self):
        pass

class Criterion:
    """Criterion Invoker class
    this class invokes the criterion or loss function 
    
    ........
    
    Attributes:
    ----------
    lambda_coeff: str
        value of lambda coefficient for barlow twins
    
    batch_size: int
        value of size of batch 
    
    t: int
        temperature value for NTXent loss
    
    commands: dict
        dictionary to hold the key value pairs of all commands
        
    Methods:
    -------
    register: None
        register the command with the command name
    
    execute: float
        returns the value of loss function
    """
    def __init__(self, criteria: Criteria):
        self.lambda_coeff = criteria.lamda
        self.batch_size = 0
        self.t = criteria.temp
        self.commands = {}
    
    def register(self, command_name: str, command: Command)-> None:
        self.commands[command_name] = command
    
    def execute(self, command_name: str, z1, z2)-> float:
        return self.commands[command_name].execute(z1, z2)
    
class Constants:
    temp = Criterion(cfg.criteria).t
    lambda_coeff = Criterion(cfg.criteria).lambda_coeff
    batch_size = get_batch_size(cfg)

#########== Command Interface ==#########

class BCEwithLogistLoss(Command):
    def execute(self, z1, z2): return nn.BCEWithLogitsLoss()(z1, z2)

class CrossEntropyLoss(Command):
    def execute(self, z1, z2): return nn.CrossEntropyLoss()(z1, z2)

class KLDiv(Command):
    def __init__(self):
        self.lg_sft = 0
        self.sft = 0
        
    def do(self, z1, z2):
        self.lg_sft = nn.functional.log_softmax(z1.float())
        self.sft = nn.functional.softmax(nn.functional.one_hot(z2, num_classes=lg_sft.shape[1]).float())
    
    def execute(self, z1, z2): 
        self.do(z1, z2)
        return nn.functional.kl_div(lg_sft, sft, reduction='batchmean')
        
class NTXent(Command):
    def __init__(self, temp = Constants.temp):
        super(NTXent, self).__init__()
        self.t = temp
        
    def execute(self, z1, z2):
        p_aug_norm = z1 / z1.norm(dim=1)[:, None]
        p_orig_norm = z2 / z2.norm(dim=1)[:, None]
        cat = torch.cat([p_aug_norm, p_orig_norm], axis=0)
        n_samples = len(cat)
        sim = torch.mm(cat, cat.t().contiguous())
        ex = torch.exp(sim/self.t)
        mask = ~torch.eye(n_samples, device=sim.device).bool()
        neg = ex.masked_select(mask).view(n_samples, -1).sum(dim=-1)
        pos = torch.exp(torch.sum(p_aug_norm*p_orig_norm, dim=-1)/self.t)
        pos = torch.cat([pos, pos], dim=0)
        return -torch.log(pos/neg).mean()

class BarlowTwins(Command):
    def __init__(self, lambda_coeff = Constants.lambda_coeff, batch_size=Constants.batch_size):
        super(BarlowTwins, self).__init__()
        self.batch_size= batch_size
        self.lambda_coeff = lambda_coeff

    def off_diagonal_ele(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def execute(self, z1, z2):

        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)

        cross_corr = torch.matmul(z1_norm.T, z2_norm) / self.batch_size

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        return on_diag + self.lambda_coeff * off_diag
        
        