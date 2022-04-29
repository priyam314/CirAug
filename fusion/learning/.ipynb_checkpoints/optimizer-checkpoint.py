import torch.nn as nn 
import torch.optim as optim
# !pip install adabelief-pytorch==0.2.0
from adabelief_pytorch import AdaBelief
from config import Optimizer

class Optim:
    def __init__(self, name: str = "adam"):
        self.optim = name
    def __call__(self, model: nn.Module, config: Optimizer):
        if self.optim == "adam":
            return optim.Adam(model.parameters(), 
                              lr = config.lr, 
                              betas=(config.beta_1, config.beta_2), 
                              weight_decay=config.weight_decay
                             )
        elif self.optim == "sgd":
            return optim.SGD(model.parameters(), 
                             lr = config.lr, 
                             momentum=config.momentum, 
                             weight_decay=config.weight_decay
                            )
        elif self.optim == "adabelief":
            return AdaBelief(model.parameters(), 
                             lr = config.lr, 
                             betas=(config.beta_1, config.beta_2), 
                             weight_decouple=config.weight_decouple, 
                             eps=config.eps*config.eps, 
                             rectify=config.rectify, 
                             weight_decay=config.weight_decay
                            )
        elif self.optim == "adamw":
            return optim.AdamW(model.parameters(),
                              lr = config.lr,
                              betas = (config.beta_1, config.beta_2),
                              eps = config.eps,
                              weight_decay = config.weight_decay,
                              amsgrad = True
                              )