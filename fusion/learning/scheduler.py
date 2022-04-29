import torch
class Scheduler:
    def __init__(self, method: str = "cosineannealingwarmrestarts"):
        
        self.method = method
    
    def __call__(self):
        
        if self.method == "cosineannealingwarmrestarts":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        if self.method == "onecyclelr":
            return torch.optim.lr_scheduler.OneCycleLR