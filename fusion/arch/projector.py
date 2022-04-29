from typing import List
import torch.nn as nn
from config import ConfigSeq, Hlayer

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
    
class TrainProjector(nn.Module):
    def __init__(self, 
                 layerList: Hlayer
                ):
        
        super(TrainProjector, self).__init__()
        self.hlayerList = layerList.train
        self.dropoutList = layerList.dropout
        self.projector = nn.Sequential()
    
    def forward(self, i_f = 12):
        return get_projector(self.hlayerList, self.dropoutList, self.projector, i_f)
 
    def __call__(self, i_f = 12):
        return self.forward(i_f)
        
class TestProjector(nn.Module):
    
    def __init__(self, 
                 layerList: Hlayer
                ):
        
        super(TestProjector, self).__init__()
        self.hlayerList = layerList.test
        self.dropoutList = layerList.dropout
        self.projector = nn.Sequential()
        
    def forward(self, i_f = 12):
        return get_projector(self.hlayerList, self.dropoutList, self.projector, i_f)
 
    def __call__(self, i_f = 12):
        return self.forward(i_f)
    
        
    