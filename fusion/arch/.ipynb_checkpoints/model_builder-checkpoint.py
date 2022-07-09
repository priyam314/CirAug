from config import Model, ConfigSeq, Hlayer
from fusion.arch.encoder import Encoder
from torch import nn
from protrack import ProTrack
from abc import ABC, abstractmethod
from fusion.utils.util import ifExistLoad
from fusion.utils.util import trainOf, fineTuneOf

cfg = ConfigSeq()

class Command(ABC):
    def __init__(self, encoder, model, pt, hlayer):
        self.encoder = encoder
        self._model = model
        self.hlayer = hlayer
        self.pt = pt
        self.skeleton = 0
        self._name = ""
    @abstractmethod
    def execute(self, projector, word):
        pass
    @property
    def model(self): return self.skeleton
    @property
    def name(self): return self._name

class ModelBuilder:
    def __init__(self, 
                encoder: Encoder,
                model: Model, 
                pt: ProTrack,
				hlayer: Hlayer
                )-> None:
        self.encoder = encoder
        self.model = model
        self.pt = pt
        self.commands = {}
    
    def register(self, command_name: str, command: Command):
        self.commands[command_name] = command(encoder = self.encoder, 
											  model = self.model, 
											  pt = self.pt, 
											  hlayer = cfg.hlayer)
    
    def execute(self, command_name: str, projector: nn.Module, word: str):
        return self.commands[command_name].execute(projector, word)

    def __repr__(self):
        return f"{self.__class__.__name__}"

class TrainModel(Command):
    def execute(self, projector, word):
        self.skeleton = ifExistLoad(self.encoder.execute(self._model, projector), self.pt, self._model, word)
        self._name = word+"Model"
        return self

class Fine_TuneModel(Command):
    def execute(self, projector, word):
        self.skeleton = ifExistLoad(self.encoder.execute(self._model, projector), self.pt, self._model, word)
        self._name = word+"Model"
        return self

class TestModel(Command):
    def execute(self, projector, word):
        self.skeleton = ifExistLoad(self.encoder.execute(self._model, projector), self.pt, self._model, word)
        self._name = word+"Model"
        return self
    