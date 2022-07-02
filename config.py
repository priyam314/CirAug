from dataclasses import dataclass, field
import torch
from typing import List
from abc import ABC, abstractmethod
# import logging

# logging.basicConfig(
#     format = "{asctime} {levelname:<8} {message}",
#     style = "{",
#     filename = "ProLogs.log",
#     filemode = "w"
# )

class ModelName:
    """Creates the model name 
    this class uses somes attributes in order to create the model name
    
    .........
    
    Methods:
    -------
    model_name: str
        joins the encoder_name, dataname, update, description, type, index 
        to create a unique key for protrack to track
    """
    def __init__(self):
        pass
    def __call__(self, encoder_name, dataname, update, desc, typ,ind):
        return self.model_name(encoder_name, dataname, update, desc, typ,ind)
    
    def model_name(self, encoder_name, dataname, update, desc, typ,ind)->str:
        """
        >>> <encoder>_<dataname>_<update>_<type>_<desc>_<index>
        """
        update_val = "aug"
        if update: update_val = "aug"
        else: update_val = "baseline"
        if typ=="train": ind = 0
        elif typ!="train" and ind==0:
            raise Exception("index can be 0 only when type == 'train', change index value")
        return "".join([encoder_name, "_", dataname, "_", update_val, "_", typ, "_", desc, "_", str(ind)])
    
config = {
    'data':{
        'dataset_size': 0,
        'name': "stl10",
        'val_percent': 100,
        'train_percent':100,
        'test_percent': 100
    },
    'batch':{
        'train_size': 128,
        'val_size': 128,
        'test_size': 128,
        'log_interval': 40
    },
    'epoch':{
        'total': 4,
        'start': 1,
        'end': 4,
        'warmup': 4
    },
    'optimizer':{
        'lr': 0.001847,
        'min_lr': 0.0008,
        'momentum': 0.05,
        'name': 'adam',
        'beta_1': 0.9,
        'beta_2': 0.99,
        'weight_decay': 0.003,
        'weight_decouple': True,
        'rectify': True,
        'eps': 1e-7
    },
    'proaug':{
        'lamda': 3,
        'update': False
    },
    'criteria':{
        'name': 'cel',
        'temp': 0.06,
        'lamda': 0.005
    },
    'protrack':{
        'dirnm': 'ProLogs',
        'jsonm': 'protrack.json'
    },
    'hlayer':{
        'train': [512, 256],
        'test': [512,10],
        'dropout': [0.9, 0.5]
    },
    'exec':{
        'typ': 'fine-tune'
    },
    'utils':{
        'cuda': torch.cuda.is_available(),
        'index': 1,
        'desc': 'new',
        'K': 3
    },
    'model':{
        'encoder': 'mobilenet-v2',
    }
}

class Config(ABC):
    
    @abstractmethod
    def __init__(self):
        self._config = config
        
    def get(self, name1, name2):
        return self._config[name1][name2]
    
    def set(self, name1, name2, value):
        self._config[name1][name2] = value

class Data(Config):
    
    def __init__(self):
        super(Data, self).__init__()
    
    @property
    def dataset_size(self): return self.get('data', 'dataset_size')
    
    ## NAME = "stl10", "svhn", "food101"
    @property
    def name(self): return self.get('data', 'name')

    @property
    def val_percent(self): return self.get('data', 'val_percent')
    
    @property
    def train_percent(self): return self.get('data', 'train_percent')
    
    @property
    def test_percent(self): return self.get('data', 'test_percent')

class Batch(Config):
    
    def __init__(self):
        super(Batch, self).__init__()
    
    @property
    def train_size(self): return self.get('batch', 'train_size')
    
    @property
    def val_size(self): return self.get('batch', 'val_size')
    
    @property
    def test_size(self): return self.get('batch', 'test_size')
    
    @property
    def log_interval(self): return self.get('batch', 'log_interval')

class Epoch(Config):
    
    def __init__(self):
        super(Epoch, self).__init__()
    
    @property
    def total(self): return self.get('epoch', 'total')
    
    @property
    def start(self): return self.get('epoch', 'start')
    
    @property
    def end(self): return self.get('epoch', 'end')
    
    @property
    def warmup(self): return self.get('epoch', 'warmup')

class Optimizer(Config):
    
    def __init__(self):
        super(Optimizer, self).__init__()
    
    @property
    def lr(self): return self.get('optimizer', 'lr')
    
    @property
    def min_lr(self): return self.get('optimizer', 'min_lr')
    
    @property
    def momentum(self): return self.get('optimizer', 'momentum')
    
    ## name = "adam", "sgd", "adabelief", "adamw"
    @property
    def name(self): return self.get('optimizer', 'name')
    
    @property
    def beta_1(self): return self.get('optimizer', 'beta_1')

    @property
    def beta_2(self): return self.get('optimizer', 'beta_2')
    
    @property
    def weight_decay(self): return self.get('optimizer', 'weight_decay')
    
    @property
    def weight_decouple(self): return self.get('optimizer', 'weight_decouple')
    
    @property
    def rectify(self): return self.get('optimizer', 'rectify')
    
    @property
    def eps(self): return self.get('optimizer', 'eps')

class Proaug(Config):
    
    def __init__(self):
        super(Proaug, self).__init__()
    
    @property
    def lamda(self): return self.get('proaug', 'lamda')
    
    @property
    def update(self): return self.get('proaug', 'update')

class Criteria(Config):
    
    def __init__(self):
        super(Criteria, self).__init__()
    
    ## name = "ntxent", "cel", "bce", "barlow", "kldiv"
    @property
    def name(self): return self.get('criteria', 'name')
    
    @property
    def temp(self): return self.get('criteria', 'temp')
    
    @property
    def lamda(self): return self.get('criteria', 'lamda')

class Protrack(Config):
    
    def __init__(self):
        super(Protrack, self).__init__()
    
    @property
    def dirnm(self): return self.get('protrack', 'dirnm')
    
    @property
    def jsonm(self): return self.get('protrack', 'jsonm')

class Hlayer(Config):
    
    def __init__(self):
        super(Hlayer, self).__init__()
    
    @property
    def train(self): return self.get('hlayer', 'train')
    
    @property
    def test(self): return self.get('hlayer', 'test')
    
    @property
    def dropout(self): return self.get('hlayer', 'dropout')

class Exec(Config):
    
    def __init__(self):
        super(Exec, self).__init__()
    
    ## typ = "train", "fine-tune", "test"
    @property
    def typ(self): return self.get('exec', 'typ')

class Utils(Config):
    
    def __init__(self):
        super(Utils, self).__init__()
    
    @property
    def cuda(self): return self.get('utils', 'cuda')
    
    @property
    def index(self): return self.get('utils', 'index')
    
    @property
    def desc(self): return self.get('utils', 'desc')

    @property
    def k(self): return self.get('utils', 'K')

class Model(Config):
    
    def __init__(self):
        super(Model, self).__init__()
    
    @property
    def encoder(self): return self.get('model', 'encoder')
    
    @property
    def name(self): 
        return ModelName()(self.encoder, Data().name, Proaug().update,  Utils().desc, Exec().typ, Utils().index)


class ConfigSeq(Config):
    
    def __init__(self):
        super(ConfigSeq, self).__init__()
        self.Data = Data()
        self.Batch = Batch()
        self.Epoch = Epoch()
        self.Optimizer = Optimizer()
        self.Proaug = Proaug()
        self.Criteria = Criteria()
        self.Protrack = Protrack()
        self.Hlayer = Hlayer()
        self.Exec = Exec()
        self.Utils = Utils()
        self.Model = Model()
    
    @property
    def data(self): return self.Data

    @property
    def batch(self): return self.Batch

    @property
    def epoch(self): return self.Epoch

    @property
    def optimizer(self): return self.Optimizer

    @property
    def criteria(self): return self.Criteria

    @property
    def proaug(self): return self.Proaug
    
    @property
    def protrack(self): return self.Protrack
    
    @property
    def hlayer(self): return self.Hlayer

    @property
    def exec(self): return self.Exec

    @property
    def utils(self): return self.Utils

    @property
    def model(self): return self.Model

# @dataclass
# class DATA:
#     DATASET_SIZE: int = 0
#     ## NAME = "stl10", "svhn", "food101"
#     NAME: str = "stl10"
#     VAL_PERCENT: float = 100
#     TRAIN_PERCENT: float = 100
#     TEST_PERCENT: float = 100
    
# @dataclass
# class BATCH:
#     TRAIN_SIZE: int = 256
#     TEST_SIZE: int = 256
#     VAL_SIZE: int = 256
#     LOG_INTERVAL: int = 100

# @dataclass
# class EPOCH:
#     TOTAL: int = 60
#     START: int = 1
#     END: int = 60
#     WARMUP: int = 5

# @dataclass
# class OPTIM:
#     LEARNING_RATE: float = 0.002
#     LEARNING_RATE_MIN: float = 0.0015
#     MOMENTUM: float = 0.5
#     ## OPTIMIZER = "adam", "sgd", "adabelief", "adamw"
#     OPTIMIZER: str = "adam"
    # LR_SCHEDULER: str = "onecyclelr"
#     BETA_1: float = 0.93
#     BETA_2: float = 0.959
#     WEIGHT_DECAY: float = 0.00003
#     WEIGHT_DECOUPLE: bool = True
#     RECTIFY: bool = True
#     EPS: float = 1e-7

# @dataclass
# class PROAUG:
#     LAMDA: int = 3
#     UPDATE: bool = True

# @dataclass
# class LOSS:
#     ## CRITERIA = "ntxent", "cel", "bce", "barlow", "kldiv"
#     CRITERIA: str = "ntxent"
#     TEMPERATURE: float = 0.07    # NTXENT HYPERPARAMETER
#     LAMBDA_COEFF: float = 0.005  # BARLOW HYPERPARAMETER

# @dataclass
# class PROTRACK:
#     DIRNM: str = 'ProLogs'
#     JSONM: str = 'protrack.json'

# @dataclass
# class HLAYER:
#     TRAIN = [512, 256]
#     TEST = [512, 10]
#     DROPOUT = [0.9, 0.5]

# @dataclass
# class EXECUTION:
#     ## TYPE = "train", "fine-tune", "test"
#     TYPE: str = "train"
    
# @dataclass
# class UTILS:
#     SEED: int = 2
#     CUDA: bool = torch.cuda.is_available()
#     INDEX: int = 6
#     DESC: str = "new"
#     K: int = 3

# @dataclass
# class MODEL:
#     ENCODER: str = "mobilenet-v2"
#     NAME: str = model_name(ENCODER, DATA.NAME, PROAUG.UPDATE,  UTILS.DESC, EXECUTION.TYPE, UTILS.INDEX)

# @dataclass
# class CONFIG:
#     data: DATA = DATA
#     batch: BATCH = BATCH
#     epoch: EPOCH = EPOCH
#     optim: OPTIM = OPTIM
#     proaug: PROAUG = PROAUG
#     loss: LOSS = LOSS
#     protrack: PROTRACK = PROTRACK
#     hlayer: HLAYER = HLAYER
#     execution: EXECUTION = EXECUTION
#     model: MODEL = MODEL
#     utils: UTILS = UTILS
