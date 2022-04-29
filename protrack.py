from typing import Any
import json
from pprint import pprint, pformat
import torch.nn as nn
from pathlib import Path
import copy
import os
from config import ConfigSeq
import torch

cfg = ConfigSeq()

class ProTrack:
    def __init__(self):
        self.trackDict: dict = {}
        self.arch: nn.Module = None
        self.Bool: bool = False
        self.Value = 0
        self.dirnm = cfg.protrack.dirnm
        self.jsonm = cfg.protrack.jsonm
        
    
    def __repr__(self):
        return pformat(self.trackDict)
    
    def init(self, typ: str=""):
        if not os.path.isdir(self.dirnm):
            os.mkdir(self.dirnm)
            with open(self.dirnm + '/' + self.jsonm, 'w+') as f:
                f.write('{}')
            print ("==> ProTrack init {}: {}/{} created".format(typ, self.dirnm, self.jsonm))
            return self
        elif os.path.isdir(self.dirnm):
            print ("==> ProTrack init {}: {}/{} exists".format(typ, self.dirnm, self.jsonm))
            return self

    def log_params(self, value: Any, name: str, model_name):
        self.trackDict[model_name]._log_params(value, name)
        return self
    
    def log_aug(self, augs, model_name: str):
        self.trackDict[model_name]._log_aug(augs)
        return self
            
    def log_config_params(self, config, model_name):
        self.trackDict[model_name]._log_config_params(config)
        return self

    def log_metric(self, value: Any, name: str, model_name):
        self.trackDict[model_name]._log_metric(value, name)
        return self
      
    def log_model(self, model, model_name):
        self.trackDict[model_name]._log_model(model)
        return self

    def log_model_name(self, model_name):
        if model_name in self.trackDict:
            print ("==> ProTrack log_model_name: {} already exist".format(model_name))
            return self
        self.trackDict[model_name] = IndiLogger(model_name)
        print ("==> ProTrack log_model_name: logged {}".format(model_name))
        return self
    
    def remove(self, model_name):
        del self.trackDict[model_name]
        self.save()
        print ("==> ProTrack {} has been deleted".format(model_name))
        return self.show
    
    def data(self, model_name: str = "", attr: str = ""):
        if model_name==attr=="":
            self.Value = self.trackDict
            return self
        elif model_name!="" and attr=="":
            if model_name in self.trackDict:
                self.Value = self.trackDict[model_name]
                self.Bool = True
                return self
            self.Value = ""
            self.Bool = False
            return self
        elif model_name!="" and attr!="":
            if model_name in self.trackDict:
                if attr in self.trackDict[model_name]:
                    self.Value = self.trackDict[model_name][attr]
                    if self.trackDict[model_name][attr] == '':
                        self.Bool = False
                    else:
                        self.Bool = True
                    return self
            self.Value = ""
            self.Bool = False
            return self
        
    @property
    def bool(self):
        return self.Bool
    
    @property
    def value(self):
        return self.Value
    
    @property
    def show(self):
        pprint (self.trackDict)
        
    def save(self):
        self.init("save")
        with open(self.dirnm  + '/' + self.jsonm, 'w+') as fp:
            trialSaveDict = self.trackDict.copy()
            for model_name, logger_details in trialSaveDict.items():
                trialSaveDict[model_name] = logger_details.state_dict()
            json.dump(trialSaveDict, fp)
            print('==> ProTrack saved to {}/{}'.format(self.dirnm, self.jsonm))
        
    def load(self):
        self.init("load")
        with open(self.dirnm + '/' + self.jsonm, 'r') as fp:
            trialLoadDict = json.loads(fp.read()) 
            for model_name, meta_data in trialLoadDict.items():
                self.trackDict[model_name] = IndiLogger(model_name).load(meta_data)
            print ("==> ProTrack Loaded from {}/{}".format(self.dirnm, self.jsonm))
        return self
            
    def load_model(self,model_name):
        return torch.load_state_dict(torch.load(self.trackDict[model_name]["model_path"]))

class IndiLogger:
    
    def __init__(self, model_name: str):
        self.loggerDict = {"params":{}, "metric":{}, "aug_params":{}, "model_path":""}
        self.model_name = model_name
    
    def __repr__(self):
        return pformat(self.loggerDict)
    
    def __contains__(self, name):
        if name in self.loggerDict:
            return True
        return False
    
    def __getitem__(self, name):
        if name in self.loggerDict:
            return self.loggerDict[name]
        return -1
    
    def state_dict(self):
        return self.loggerDict
    
    def _log_params(self, value: Any, name: str):
        self.loggerDict['params'][name] = value
    
    def _log_aug(self, augs):
        self.loggerDict['aug_params'] = augs
            
    def _log_config_params(self, config):
        for name in config._config.keys():
            self._log_params(config._config[name], name)
        # for category in dir(config):
        #     if not category.startswith("__") and not category == "get":
        #         for name in dir(getattr(config, category)):
        #             if not name.startswith("__") and not category == "get":
        #                 self._log_params(getattr(getattr(config, category), name), category+"."+name)

    def _log_metric(self, value: Any, name: str):
        if isinstance(self.loggerDict['metric'].get(name, None), type(None)):
            self.loggerDict['metric'][name] = []
        self.loggerDict['metric'][name].append(value)
    
    def _log_model(self, model):
        self.arch = model
        self.loggerDict["model_path"] = cfg.protrack.dirnm + '/' + str(self.model_name) + '.pt'
        torch.save(self.arch.state_dict(), self.loggerDict['model_path'])
        print ("==> ProTrack log_model: model weights saved to {}".format(cfg.protrack.dirnm))
    
    def load(self, logger_details: dict):
        for attr, value in logger_details.items():
            self.loggerDict[attr] = value
        return self