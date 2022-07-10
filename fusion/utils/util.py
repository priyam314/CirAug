import torch
from termcolor import colored
from shared.config import Config, Model

def get_batch_size(config: Config):
    if config.exec.typ == "train": return config.batch.train_size
    elif config.exec.typ == "fine-tune": return config.batch.val_size
    elif config.exec.typ == "test": return config.batch.test_size
    
def changeName(withname: str, instring: str, searchof: str)-> str:
    for string in searchof:
        if string in instring:
            return instring.replace(withname, string)
    raise Exception("string not found in searchof")

def trainOf(model: Model)->str:
    """replace subString
    Replaces the 'fine-tune' in name with 'train' and add 0 at end
    
    .......
    
    Parameters:
    ----------
    name: str
        word whose substring would be replaced
    
    Returns:
    -------
    str
        string after the certain subString has been replaced
    """
    return model.name.replace("fine-tune", "train")[:-1]+"0"

def fineTuneOf(model: Model)->str:
    """replace subString
    Replaces the 'test' in name with 'fine-tune'
    
    .......
    
    Parameters:
    ----------
    name: str
        word whose substring would be replaced
    
    Returns:
    -------
    str
        string after the certain subString has been replaced
    """
    return model.name.replace("test","fine-tune")

def loader(model, pt, name, typ):
    """Loads the weight of model 
    Loads the weight of model named name
    
    ..........
    
    Parameters:
    ----------
    model: nn.Module
        pyTorch model that has blueprint of the struture but no weights
    
    pt: ProTrack
        object of ProTrack 
    
    name: str
        name of the model in ProTrack (Key)
        
    typ: str
        type of execution
        
    Returns:
    -------
    model: nn.Module
        after loading the weights in model, we return the model
    """
    print ("{}:   Loading Weights of {}".format(typ, name))
    model.load_state_dict(torch.load(pt.data(name, 'model_path').value))
    return model

def ifExistLoad(model, pt, modelObj, typ):
    """checks if certain model already exist or not
    If model exist in the protrack, then it loads the weights 
    If model does not exist, it returns the skeleton of the model
    
    ..........
    
    Parameters:
    ----------
    model: nn.Module
        pyTorch model that has blueprint of the struture but no weights
    
    pt: ProTrack
        object of ProTrack 
    
    modelObj: Model
        Model class from config
        
    typ: str
        type of execution
        
    Returns:
    -------
    model: nn.Module
        after loading the weights in model, we return the model
    """
    if pt.data(modelObj.name, 'model_path').bool:
        return loader(model, pt, modelObj.name, typ)
    print ("{}:   {} does not exist in ProLogs/protrack.json, initiating new one".format(typ, modelObj.name))
    return model

def Print(msg: str)->None:
    """Prints message in red color
    uses colored function
    """
    print()
    print(colored(msg, 'red'))
    print()
    