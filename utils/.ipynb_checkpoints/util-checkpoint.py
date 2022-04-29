import torch
from termcolor import colored

def changeName(withname, instring, searchof):
    for string in searchof:
        if string in instring:
            return instring.replace(withname, string)
    raise Exception("string not found in searchof")

def trainOf(name: str)->str:
    return name.replace("fine-tune", "train")[:-1]+"0"

def fineTuneOf(name: str)->str:
    return name.replace("test","fine-tune")

def loader(model, pt, name, typ):
    print ("{}:   Loading Weights of {}".format(typ, name))
    model.load_state_dict(torch.load(pt.data(name, 'model_path').value))
    return model

def ifExistLoad(model, pt, name, typ):
    if pt.data(name, 'model_path').bool:
        return loader(model, pt, name, typ)
    print ("{}:   {} does not exist in ProLogs/protrack.json, initiating new one".format(typ, name))
    return model

def Print(msg: str):
    print()
    print(colored(msg, 'red'))
    print()
    