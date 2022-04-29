from fusion.execute import Executioner, Train, Test, FineTune
from config import Exec, ConfigSeq
from typing import List

cfg = ConfigSeq()

class PipeLine:
    def __init__(self, typ: str):
        self.object = {}
        self.type = typ
    def add(self, objec: Executioner):
        self.object.update({objec.name:objec})
    def set_on_start(self):
        self.object[self.type].run(cfg.epoch)
        if self.type == "fine-tune":
            self.object[self.type].add_in_pipe(self.object['test']).run(cfg.epoch)

            