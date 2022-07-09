import fusion as fuse
import ProAug
import torch
import time
from log import setter
from torchvision import transforms
from protrack import ProTrack
from config import ConfigSeq, Data, Epoch
from abc import ABC, abstractmethod
from fusion.utils.data import DataLoader

cfg = ConfigSeq()
logger = setter(__name__)

class Executioner(ABC):
    def __init__(self):
        self.pt: ProTrack = ProTrack()
        self.t_time = {'total_aug_time':0, 'total_model_time':0, 'total_lossf_time':0, 
                  'total_lossb_time':0, 'total_optim_time':0}
        self.epoch = 0
        self.loss = 0
        self.total_loss = 0
        self.once = True
        self.corrects = 0
    
    def __repr__(self):
        return f"{self.__class__.__name__}"
    
    @abstractmethod
    def execute(self):
        pass
    
    @property
    def proTrack(self): return self.pt
    
    def print_stats(self, typ="train", batch_idx=0, len_data=0, dataset_size=0):
        print ()
        logger.info('{} Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
            typ, self.epoch, batch_idx * len_data, dataset_size,
            (100 * batch_idx * len_data) / dataset_size, self.loss.data))
    
    def oneEpochRun(self, epoch):
        time_epoch_start = torch.cuda.Event(enable_timing=True)
        time_epoch_end = torch.cuda.Event(enable_timing=True)
        time_epoch_start.record()
        # for execute in self.runable:
        
        self.execute(epoch)
        
        torch.cuda.synchronize()
        time_epoch_end.record()
        print ("*********")
        print ("Time Taken in Epoch {} is {:.1f}ms".format(epoch, time_epoch_start.elapsed_time(time_epoch_end)))
        print ("*********")
        self.pt[0].log_params(epoch, "EPOCH", cfg.model.name)
        self.pt[0].log_model(self.model, cfg.model.name)
        self.pt[0].log_aug(self.augs.get_parameters_value("all"), cfg.model.name)
        self.pt[0].save()
        
    def run(self, config: Epoch):
        for epoch in range(config.start, config.end+1): 
            self.oneEpochRun(epoch)
        
    def compositeRun(self, execution, config: Epoch):
        for epoch in range(config.start, config.end+1):
            self.oneEpochRun(epoch)
            execution.oneEpochRun(epoch)
    
    # def add_in_pipe(self, objec):
    #     self.runable.append(objec.execute)
    #     return self

class TrainExecution(Executioner):
    def __init__(self,
                 model: torch.nn.Module, 
                 dataloader: DataLoader, 
                 Augs: ProAug.augSeq.AugSeq,
                 optim: fuse.learning.optimizer.Optim,
                 criterion: fuse.learning.criterion.Criterion,
                 # scheduler: fuse.learning.scheduler.Scheduler,
                 protrack: ProTrack,
                 data: Data
    )->None:
        super(TrainExecution, self).__init__()
        self.model = model
        self.dataloader = dataloader.data
        self.augs = Augs
        self.optimizer = optim.execute(cfg.optimizer.name, self.model, cfg.optimizer),
        self.criterion = criterion,
        # self.scheduler = scheduler,
        self.pt = protrack,
        self.data_percent = data.train_percent
    
    def execute(self, epoch):
        self.epoch = epoch
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.dataloader):
            if not (batch_idx*len(data)*100)/len(self.dataloader.dataset) <= self.data_percent:
                print ("BREAK: model has been trained on {}% of train data".format(self.data_percent))
                break
            if cfg.utils.cuda: data = data.cuda()            
            x_aug, x = self.augs.apply_random(data, self.epoch, update = cfg.proaug.update), data
            self.optimizer[0].zero_grad()
            projector_p_aug, projector_p_orig = self.model(x_aug), self.model(x)
            self.loss = self.criterion[0].execute(cfg.criteria.name, projector_p_aug, projector_p_orig)
            self.total_loss += self.loss.data
            self.loss.backward()
            self.optimizer[0].step()
            if batch_idx % cfg.batch.log_interval == 0:
                self.print_stats('train', batch_idx, len(data), len(self.dataloader.dataset))
            # self.scheduler.step()
            # self.scheduler_list.append(
            #     self.optim.param_groups[0]["lr"]
            # )

        avg_epoch_loss = self.total_loss/batch_idx
                
        self.pt[0].log_params(batch_idx, "TOTAL_BATCHES", cfg.model.name)
        self.pt[0].log_metric(avg_epoch_loss.item(), "AVG_EPOCH_LOSS", cfg.model.name)

        self.total_loss = 0
    
    @staticmethod
    def name():
        return "trainExecution"

class FineTuneExecution(Executioner):
    def __init__(self,
                 model: torch.nn.Module, 
                 projector: fuse.arch.projector.TestProjector,
                 dataloader: DataLoader, 
                 Augs: ProAug.augSeq.AugSeq,
                 optim: fuse.learning.optimizer.Optim,
                 criterion: fuse.learning.criterion.Criterion,
                 # scheduler: fuse.learning.scheduler.Scheduler,
                 protrack: ProTrack,
                 data: Data
    )->None:
        super(FineTuneExecution, self).__init__()
        self.model = model
        self.projector = projector
        self.dataloader = dataloader.data
        self.augs = Augs
        self.optimizer = optim.execute(cfg.optimizer.name, self.model, cfg.optimizer),
        self.criterion = criterion,
        # self.scheduler = scheduler,
        self.pt = protrack,
        self.data_percent = data.val_percent
    
    def execute(self, epoch):
        self.epoch = epoch
        self.model = self.freeze(self.model)
        self.model.train()
        self.model.cuda()
        for batch_idx, (data, labels) in enumerate(self.dataloader):
            if not (batch_idx*len(data)*100)/len(self.dataloader.dataset) <= self.data_percent:
                print ("BREAK: model has been finetuned on {}% of val data".format(self.data_percent))
                break
            if cfg.utils.cuda: 
                data = data.cuda()
                labels = labels.cuda()
            self.optimizer[0].zero_grad()
            outputs = self.model(data)
            _, preds = torch.max(outputs, 1)
            self.loss = self.criterion[0].execute(cfg.criteria.name, outputs, labels)
            self.loss.backward()
            self.optimizer[0].step()
            self.total_loss += self.loss.data
            self.corrects += torch.sum(preds == labels.data)
            if batch_idx % cfg.batch.log_interval == 0:
                self.print_stats('Fine-Tune', batch_idx, len(data), len(self.dataloader.dataset))
            # self.scheduler.step()
            # self.scheduler_list.append(
            #     self.optim.param_groups[0]["lr"]
            # )
        avg_epoch_loss = self.total_loss / batch_idx
        avg_epoch_acc = self.corrects.double() / len(self.dataloader.dataset) * 100
        self.pt[0].log_metric(avg_epoch_loss.item(), "AVG_EPOCH_LOSS_FINE-TUNE", cfg.model.name)
        self.pt[0].log_metric(avg_epoch_acc.item(), "AVG_EPOCH_ACCURACY_FINE-TUNE", cfg.model.name)
        print (f'Fine-Tune Loss: {avg_epoch_loss:.4f} Acc: {avg_epoch_acc:.4f}')
        self.total_loss = 0
        self.corrects = 0
        self.loss = 0
    
    def freeze(self, model):
        if self.once:
            for f in model.features.parameters():
                f.required_grad = False
            model.classifier = self.projector.projector(model.classifier[0].in_features)
            self.once = False
        return model
    
    @staticmethod
    def name():
        return "fine-tuneExecution"

class TestExecution(Executioner):
    def __init__(self,
                 model: torch.nn.Module, 
                 dataloader: DataLoader, 
                 Augs: ProAug.augSeq.AugSeq,
                 optim: fuse.learning.optimizer.Optim,
                 criterion: fuse.learning.criterion.Criterion,
                 # scheduler: fuse.learning.scheduler.Scheduler,
                 protrack: ProTrack,
                 data: Data
    )->None:
        super(TestExecution, self).__init__()
        self.model = model
        self.dataloader = dataloader.data
        self.augs = Augs
        self.optimizer = optim.execute(cfg.optimizer.name, self.model, cfg.optimizer),
        self.criterion = criterion,
        # self.scheduler = scheduler,
        self.pt = protrack,
        self.data_percent = data.test_percent
    
    def execute(self, epoch):
        self.epoch = epoch
        self.model.eval()
        for batch_idx, (data, labels) in enumerate(self.dataloader):
            if not (batch_idx*len(data)*100)/len(self.dataloader.dataset) <= self.data_percent:
                print ("BREAK: model has been tested on {}% of test data".format(self.data_percent))
                break
            if cfg.utils.cuda: 
                data = data.cuda()
                labels = labels.cuda()
            self.optimizer[0].zero_grad()
            outputs = self.model(data)
            _, preds = torch.max(outputs, 1)
            self.loss = self.criterion[0].execute(cfg.criteria.name, outputs, labels)
            self.total_loss += self.loss.data
            self.corrects += torch.sum(preds==labels.data)
            if batch_idx % cfg.batch.log_interval == 0:
                self.print_stats('Test', batch_idx, len(data), len(self.dataloader.dataset))
        avg_epoch_loss = self.total_loss / batch_idx
        avg_epoch_acc = self.corrects.double() / len(self.dataloader.dataset) * 100
        self.pt[0].log_metric(avg_epoch_loss.item(), "AVG_EPOCH_LOSS_TEST", cfg.model.name)
        self.pt[0].log_metric(avg_epoch_acc.item(), "AVG_EPOCH_ACCURACY_TEST", cfg.model.name)
        print (f'Test Loss: {avg_epoch_loss:.4f} Acc: {avg_epoch_acc:.4f}')
        self.total_loss = 0
        self.corrects = 0 
        self.loss = 0
    
    @staticmethod
    def name():
        return "testExecution"