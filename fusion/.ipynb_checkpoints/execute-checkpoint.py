from torchvision import transforms
from protrack import ProTrack
from config import ConfigSeq, Data, Epoch
from abc import ABC, abstractmethod

import fusion as fuse
import ProAug
import torch
import time

cfg = ConfigSeq()

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
        self.runable = [self.execute]
    
    @abstractmethod
    def execute(self):
        pass
    
    @property
    def proTrack(self):
        return self.pt
    
    def print_stats(self, typ="train", batch_idx=0, len_data=0, dataset_size=0):
        print ()
        print('{} Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
            typ, self.epoch, batch_idx * len_data, dataset_size,
            (100 * batch_idx * len_data) / dataset_size, self.loss.data))
    
    def oneEpochRun(self, epoch):
        time_epoch_start = torch.cuda.Event(enable_timing=True)
        time_epoch_end = torch.cuda.Event(enable_timing=True)
        time_epoch_start.record()
        for execute in self.runable:
            execute(epoch)
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
        return 
    
    def add_in_pipe(self, objec):
        self.runable.append(objec.execute)
        return self

class Train(Executioner):
    def __init__(self,
                 model: torch.nn.Module, 
                 dataloader: torch.utils.data.DataLoader, 
                 Augs: ProAug.augSeq.AugSeq,
                 optim: fuse.learning.optimizer.Optim,
                 criterion: fuse.learning.criterion.Criterion,
                 scheduler: fuse.learning.scheduler.Scheduler,
                 protrack: ProTrack,
                 data: Data
    )->None:
        super(Train, self).__init__()
        self.model = model
        self.dataloader = dataloader
        self.augs = Augs
        self.optimizer = optim(self.model, cfg.optimizer),
        self.criterion = criterion,
        self.scheduler = scheduler,
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
            self.loss = self.criterion[0](projector_p_aug, projector_p_orig)
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
    
    @property
    def name(self):
        return "train"

class FineTune(Executioner):
    def __init__(self,
                 model: torch.nn.Module, 
                 projector: fuse.arch.projector.TestProjector,
                 dataloader: torch.utils.data.DataLoader, 
                 Augs: ProAug.augSeq.AugSeq,
                 optim: fuse.learning.optimizer.Optim,
                 criterion: fuse.learning.criterion.Criterion,
                 scheduler: fuse.learning.scheduler.Scheduler,
                 protrack: ProTrack,
                 data: Data
    )->None:
        super(FineTune, self).__init__()
        self.model = model
        self.projector = projector
        self.dataloader = dataloader
        self.augs = Augs
        self.optimizer = optim(self.model, cfg.optimizer),
        self.criterion = criterion,
        self.scheduler = scheduler,
        self.pt = protrack,
        self.data_percent = data.val_percent
    
    def execute(self, epoch):
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
            self.loss = self.criterion[0](outputs, labels)
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
            model.classifier = self.projector(model.classifier[0].in_features)
            self.once = False
        return model
    
    @property
    def name(self):
        return "fine-tune"

class Test(Executioner):
    def __init__(self,
                 model: torch.nn.Module, 
                 dataloader: torch.utils.data.DataLoader, 
                 Augs: ProAug.augSeq.AugSeq,
                 optim: fuse.learning.optimizer.Optim,
                 criterion: fuse.learning.criterion.Criterion,
                 scheduler: fuse.learning.scheduler.Scheduler,
                 protrack: ProTrack,
                 data: Data
    )->None:
        super(Test, self).__init__()
        self.model = model
        self.dataloader = dataloader
        self.augs = Augs
        self.optimizer = optim(self.model, cfg.optimizer),
        self.criterion = criterion,
        self.scheduler = scheduler,
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
            self.loss = self.criterion[0](outputs, labels)
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
    
    @property
    def name(self):
        return "test"
        
# class Executioner:
#     def __init__(self, 
#                  model: torch.nn.Module, 
#                  projector: fuse.arch.projector.Projector,
#                  train_dataloader: torch.utils.data.DataLoader, 
#                  val_dataloader: torch.utils.data.DataLoader, 
#                  test_dataloader: torch.utils.data.DataLoader, 
#                  Augs: ProAug.augSeq.AugSeq,
#                  optim: fuse.learning.optimizer.Optim,
#                  criterion: fuse.learning.criterionCriterion,
#                  scheduler: fuse.learning.scheduler.Scheduler,
#                  protrack: ProTrack,
#                  config: CONFIG,
#                  data_percent: float):

#         self.model: torch.nn.Module = model
#         self.typ: str = "train"
#         self.train_dataloader: torch.utils.data.DataLoader = train_dataloader
#         self.val_dataloader: torch.utils.data.Dataloader = val_dataloader
#         self.test_dataloader: torch.utils.data.DataLoader = test_dataloader
#         self.optim: fuse.learning.optimizer.Optim = optim
#         self.criterion: fuse.learning.criterion.Criterion = criterion
#         self.loss: torch.Tensor = torch.Tensor(0)
#         self.test_loss: torch.Tensor = torch.Tensor(0)
#         self.correct: int = 0
#         self.epoch: int = 1
#         self.total_loss: float = 0
#         self.pt: ProTrack = protrack
#         self.train_data_length: int = len(self.train_dataloader.dataset)
#         self.val_data_length: int = len(self.val_dataloader.dataset)
#         self.test_data_length: int = len(self.test_dataloader.dataset)
#         self.config: CONFIG = config
#         self.once: bool = True
#         self.projector: fuse.arch.projector.Projector = projector
#         self.Augs = Augs
#         self.val_percent: float = data_percent.VAL_PERCENT
#         self.train_percent: float = data_percent.TRAIN_PERCENT
#         self.test_percent: float = data_percent.TEST_PERCENT
#         self.scheduler = scheduler 
#         Scheduler(cfg.optim.LR_SCHEDULER)()(self.optim, max_lr = 9e-2, epochs = cfg.epoch.TOTAL, steps_per_epoch = self.check_exec_steps+1, last_epoch = -1)
#         self.scheduler_list = []
    
#     @property
#     def proTrack(self):
#         return self.pt
    
#     def print_stats(self, typ="train", batch_idx=0, len_data=0, dataset_size=0):
#         print('{} Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
#             typ, self.epoch, batch_idx * len_data, dataset_size,
#             (100 * batch_idx * len_data) / dataset_size, self.loss.data))
    
#     def train(self):
#         self.model.train()
        
#         t_time = {'total_aug_time':0, 'total_model_time':0, 'total_lossf_time':0, 
#                   'total_lossb_time':0, 'total_optim_time':0}
#         batch_idx = 0

#         for batch_idx, (data, target) in enumerate(self.train_dataloader):
            
#             if not (batch_idx*len(data)*100)/self.train_data_length <= self.train_percent:
#                 print ("BREAK: model has been trained on {}% of train data".format(self.train_percent))
#                 break
            
#             if self.config.utils.CUDA: data = data.cuda()            
            
#             x_aug, x = self.Augs.apply_random(data, self.epoch, update = self.config.proaug.UPDATE), data

#             self.optim.zero_grad()
            
#             projector_p_aug, projector_p_orig = self.model(x_aug), self.model(x)

#             self.loss = self.criterion(projector_p_aug, projector_p_orig)
             
#             self.total_loss += self.loss.data
            
#             self.loss.backward()
            
#             self.optim.step()

#             if batch_idx % self.config.batch.LOG_INTERVAL == 0:
#                 self.print_stats('train', batch_idx, len(data), self.train_data_length)
#             self.scheduler.step()
#             self.scheduler_list.append(
#                 self.optim.param_groups[0]["lr"]
#             )

#         avg_epoch_loss = self.total_loss/batch_idx
                
#         self.pt.log_params(batch_idx, "TOTAL_BATCHES", self.config.model.NAME)
#         self.pt.log_metric(avg_epoch_loss.item(), "AVG_EPOCH_LOSS", self.config.model.NAME)

#         self.total_loss = 0
    
#     def fine_tune(self):
        
#         self.model = self.freeze(self.model)
#         self.model.train()
                
#         self.total_loss = 0.0
#         self.corrects = 0

#         for batch_idx, (data, labels) in enumerate(self.val_dataloader):
            
#             if not (batch_idx*len(data)*100)/self.val_data_length <= self.val_percent:
#                 print ("BREAK: model has been finrtuned on {}% of val data".format(self.val_percent))
#                 break
                
#             if self.config.utils.CUDA: 
#                 data = data.cuda()
#                 labels = labels.cuda()
#                 self.model = self.model.cuda()

#             self.optim.zero_grad()
            
#             outputs = self.model(data)
#             _, preds = torch.max(outputs, 1)
            
#             self.loss = self.criterion(outputs, labels)
#             self.loss.backward()
            
#             self.optim.step()
            
#             self.total_loss += self.loss.data
#             self.corrects += torch.sum(preds == labels.data)

#             if batch_idx % self.config.batch.LOG_INTERVAL == 0:
#                 self.print_stats('Fine-Tune', batch_idx, len(data), self.val_data_length)
            
#             self.scheduler.step()
#             self.scheduler_list.append(
#                 self.optim.param_groups[0]["lr"]
#             )

#         avg_epoch_loss = self.total_loss / batch_idx
#         avg_epoch_acc = self.corrects.double() / self.val_data_length * 100

#         self.pt.log_metric(avg_epoch_loss.item(), "AVG_EPOCH_LOSS_FINE-TUNE", self.config.model.NAME)
#         self.pt.log_metric(avg_epoch_acc.item(), "AVG_EPOCH_ACCURACY_FINE-TUNE", self.config.model.NAME)
        
#         print ()
#         print (f'Fine-Tune Loss: {avg_epoch_loss:.4f} Acc: {avg_epoch_acc:.4f}')
#         print ()
        
#         self.total_loss = 0
#         self.corrects = 0

#     def test(self):
#         self.model.eval()
#         self.total_loss = 0.0
#         self.corrects = 0
#         for batch_idx, (data, labels) in enumerate(self.test_dataloader):
#             if not (batch_idx*len(data)*100)/self.test_data_length <= self.test_percent:
#                 print ("BREAK: model has been tested on {}% of test data".format(self.test_percent))
#                 break
                
#             if self.config.utils.CUDA: 
#                 data = data.cuda()
#                 labels = labels.cuda()
#                 self.model = self.model.cuda()

#             self.optim.zero_grad()
            
#             outputs = self.model(data)
#             _, preds = torch.max(outputs, 1)
#             self.loss = self.criterion(outputs, labels)
            
#             self.total_loss += self.loss.data
#             self.corrects += torch.sum(preds==labels.data)

#             if batch_idx % self.config.batch.LOG_INTERVAL == 0:
#                 self.print_stats('Test', batch_idx, len(data), self.test_data_length)

#         avg_epoch_loss = self.total_loss / batch_idx
#         avg_epoch_acc = self.corrects.double() / self.test_data_length * 100

#         self.pt.log_metric(avg_epoch_loss.item(), "AVG_EPOCH_LOSS_TEST", self.config.model.NAME)
#         self.pt.log_metric(avg_epoch_acc.item(), "AVG_EPOCH_ACCURACY_TEST", self.config.model.NAME)
        
#         print (f'Test Loss: {avg_epoch_loss:.4f} Acc: {avg_epoch_acc:.4f}')
        
#         self.total_loss = 0
#         self.corrects = 0 
    
#     def freeze(self, model):
#         if self.once:
#             for f in model.features.parameters():
#                 f.required_grad = False
#             model.classifier = self.projector(model.classifier[0].in_features)
#             self.once = False
#         return model
    
#     def run(self, typ, start: int = 1, epochs: int = 10):
#         self.typ = typ
#         for epoch in range(start, epochs+1):
#             if epoch > self.config.epoch.END:
#                 break
#             time_epoch_start = torch.cuda.Event(enable_timing=True)
#             time_epoch_end = torch.cuda.Event(enable_timing=True)
            
#             time_epoch_start.record()
#             self.epoch = epoch
            
#             if self.typ == "train": 
#                 self.train()
#                 # self.scheduler.step()
#                 # self.scheduler_list.append(
#                 #     self.optim.param_groups[0]["lr"]
#                 # )
#             elif self.typ == "fine-tune": 
#                 self.fine_tune()
#                 self.test()
#             elif self.typ == "test": self.test()
            
            
#             torch.cuda.synchronize()
#             time_epoch_end.record()
#             print ("*********")
#             print ("Time Taken in Epoch {} is {:.1f}ms".format(self.epoch, time_epoch_start.elapsed_time(time_epoch_end)))
#             print ("*********")
            
#             self.pt.log_params(self.epoch, "epoch", self.config.model.NAME)
#             self.pt.log_model(self.model, self.config.model.NAME)
#             self.pt.log_aug(self.Augs.get_parameters_value("all"), self.config.model.NAME)
#             self.pt.save()
#         return 