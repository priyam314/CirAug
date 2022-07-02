from torch.utils.data import DataLoader
from pprint import pprint
from protrack import ProTrack
from config import ConfigSeq
from fusion.execute import Train, FineTune, Test

import torch
import fusion as fuse
from fusion.utils.data import TrainLoader, ValLoader, TestLoader, Dataset
from fusion.arch.encoder import Encoder, MobileNet_V2, EfficientNet_B0
from fusion.arch.projector import TrainProjector, TestProjector
from fusion.utils.util import loader, ifExistLoad, Print, trainOf, fineTuneOf
from fusion.learning.optimizer import Optim
from fusion.learning.criterion import Criterion, BCEwithLogistLoss, CrossEntropyLoss, KLDiv, NTXent, BarlowTwins
from fusion.learning.scheduler import Scheduler
from pipeline import PipeLine

torch.backends.cudnn.benchmark = True
cfg = ConfigSeq()

## ProAug
from augs import Augs

######== Setting up Commands ==######
encoder = Encoder()
encoder.register("mobilenet-v2", MobileNet_V2)
encoder.register("efficientnet-b0", EfficientNet_B0)
criterion = Criterion(cfg.criteria)
criterion.register("bce", BCEwithLogistLoss)
criterion.register("cel", CrossEntropyLoss)
criterion.register("kldiv", KLDiv)
criterion.register("ntxent", NTXent)
criterion.register("barlow", BarlowTwins)

######== Dataset Setup Start ==######

Print("###### Data downloading and verification Init ######")

train_dataloader = TrainLoader(Dataset(cfg.data.name)('train'), cfg.batch).execute()
val_dataloader = ValLoader(Dataset(cfg.data.name)('train'), cfg.batch).execute()
test_dataloader = TestLoader(Dataset(cfg.data.name)('test'), cfg.batch).execute()

# fuse.utils.data.data_summary(train_dataloader, val_dataloader, test_dataloader)

######== Dataset Setup End ==######

cfg.set('data', 'dataset_size', len(train_dataloader.dataset))

######== ProAug Init Start==######

Print("###### ProAug Augs Init ######")

pprint(
    Augs.init(dataset_size=cfg.data.dataset_size,
              total_epochs=cfg.epoch.total,
              batch_size=cfg.batch.train_size,
              lamda=cfg.proaug.lamda))

######== ProAug Init End ==######

######== ProTrack Start ==######

Print("###### ProTrack Tracking Init ######")

pt = ProTrack()
# init the ProTrack and load the ProLogs
pt.init().load()
# log the model name
pt.log_model_name(cfg.model.name)
# log the config params
pt.log_config_params(cfg, cfg.model.name)
# log the param "OMEGA" which is size of augs pool
pt.log_params(Augs.util.o, "OMEGA", cfg.model.name)
# load the current parameters values from protrack to Augs
Augs.load_params(pt.data(cfg.model.name, 'aug_params').value)
# save the protrack.json
pt.save()

######== ProTrack End ==######

######== Model Start ==######

Print("###### Model Building and Loading ######")

trainModel = ifExistLoad(encoder.execute(cfg.model, TrainProjector(cfg.hlayer)), pt, cfg.model, "train")
valModel = ifExistLoad(encoder.execute(cfg.model, TrainProjector(cfg.hlayer)), pt, trainOf(cfg.model), "fine-tune")
testModel = ifExistLoad(encoder.execute(cfg.model, TestProjector(cfg.hlayer)), pt, fineTuneOf(cfg.model), "test")

if cfg.utils.cuda:
    trainModel = trainModel.cuda()
    valModel = valModel.cuda()
    testModel = testModel.cuda()

######== Model End ==######

######== Executioner Start ==######

Print("###### Execution Initiated ######")

widgets = {
    'Augs': Augs,
    'protrack': pt,
    'optim': Optim(cfg.optimizer.name),
    'criterion': Criterion(cfg),
    'scheduler': Scheduler('yo')
}
pipe = PipeLine(cfg.exec.typ)
pipe.add(
    Train(model=trainModel,
          dataloader=train_dataloader,
          data=cfg.data,
          **widgets))
pipe.add(
    FineTune(model=valModel,
             projector=TestProjector(cfg.hlayer),
             dataloader=val_dataloader,
             data=cfg.data,
             **widgets))
pipe.add(
    Test(model=testModel, dataloader=test_dataloader, data=cfg.data,
         **widgets))
pipe.set_on_start()

# executioner = Executioner(
#     model = model,
#     projector = Projector(cfg.hlayer.test),
#     train_dataloader = train_dataloader,
#     val_dataloader = val_dataloader,
#     test_dataloader = test_dataloader,
#     Augs = Augs,
#     optim =  Optim(cfg.optim.name)(model, cfg),
#     criterion = Criterion(
#         cfg.loss.CRITERIA,
#         cfg.loss.TEMPERATURE,
#         cfg.loss.LAMBDA_COEFF,
#         BATCH_SIZE
#     ),
#     scheduler = fuse.learning.scheduler.Scheduler(cfg.optim.LR_SCHEDULER)(
#         optim,
#         T_0 = cfg.epoch.WARMUP*exec_steps(),

#     ),
#     protrack = pt,
#     config = cfg,
#     data_percent = cfg.data
# )

# executioner.run(cfg.execution.TYPE, cfg.epoch.START, cfg.epoch.TOTAL)

######== Executioner End ==######

# def exec_steps(self):
#     if cfg.execution.TYPE == "train": return len(train_dataloader.dataset)//cfg.batch.TRAIN_SIZE+1
#     elif cfg.execution.TYPE == "fine-tune": return len(val_dataloader.dataset)//cfg.batch.VAL_SIZE+1
#     elif cfg.execution.TYPE == "test": return len(test_dataloader.dataset)//cfg.batch.TEST_SIZE+1
