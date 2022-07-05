import torch
from augs import Augs
from torch.utils.data import DataLoader
from pprint import pprint
from protrack import ProTrack
from config import ConfigSeq
from pipeline import Pipeline, Training, FineTuning
from fusion.execute import TrainExecution, FineTuneExecution, TestExecution
from fusion.utils.data import TrainLoader, ValLoader, TestLoader, Dataset, STL10, SVHN, FOOD101
from fusion.utils.util import loader, ifExistLoad, Print, trainOf, fineTuneOf
from fusion.arch.encoder import Encoder, MobileNet_V2, EfficientNet_B0
from fusion.arch.projector import TrainProjector, TestProjector
from fusion.arch.model_builder import TrainModel, Fine_TuneModel, TestModel, ModelBuilder
from fusion.learning.optimizer import Optim, Adam, AdamW, AdaBelief, SGD
from fusion.learning.criterion import Criterion, BCEwithLogistLoss, CrossEntropyLoss, KLDiv, NTXent, BarlowTwins
from fusion.learning.scheduler import Scheduler

#######== Configuration ==########
torch.backends.cudnn.benchmark = True
cfg = ConfigSeq()
pipe = Pipeline(cfg.exec)
pipe.register('train', Training)
pipe.register('fine-tune', FineTuning)

######== Setting up Commands ==######
dataset = Dataset()
dataset.register("svhn", SVHN)
dataset.register("stl10", STL10)
dataset.register("food101", FOOD101)

encoder = Encoder()
encoder.register("mobilenet-v2", MobileNet_V2)
encoder.register("efficientnet-b0", EfficientNet_B0)

criterion = Criterion(cfg.criteria)
criterion.register("bce", BCEwithLogistLoss)
criterion.register("cel", CrossEntropyLoss)
criterion.register("kldiv", KLDiv)
criterion.register("ntxent", NTXent)
criterion.register("barlow", BarlowTwins)

optimizer = Optim()
optimizer.register("adam", Adam)
optimizer.register("adamw", AdamW)
optimizer.register("sgd", SGD)
optimizer.register("adabelief", AdaBelief)



######== Dataset Setup Start ==######
Print("###### Data downloading and verification Init ######")

train_dataloader = TrainLoader(dataset.execute(cfg.data, 'train'), cfg.batch).execute()
val_dataloader = ValLoader(dataset.execute(cfg.data, 'train'), cfg.batch).execute()
test_dataloader = TestLoader(dataset.execute(cfg.data, 'test'), cfg.batch).execute()

######== Dataset Setup End ==######

cfg.set('data', 'dataset_size', len(train_dataloader.data.dataset))

######== ProAug Init Start==######
Print("###### ProAug Augs Init ######")

Augs.init(dataset_size = cfg.data.dataset_size,
          total_epochs = cfg.epoch.total,
          batch_size = cfg.batch.train_size,
          lamda = cfg.proaug.lamda)

######== ProAug Init End ==######

######== ProTrack Start ==######
Print("###### ProTrack Tracking Init ######")

# created the object of ProTrack
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

modelBuilder = ModelBuilder(encoder = encoder, model = cfg.model, pt = pt, hlayer = cfg.hlayer)

modelBuilder.register("trainModel", TrainModel)
modelBuilder.register("fine-tuneModel", Fine_TuneModel)
modelBuilder.register("testModel", TestModel)

trainModel = modelBuilder.execute("trainModel",TrainProjector, "train")
fine_tuneModel = modelBuilder.execute("fine-tuneModel",TrainProjector, "fine-tune")
testModel = modelBuilder.execute("testModel",TestProjector, "test")

# if cfg.utils.cuda:
#     trainModel = trainModel.model.cuda()
#     fine_tuneModel = fine_tuneModel.model.cuda()
#     testModel = testModel.model.cuda()

######== Model End ==######
pipe.addLoaders(train_dataloader, val_dataloader, test_dataloader)
pipe.addModels(trainModel, fine_tuneModel, testModel)
pipe.addCriterion(criterion)
pipe.addOptimizer(optimizer)
pipe.addExecutions(TrainExecution, FineTuneExecution, TestExecution)
pipe.addAugs(Augs) 
pipe.addProtrack(pt)
######== Executioner Start ==######
Print("###### Execution Initiated ######")

pipe.execute(cfg.exec)


# widgets = {
#     'Augs': Augs,
#     'protrack': pt,
#     'optim': optimizer,
#     'criterion': criterion,
#     'scheduler': Scheduler('yo')
# }


# pipe.add(
#     Train(model=trainModel,
#           dataloader=train_dataloader,
#           data=cfg.data,
#           **widgets))
# pipe.add(
#     FineTune(model=valModel,
#              projector=TestProjector(cfg.hlayer),
#              dataloader=val_dataloader,
#              data=cfg.data,
#              **widgets))
# pipe.add(
#     Test(model=testModel, dataloader=test_dataloader, data=cfg.data,
#          **widgets))
# pipe.set_on_start()

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
