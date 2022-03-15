import os
import argparse
import vae
from model import Model
from pytorch_lightning import Trainer
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
from dataloader import Dataset
from pytorch_lightning.plugins import DDPPlugin
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
import torch

parser = argparse.ArgumentParser(description='Generic experiment driver')
parser.add_argument('--modelconfig', '-m', help='yaml file of model hyperparameters', default='conf/model/vae.yaml')
parser.add_argument('--dataconfig', '-d', help='yaml file of dataset settings', default='conf/data/mnist.yaml')
args = parser.parse_args()

# interpret given hyperparameters
with open(args.modelconfig, 'r') as file:
    try:
        modelconfig = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

with open(args.dataconfig, 'r') as file:
    try:
        dataconfig = yaml.safe_load(file)
        modelconfig['model_params']['in_channels'] = dataconfig['in_channels']
    except yaml.YAMLError as exc:
        print(exc)

# set up model
name=modelconfig['model_params']['name']
if name == 'vae':
    print('will run vanilla VAE experiment')
    model = vae.vanilla_vae(**modelconfig['model_params'])
elif name=='betavae':
    print('will run beta vae experiment')
    model = vae.beta_vae(**modelconfig['model_params'])
else:
    raise NotImplementedError

tb_logger =  TensorBoardLogger(save_dir=modelconfig['logging_params']['save_dir'],
                               name=modelconfig['model_params']['name'],)

# For reproducibility
# seed_everything(modelconfig['exp_params']['manual_seed'], True)

experiment = Model(model, modelconfig['exp_params'])
data = Dataset(**dataconfig, pin_memory=len(modelconfig['trainer_params']['gpus']) != 0)
data.setup()

runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2, 
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                     monitor= "val_loss",
                                     save_last= True),
                 ],
                 strategy=DDPPlugin(find_unused_parameters=False),
                 **modelconfig['trainer_params'])


Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {modelconfig['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)


print(f"======= Calculating metrics for trained {modelconfig['model_params']['name']} =======")
inception = InceptionScore()
fid = FrechetInceptionDistance()
model.cuda()
recons, samples, origs = self.sample_images(tofile=False, num_samples=128, orig=True)
samples.cuda()
samples = samples*255
inception.update(samples.type(torch.uint8))
a,b = inception.compute()
print(a.item(), b.item())
fid.update(recons.type(torch.uint8), real=False)
fid.update(origs.type(torch.uint8), real=True)
print(fid.compute())