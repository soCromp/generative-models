## performs evaluation using validation or test set
from ast import Expression
from ensurepip import version
import os
import argparse
import vae
import diffusion
from model import Model
from pytorch_lightning import Trainer
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
from dataloader import Dataset
from pytorch_lightning.strategies.ddp import DDPStrategy

parser = argparse.ArgumentParser(description='Generic experiment driver')
# parser.add_argument('--test', '-t', help='True if wish to use test set, false for validation set', default=False)
parser.add_argument('--path', '-p', help='Path to log directory of model you want to evaluate (i.e. .../version_X')
parser.add_argument('--modelconfig', '-m', help='yaml file of model hyperparameters if they are not included in logdir', default=None) #optional
parser.add_argument('--dataconfig', '-d', help='yaml file of dataset settings if they are not included in logdir', default=None) #optional
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
        modelconfig['model_params']['patch_size'] = dataconfig['in_channels']
        modelconfig['model_params']['num_classes'] = dataconfig['num_classes']
    except yaml.YAMLError as exc:
        print(exc)
data = Dataset(**dataconfig, pin_memory=True)#use GPU
data.setup()

# find desired (best) checkpoint
ckpt = os.path.join(args.path, 'checkpoints', sorted(os.listdir(os.path.join(args.path, 'checkpoints')))[-2])
print(ckpt)

# set up model
name=modelconfig['model_params']['name']
if name == 'vae':
    print('will run vanilla VAE experiment')
    model = vae.vanilla_vae(**modelconfig['model_params']) 
elif name=='betavae':
    print('will run beta vae experiment')
    model = vae.beta_vae(**modelconfig['model_params'])
elif name=='diffusion':
    print('will run diffusion model experiment')
    model = diffusion.base_diffusion(**modelconfig['model_params'])
else:
    raise NotImplementedError



experiment = Model(model.cuda(), modelconfig['exp_params']).cuda()
experiment.load_from_checkpoint(ckpt, model=model.cuda(), params=modelconfig['exp_params'])
# experiment = Model(model.cuda(), modelconfig['exp_params']).cuda()


version = args.path.split('/')[-1]
logdir = '/'.join(args.path.split('/')[:-1])
print(logdir, version)
tb_logger =  TensorBoardLogger(save_dir=logdir, name=None, version=version)

trainer = Trainer(gpus=[0], logger=tb_logger, log_every_n_steps=1,)#https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing.html#checkpoint-loading

trainer.test(model=experiment, dataloaders=data.val_dataloader())

tb_logger.save()
