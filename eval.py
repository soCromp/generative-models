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
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path
from dataloader import Dataset
from pytorch_lightning.strategies.ddp import DDPStrategy
import simplejson
import time
ts = time.time()

parser = argparse.ArgumentParser(description='Generic experiment driver')
# parser.add_argument('--test', '-t', help='True if wish to use test set, false for validation set', default=False)
parser.add_argument('--path', '-p', help='Path to log directory of model you want to evaluate (i.e. .../version_X')
parser.add_argument('--epoch', '-e', help='Epoch of checkpoint you want (default is -1, ie last epoch)', default=-1, type=int)
args = parser.parse_args()

# interpret given hyperparameters
with open(os.path.join(args.path, 'modelconfig.json'), 'r') as file:
    try:
        modelconfig = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

with open(os.path.join(args.path, 'dataconfig.json'), 'r') as file:
    try:
        dataconfig = yaml.safe_load(file)
        modelconfig['model_params']['in_channels'] = dataconfig['in_channels']
        modelconfig['model_params']['patch_size'] = dataconfig['in_channels']
        modelconfig['model_params']['num_classes'] = dataconfig['num_classes']
    except yaml.YAMLError as exc:
        print(exc)
data = Dataset(**dataconfig, pin_memory=True)#use GPU
data.setup()

if args.epoch == -1:
    print('Will use checkpoint last.ckpt')
    ckpt = os.path.join(args.path, 'checkpoints/last.ckpt')
else:
    d = os.listdir(os.path.join(args.path, 'checkpoints')) # epoch=0....ckpt, epoch=1....ckpt, ...., last.ckpt
    checkpoint_name = [ckpt for ckpt in d if ckpt.startswith(f'epoch={args.epoch}-')][0]
    print('Will use checkpoint', checkpoint_name)
    ckpt = os.path.join(args.path, f'checkpoints/{checkpoint_name}')

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
id = sorted(os.listdir(os.path.join(args.path, 'wandb/latest-run')))[2][4:-6] #pulls id from file eg run-1cxm3z9c.wandb
logger =  WandbLogger(id=id, resume='must')

trainer = Trainer(gpus=[0], logger=logger, log_every_n_steps=1,)#https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing.html#checkpoint-loading

trainer.test(model=experiment, dataloaders=data.val_dataloader())

logger.save()

te = time.time()
msg = f'Location: {os.environ["SSH_CLIENT"]}\nStart time: {time.ctime(ts)}\nEnd time: {time.ctime(te)}\nDuration: {(te-ts)//60} minutes or {(te-ts)//3600} hours\n\n'+\
    f'Model metadata: {simplejson.dumps(modelconfig, indent=4)}\nData metadata: {simplejson.dumps(dataconfig, indent=4)}'
sbj = f'Finished evaluating {logger.save_dir}!'

try:    
    import sys
    sys.path.insert(0, f'/home/{os.environ["USER"]}') #because notify.py is in ~
    from notify import email
    email(sbj, msg)
except: print('notify.py not found in home directory or current directory')
