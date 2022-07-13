import os
import argparse
import vae
import diffusion
from model import Model
from pytorch_lightning import Trainer
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
from dataloader import Dataset
from pytorch_lightning.strategies.ddp import DDPStrategy
import simplejson

import time
ts = time.time()

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
        modelconfig['model_params']['patch_size'] = dataconfig['in_channels']
        modelconfig['model_params']['num_classes'] = dataconfig['num_classes']
    except yaml.YAMLError as exc:
        print(exc)

# setup logging
if not os.path.isdir(os.path.join(modelconfig['logging_params']['save_dir'], modelconfig['logging_params']['name'],
                                dataconfig['data_name'], modelconfig['model_params']['name'])):
    os.makedirs(os.path.join(modelconfig['logging_params']['save_dir'], modelconfig['logging_params']['name'],
                                    dataconfig['data_name'], modelconfig['model_params']['name']))
tb_logger =  TensorBoardLogger(save_dir=os.path.join(modelconfig['logging_params']['save_dir'], modelconfig['logging_params']['name'],
                                dataconfig['data_name']),
                               name=modelconfig['model_params']['name'],)
print('logging to', tb_logger.log_dir)

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

# For reproducibility
# seed_everything(modelconfig['exp_params']['manual_seed'], True)

experiment = Model(model, modelconfig['exp_params'])
data = Dataset(**dataconfig, pin_memory=len(modelconfig['trainer_params']['gpus']) != 0)
data.setup()
print(len(data.train_dataset_all))

Path(tb_logger.log_dir).mkdir(parents=True, exist_ok=True)
with open(tb_logger.log_dir+'/hparams.txt', 'w') as f:
    f.write('-------hyperparameters-------\n')
    f.write(simplejson.dumps(modelconfig, indent=4)+'\n')
    f.write(simplejson.dumps(dataconfig, indent=4)+'\n')

runner = Trainer(logger=tb_logger,
                log_every_n_steps=1,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=1, #save the one best model 
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                     monitor= "val_loss",
                                     save_last= True),
                 ],
                 strategy=DDPStrategy(find_unused_parameters=False),
                 **modelconfig['trainer_params'])

Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {modelconfig['model_params']['name']} =======")
# try:
runner.fit(experiment, datamodule=data)
tb_logger.save()
error = ''
sbj = f'Finished training {tb_logger.log_dir}!'
# except Exception as e:
#     sbj = f'ERROR in training {tb_logger.log_dir}'
#     error = str(e) + '\n'

te = time.time()
msg = f'Location: {os.environ["SSH_CLIENT"]}\nStart time: {time.ctime(ts)}\nEnd time: {time.ctime(te)}\nDuration: {(te-ts)//60} minutes or {(te-ts)//3600} hours\n\n'+\
    f'{error}Model metadata: {simplejson.dumps(modelconfig, indent=4)}\nData metadata: {simplejson.dumps(dataconfig, indent=4)}'

try:    
    import sys
    sys.path.insert(0, f'/home/{os.environ["USER"]}') #because notify.py is in ~
    from notify import email
    email(sbj, msg)
except: print('notify.py not found in home directory or current directory')
