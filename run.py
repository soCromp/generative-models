import os
import argparse
import vae
from experiment import Experiment
from torchvision import datasets, transforms
from pytorch_lightning import Trainer
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
from torch.utils.data import DataLoader
from dataloader import Dataset
from pytorch_lightning.plugins import DDPPlugin

parser = argparse.ArgumentParser(description='Generic experiment driver')
parser.add_argument('--model',  '-m', help =  'which model to use. Options: vae', default='vae')
parser.add_argument('--configfile', '-cf', help='yaml file of hyperparameters', default='config/vae.yaml')
args = parser.parse_args()

# interpret given hyperparameters
with open(args.configfile, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# set up model
if args.model == 'vae':
    print('will run vanilla VAE experiment')
    model = vae.vae(**config['model_params'])
else:
    raise NotImplementedError

tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['model_params']['name'],)

# For reproducibility
# seed_everything(config['exp_params']['manual_seed'], True)

experiment = Experiment(model,
                          config['exp_params'])

data = Dataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
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
                 **config['trainer_params'])


Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)
