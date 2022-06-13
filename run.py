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
from pytorch_lightning.plugins import DDPPlugin
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
import simplejson

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

tb_logger.save()

print(f"======= Evaluating {modelconfig['model_params']['name']} =======")
t =runner.test(ckpt_path="best", dataloaders=data.test_dataloader())[0]
print(t)

with open(tb_logger.log_dir+'/testresult.txt', 'w') as f:
    f.write('---------test scores---------\n')
    f.write('inception mean\n')
    f.write(str(t['inception mean']))
    f.write('\ninception stdev\n')
    f.write(str(t['inception stdv']))
    f.write('\nfrechet\n')
    f.write(str(t['frechet']))
    f.write('\n-------hyperparameters-------\n')
    f.write(simplejson.dumps(modelconfig, indent=4)+'\n')
    f.write(simplejson.dumps(dataconfig, indent=4)+'\n')

