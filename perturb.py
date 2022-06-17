# %%
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
from torchvision.utils import make_grid
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
import torch
from math import e
from tqdm import tqdm


# %%
# interpret given hyperparameters
with open('./conf/model/betavae.yaml', 'r') as file:
    modelconfig = yaml.safe_load(file)

with open('./conf/data/celeba.yaml', 'r') as file:
    dataconfig = yaml.safe_load(file)
    modelconfig['model_params']['in_channels'] = dataconfig['in_channels']

# %%
vae = vae.beta_vae(**modelconfig['model_params'])
experiment = Model(vae, modelconfig['exp_params'])
checkpoint = torch.load('logs/betavae/version_3/checkpoints/last.ckpt')
experiment.load_state_dict(checkpoint['state_dict'])

# %%
data = Dataset(**dataconfig, pin_memory=len(modelconfig['trainer_params']['gpus']) != 0)
data.setup()
test = data.test_dataloader()

# %%
ex, label = next(iter(test)) #get random batch
mu, var = vae.encode(ex)

# %%
decode, _, mu, log_var = vae.forward(ex)

# %%
im=decode

# %% [markdown]
# ## Try each dimension seperately

# %%
dims = var.shape[1]
ims=[]
print('Perturbing along dimensions')
for d in tqdm(range(dims)):
    mum = mu
    mum[:,d]=mum[:,d]-e**(log_var[:,d])
    zm =vae.reparameterize(mum, log_var)
    imm=vae.decode(zm)
    
    mup = mu
    mup[:,d]=mup[:,d]+e**(log_var[:,d])
    zp =vae.reparameterize(mup, log_var)
    imp=vae.decode(zp)

    ims = ims + [imm, im, imp]

# %%
print('Saving images')
for ex in tqdm(range(im.shape[0])):
    res=to_pil_image(make_grid( [T.Resize(256)(i[ex]) for i in ims], nrow=3 ))
    res.save(f'exs/betavae{ex}.png')

# %%
res

# %%



