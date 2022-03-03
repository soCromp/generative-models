# adapted from https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

class vae(pl.LightningModule): # will extend some abstract model class
    def __init__(self, in_channels:int, latent_dims:int, hidden_dims:List=None, **kwargs):
        super().__init__()
        
        self.latent_dims = latent_dims
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # build encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                        kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels=h_dim
        
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dims) # mean
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dims) # variance

        # build decoder
