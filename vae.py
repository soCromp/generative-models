# adapted from https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py

import torch
from torch import nn, optim
from torch.nn import functional as F
import pytorch_lightning as pl
from typing import List, Callable, Union, Any, TypeVar, Tuple
Tensor = TypeVar('torch.tensor')

class base_vae(pl.LightningModule): 
    def __init__(self, in_channels:int, latent_dims:int, hidden_dims:List=None, **kwargs)->None:
        super().__init__()
        size = 32
        h =4
        self.latent_dims = latent_dims
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # build encoder
        ic = in_channels
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(ic, out_channels=h_dim,
                        kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            ic=h_dim
        
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*h, latent_dims) # mean *4
        self.fc_var = nn.Linear(hidden_dims[-1]*h, latent_dims) # variance *4

        # build decoder
        self.decoder_input = nn.Linear(latent_dims, hidden_dims[-1]*h) # *4
        hidden_dims.reverse()

        modules = []
        for i in range(len(hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                        hidden_dims[i+1],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU())
            )
        
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[-1], size,
                                        # hidden_dims[-1],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        output_padding=1),
                    nn.BatchNorm2d(hidden_dims[-1]),
                    nn.LeakyReLU(),
                    # nn.Conv2d(hidden_dims[-1], out_channels=in_channels,
                    nn.Conv2d(size, out_channels=in_channels,
                                kernel_size=3, padding=1),
                    nn.Tanh())

    def to_latent(self, input: Tensor):
        mu, log_var = self.encode(input)
        return self.reparameterize(mu, log_var)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        return NotImplementedError

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dims)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        f = self.forward(x.cuda())[0]
        return f

    def configure_optimizers(self, params):
        # could get extended later
        optims = []
        scheds = []
        optimizer = optim.Adam(self.parameters(), #self.model.parameters(),
                               lr=params['LR'],
                               weight_decay=params['weight_decay'])
        optims.append(optimizer)
        scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                        gamma = params['scheduler_gamma'])
        scheds.append(scheduler)
        return optims, scheds

class conditional_vae(base_vae):
    def __init__(self, num_classes: int, img_size: int, in_channels: int, latent_dims: int, hidden_dims: List = None, **kwargs) -> None:
        self.img_size = img_size
        in_channels = in_channels + 1 # To account for the extra label channel
        super().__init__(in_channels, latent_dims, hidden_dims, **kwargs)
        self.embed_class = nn.Linear(num_classes, img_size * img_size)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        y = kwargs['labels'].float()
        embedded_class = self.embed_class(y)
        embedded_class = embedded_class.view(-1, self.img_size, self.img_size).unsqueeze(1)
        embedded_input = self.embed_data(input)
        x = torch.cat([embedded_input, embedded_class], dim=1)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        z = torch.cat([z,y], dim=1)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['kld_weight'] # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + kld_weight*kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        y = kwargs['labels'].float()
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        z = torch.cat([z,y], dim=1)
        samples = self.decode(z)
        return samples

class vanilla_vae(base_vae):
    def __init__(self, in_channels: int, latent_dims: int, hidden_dims: List = None, **kwargs) -> None:
        super().__init__(in_channels, latent_dims, hidden_dims, **kwargs)

    def loss_function(self, *args, batch=True, **kwargs) -> dict: #batch true is find avg loss. batch false is loss for each point alone
        # return super().loss_function(*args, **kwargs)
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['kld_weight'] # Account for the minibatch samples from the dataset
        if batch:
            recons_loss =F.mse_loss(recons, input)
        else:
            recons_loss =F.mse_loss(recons, input, reduction='none').mean(dim=[1,2,3])

        kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1)
        if batch:
            kld_loss = torch.mean(kld_loss, dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}


class beta_vae(base_vae):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self, in_channels: int, latent_dims: int, hidden_dims: List = None, beta: int = 4, gamma:float = 1000., 
        max_capacity: int = 25, Capacity_max_iter: int = 1e5, loss_type:str = 'B', **kwargs) -> None:
        
        super().__init__(in_channels, latent_dims, hidden_dims, **kwargs)

        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

    def loss_function(self, *args, **kwargs) -> dict:
        # return super().loss_function(*args, **kwargs)
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['kld_weight']  # Account for the minibatch samples from the dataset

        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}
