# based on https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py

from turtle import forward
import torch
from torch import nn, optim
from torch.nn import functional as F
import pytorch_lightning as pl
Tensor = TypeVar('torch.tensor')

class Generator(nn.Module):
    def __init__(self, latent_dims:int, out_dim:int, out_channels:int, **kwargs) -> None:
        super().__init__() #output image is out_channels x out_dim x out_dim
        self.latent_dims = latent_dims
        self.out_dim = out_dim
        self.out_channels = out_channels

        def block(in_feat, out_feat, normalize=True):
                layers = [nn.Linear(in_feat, out_feat)]
                if normalize:
                    layers.append(nn.BatchNorm1d(out_feat, 0.8))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return layers

        self.model = nn.Sequential(
            *block(self.params(latent_dims), 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(out_channels*out_dim*out_dim)),
            nn.Tanh()
        )


    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        img = self.model(input)
        img = img.view(img.size(0), self.out_channels, self.out_dim, self.out_dim)
        return img


    def configure_optimizers(self):
        optims = []
        opts = optim.Adam(self.params, lr=opt.lr, betas=(self.params['b1'], self.params['b2'])
        optims.append(opts)
        return optims

    
class Discriminator(nn.Module):
    def __init__(self, in_dim:int, in_channels:int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.in_channels = in_channels

        self.model = nn.Sequential(
            nn.Linear(int(in_dim*in_dim*in_channels), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        flat = input.view(input.size(0), self.in_channels, self.in_dim, self.in_dim)
        preds = self.model(flat)
        return preds    

    def configure_optimizers(self):
        optims = []
        opts = optim.Adam(self.params, self.params['LR'], betas=(self.params['b1'], self.params['b2']))
        optims.append(opts)
        return opts


class base_gan(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.G = Generator()
        self.D = Discriminator()

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        z = Tensor(np.random.normal(0, 1, (imgs.shape[0], self.params(latent_dims)))) #sample rand noise
        fakes = G(z) 
        # mix fakes with real data
        # discriminator predicts fake v real
        return NotImplementedError #return fakes, preds, truth labels

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        preds = args[1] #index is return order from forward method
        truth = args[2]
        return {'loss': torch.nn.BCELoss(preds, truth)}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        return NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.G(x)

    def configure_optimizers(self):
        return NotImplementedError


class vanilla_gan(base_gan):
    def __init__(self) -> None:
        super().__init__()
