# based on https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py

from turtle import forward
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
Tensor = TypeVar('torch.tensor')

class Generator(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        return NotImplementedError

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        return NotImplementedError

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        return NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return NotImplementedError

    def configure_optimizers(self):
        optims = []
        opts = optim.Adam(self.params, lr=opt.lr, betas=(self.params['b1'], self.params['b2'])
        optims.append(opts)
        return optims

    
class Discriminator(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        return NotImplementedError

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        return NotImplementedError

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
        return NotImplementedError

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        return NotImplementedError
        # adversarial_loss = torch.nn.BCELoss()

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        return NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return NotImplementedError

    def configure_optimizers(self):
        return NotImplementedError


class vanilla_gan(base_gan):
    def __init__(self) -> None:
        super().__init__()
