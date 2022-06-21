import os
import math
import torch
from torch import optim
from typing import List, Callable, Union, Any, TypeVar, Tuple
Tensor = TypeVar('torch.tensor')
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance


class Model(pl.LightningModule):

    def __init__(self,
                 model,
                 params: dict,
                 S1func='random') -> None:
        super(Model, self).__init__()

        self.model = model
        self.params = params
        S1funcs = {'random': self.randomS1, 'cluster': self.clusterS1, 'frechet': self.frechetS1, 
                    'inception': self.inceptionS1, 'loss': self.lossS1}
        self.S1func = S1funcs[S1func]
        self.inception = InceptionScore()
        self.fid = FrechetInceptionDistance()
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        # print('forward')
        return self.model(input.cuda(), **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        # print('training step')
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results,
                                            #   M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                              **self.params,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        self.log_dict({f"train_{key}": val.item() for key, val in train_loss.items()}, sync_dist=True, prog_bar=True)
        return train_loss['loss']

    def randomS1(self):
        return torch.rand((len(self.trainer.datamodule.train_dataset_all), ))

    def clusterS1(self):
        return NotImplementedError
        
    def frechetS1(self):
        return NotImplementedError

    def inceptionS1(self):
        return NotImplementedError

    def lossS1(self):
        return NotImplementedError

    def on_train_epoch_end(self) -> None:
        #https://pytorch-lightning.readthedocs.io/en/stable/guides/data.html#accessing-dataloaders-within-lightningmodule
        v = self.S1func()
        self.trainer.datamodule.v_update(v, len(self.trainer.train_dataloader.dataset)//2) # = self.trainer.datamodule.v_update(randy, 100)
        self.trainer.reset_train_dataloader(self)
        print(len(self.trainer.train_dataloader.dataset))
        # self.trainer.train_dataloaders[0] 

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        # print('validation step')
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels, optimizer_idx=optimizer_idx)
        val_loss = self.model.loss_function(*results,
                                            kld_weight = 1.0, #real_img.shape[0]/ self.num_val_imgs, #for VAEs
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

        
    def on_validation_epoch_end(self) -> None:
        # print('on_validation_epoch_end')
        self.sample_images()
        recons, samples, origs = self.sample_images(tofile=False, num_samples=128, orig=True)
        samples.cuda()
        samples = samples*255
        self.inception.update(samples.type(torch.uint8))
        imean, istd = self.inception.compute()
        self.log('val_inception_mean', imean.item())
        self.log('val_inception_stdv', istd.item())

        self.fid.update(recons.type(torch.uint8), real=False)
        self.fid.update(origs.type(torch.uint8), real=True)
        self.log('val_frechet', self.fid.compute().item())


    def test_step(self, batch, batch_idx) -> None:
        # print('test_step')
        recons, samples, origs = self.sample_images(tofile=False, num_samples=64, orig=True)
        samples = samples*255
        self.inception.update(samples.type(torch.uint8))
        self.fid.update(recons.type(torch.uint8), real=False)
        self.fid.update(origs.type(torch.uint8), real=True)


    def on_test_epoch_end(self) -> None:
        # print('on_test_epoch_end')
        imean, istd = self.inception.compute()
        frechet = self.fid.compute().item()
        self.log('test_inception_mean', imean.item())
        self.log('test_inception_stdv', istd.item())
        self.log('test_frechet', self.fid.compute().item())


    def sample_images(self, tofile=True, num_samples=144, orig=False):
        # print('sample images')
        # Get sample reconstruction image            
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        #       test_input, test_label = batch
        recons = self.model.generate(test_input, labels = test_label)
        if tofile:
            vutils.save_image(recons.data,
                            os.path.join(self.logger.log_dir , 
                                        "Reconstructions", 
                                        f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                            normalize=True,
                            nrow=12)

        try:
            samples = self.model.sample(num_samples,
                                        self.curr_device,
                                        labels = test_label)
            if tofile:
                vutils.save_image(samples.cpu().data,
                                os.path.join(self.logger.log_dir , 
                                            "Samples",      
                                            f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                                normalize=True,
                                nrow=12)
            elif orig: return recons, samples, test_input
            else: return recons, samples
        except Warning:
            if tofile:
                pass
            elif orig:
                return recons, None, test_input
            else:
                return recons

    def configure_optimizers(self):
        return self.model.configure_optimizers(self.params)
