import math
import os
import torch
from torch import optim
from typing import List, Callable, Union, Any, TypeVar, Tuple
from yaml import load
Tensor = TypeVar('torch.tensor')
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import tqdm


class Model(pl.LightningModule):

    def __init__(self,
                 model,
                 params: dict,) -> None:
        super(Model, self).__init__()
        self.model = model
        self.params = params
        S1funcs = {'random': self.randomS1, 'rarest': self.rarestS1, 'frechet': self.frechetS1, 
                    'inception': self.inceptionS1, 'loss': self.lossS1, 'none': None}
        self.S1func = S1funcs[self.params['S1func']]
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

    def rarestS1(self):
        fit_loader = self.trainer.datamodule.train_dataloader()
        pred_loader = self.trainer.datamodule.train_all_dataloader()
        batch_size = self.trainer.datamodule.train_batch_size
        kmeans = MiniBatchKMeans(n_clusters=5, random_state=2, batch_size=batch_size)
        
        metrics = []
        for batch, i in fit_loader:
            latent = self.model.to_latent(batch.cuda()).cpu().detach().numpy()
            kmeans.partial_fit(latent)

        predbatches = [] #will be list of batch-size ndarray vectors
        for batch, i in pred_loader:
            latent = self.model.to_latent(batch.cuda()).cpu().detach().numpy()
            predbatches.append(kmeans.predict(latent))

        # concatenate preds 
        predictions = np.concatenate(predbatches) #cluster of each point
        _, counts = np.unique(predictions[self.trainer.datamodule.indicesS0==1], return_counts=True)
        rarest = np.argmin(counts) #index (=group id) of rarest group
        
        v = torch.zeros((len(pred_loader.dataset) ,))
        v[predictions==rarest] = 1
        return v

    def frechetS1(self):
        loader = self.trainer.datamodule.train_all_dataloader()
        return NotImplementedError

    def inceptionS1(self):
        return NotImplementedError

    def lossS1(self):
        fit_loader = self.trainer.datamodule.train_dataloader()
        pred_loader = self.trainer.datamodule.train_all_dataloader()
        batch_size = self.trainer.datamodule.train_batch_size
        numclusters=5
        kmeans = MiniBatchKMeans(n_clusters=numclusters, random_state=2, batch_size=batch_size)
        
        metrics = []
        for batch, i in fit_loader:
            latent = self.model.to_latent(batch.cuda()).cpu().detach().numpy()
            m = self.model.loss_function(*self.model.forward(batch.cuda()), kld_weight = 1.0, batch=False)['loss']
            metrics.append(m.cpu().detach().numpy())
            kmeans.partial_fit(latent)

        predbatches = [] #will be list of batch-size ndarray vectors
        for batch, i in pred_loader:
            latent = self.model.to_latent(batch.cuda()).cpu().detach().numpy()
            predbatches.append(kmeans.predict(latent))

        # concatenate preds 
        predictions = np.concatenate(predbatches) #cluster of each point
        #get just points in S0 or S1 trained on in this epoch:
        predictionsS0 = predictions[self.trainer.datamodule.indicesS0+self.trainer.datamodule.indicesS1==1] 
        df = np.stack([predictionsS0, np.concatenate(metrics)]).T #col0 is cluster and col1 is loss
        #ensure all clusters have at least one point
        ids = set(np.unique(df[:, 0]))
        allids = set(range(numclusters))
        absent = list(allids - ids)
        if len(absent)>0:
            loss = [1e10]*len(absent)
            a = np.stack([absent, loss]).T
            df = np.concatenate([df, a])

        dfs = df[df[:, 0].argsort()] #sorting
        gb = np.split(dfs[:, 1], np.unique(dfs[:, 0], return_index=True)[1][1:]) #"group by": ith elem is list of pts in ith group
        avgs=np.stack([g.mean() for g in gb]) #average loss of S0 points in the group
        worst = np.argmax(avgs) #group with highest loss
        
        v = torch.zeros((len(pred_loader.dataset) ,))
        v[predictions==worst] = 1
        return v

    def on_train_epoch_end(self) -> None:
        #https://pytorch-lightning.readthedocs.io/en/stable/guides/data.html#accessing-dataloaders-within-lightningmodule
        if self.S1func is not None:
            v = self.S1func()
            self.trainer.datamodule.v_update(v, 20) # = self.trainer.datamodule.v_update(randy, 100)
            self.trainer.reset_train_dataloader(self)

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
