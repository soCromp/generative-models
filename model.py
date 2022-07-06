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
        try:
            self.S1func = S1funcs[self.params['S1func']]
            self.k = self.params['k']
            if self.params['S1func'] == 'inception':
                self.train_inception = InceptionScore(feature=64)
            elif self.params['S1func'] == 'frechet':
                self.train_fid = FrechetInceptionDistance(feature=64)#, reset_real_features=False)
            # self.val_inception = InceptionScore(reset_real_features=False)
            # self.val_fid = FrechetInceptionDistance(reset_real_features=False)
        except:
            self.S1func = None

        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input.cuda(), **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results,
                                            #   M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                              **self.params,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        self.log_dict({f"train_{key}": val.item() for key, val in train_loss.items()}, sync_dist=True, prog_bar=True)
        # self.train_fid.update(real_img.type(torch.uint8), real=True)
        # self.train_fid.update(results[0].type(torch.uint8), real=False)
        return train_loss['loss']

    def randomS1(self):
        return torch.rand((len(self.trainer.datamodule.train_dataset_all), ))

    def rarestS1(self):
        fit_loader = self.trainer.datamodule.train_dataloader()
        pred_loader = self.trainer.datamodule.train_all_dataloader()
        batch_size = self.trainer.datamodule.train_batch_size
        num_clusters = 5
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=2, batch_size=batch_size)
        
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
        ids, counts = np.unique(predictions[self.trainer.datamodule.indicesS0+self.trainer.datamodule.indicesS1==1], return_counts=True)
        
        v = torch.zeros((len(pred_loader.dataset) ,))
        if ids.shape[0] != num_clusters: #if not all clusters are present in predictions, non-present ones are the rarest
            missing = set(range(num_clusters)) - set(ids)
            for m in missing: #not efficient but there will only be one or two such m. revisit later
                v[predictions==m] = 1
        else: #all are present so select the one with fewest occurences
            rarest = np.argmin(counts) #index (=group id) of rarest group
            v[predictions==rarest] = 1

        self.analyzeClusters(predictions, num_clusters)
        return v

    def frechetS1(self):
        self.train_fid.reset() #runs after this train epoch is logged, so reset is ok
        fit_loader = self.trainer.datamodule.train_dataloader()
        pred_loader = self.trainer.datamodule.train_all_dataloader()
        batch_size = self.trainer.datamodule.train_batch_size
        num_clusters=5
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=2, batch_size=batch_size)

        for batch, i in fit_loader:
            latent = self.model.to_latent(batch.cuda()).cpu().detach().numpy()
            kmeans.partial_fit(latent)
            # self.train_fid.update(batch.cuda().type(torch.uint8), real=True)

        predbatches = [] #will be list of batch-size ndarray vectors
        for batch, i in pred_loader:
            latent = self.model.to_latent(batch.cuda()).cpu().detach().numpy()
            predbatches.append(kmeans.predict(latent))
        predictions = np.concatenate(predbatches) #cluster of each point

        frech = torch.zeros((num_clusters, ))
        in_train=(self.trainer.datamodule.indicesS0 + self.trainer.datamodule.indicesS1).type(torch.bool)
        for g in range(num_clusters): #get S0 and training S1 points in this cluster
            incl = torch.bitwise_and(in_train, torch.tensor(predictions)==g)
            if incl.sum() == 0: frech[g] = 1e10
            else:
                d = torch.utils.data.Subset(pred_loader.dataset, incl.nonzero(as_tuple=True)[0])
                dl = DataLoader(d, batch_size=int(incl.sum()))
                images, labels = next(iter(dl))
                results = self.forward(images.cuda(), labels = labels.cuda())
                self.train_fid.update(images.cuda().type(torch.uint8), real=True)
                self.train_fid.update(results[0].type(torch.uint8), real=False)
                if images.shape[0] == 1: #because fid only works if you have more than one
                    self.train_fid.update(images.cuda().type(torch.uint8), real=True)
                    self.train_fid.update(results[0].type(torch.uint8), real=False)
                frech[g] = self.train_fid.compute().item()
                self.train_fid.reset()

        v = torch.zeros((len(pred_loader.dataset) ,))
        # print(frech)
        worst = frech.argmax() #group with worst fid
        v[predictions==worst] = 1
        return v

    def inceptionS1(self):
        self.train_inception.reset() #just in-case
        fit_loader = self.trainer.datamodule.train_dataloader()
        pred_loader = self.trainer.datamodule.train_all_dataloader()
        batch_size = self.trainer.datamodule.train_batch_size
        num_clusters=5
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=2, batch_size=batch_size)

        # create clusters
        for batch, i in fit_loader:
            latent = self.model.to_latent(batch.cuda()).cpu().detach().numpy()
            kmeans.partial_fit(latent)

        # place each point in a cluster
        predbatches = [] #will be list of batch-size ndarray vectors
        for batch, i in pred_loader:
            latent = self.model.to_latent(batch.cuda()).cpu().detach().numpy()
            predbatches.append(kmeans.predict(latent))
        predictions = np.concatenate(predbatches) #cluster of each point

        incep = torch.zeros((num_clusters, ))
        in_train=(self.trainer.datamodule.indicesS0 + self.trainer.datamodule.indicesS1).type(torch.bool)
        for g in range(num_clusters): #get S0 and training S1 points in this cluster
            incl = torch.bitwise_and(in_train, torch.tensor(predictions)==g)
            if incl.sum() == 0: incep[g] = 1e10
            else:
                d = torch.utils.data.Subset(pred_loader.dataset, incl.nonzero(as_tuple=True)[0])
                dl = DataLoader(d, batch_size=int(incl.sum()))
                images, labels = next(iter(dl))
                results = self.forward(images.cuda(), labels = labels.cuda())
                self.train_inception.update(results[0].type(torch.uint8))
                incep[g] = self.train_inception.compute()[0].item() #[mean, stdv]
                self.train_inception.reset()

        v = torch.zeros((len(pred_loader.dataset) ,))
        print(incep)
        worst = incep.argmax() #group with worst fid
        v[predictions==worst] = 1
        return v

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
        predictionsS0 = predictions[self.trainer.datamodule.indicesS0+self.trainer.datamodule.indicesS1>0] 
        # print(predictionsS0.shape, np.concatenate(metrics).shape)
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

    def analyzeClusters(self, clusters, num_clusters): #pass in JUST the available training data
        for c in range(num_clusters):
            # find examples in cluster c
            # if there's none, create a black square
            # if there's some, arrange their images into a grid
            # figure out which labels and other metadata they are, how many are in S0 vs S1 per cluster
            return

    def on_train_epoch_end(self) -> None:
        #https://pytorch-lightning.readthedocs.io/en/stable/guides/data.html#accessing-dataloaders-within-lightningmodule
        # self.log('train_frechet', self.train_fid.compute().item())
        if self.S1func is not None:
            v = self.S1func()
            self.trainer.datamodule.v_update(v, self.k) 
            self.trainer.reset_train_dataloader(self)

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels, optimizer_idx=optimizer_idx) #[output, input, intermediates...]
        val_loss = self.model.loss_function(*results,
                                            kld_weight = 1.0, #real_img.shape[0]/ self.num_val_imgs, #for VAEs
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        # self.val_fid.update(real_img.type(torch.uint8), real=True)
        # self.val_fid.update(results[0].type(torch.uint8), real=False)

        
    def on_validation_epoch_end(self) -> None:
        samples = self.sample_images()[1] #saves images to file
        # samples = 255*self.model.sample(128, self.curr_device)
        # self.val_inception.update(samples.type(torch.uint8))

        # imean, istd = self.val_inception.compute()
        # self.log('val_inception_mean', imean.item())
        # self.log('val_inception_stdv', istd.item())

        # self.log('val_frechet', self.val_fid.compute().item())


    def on_test_start(self) -> None:
        self.test_inception = InceptionScore().cuda()
        self.test_fid = FrechetInceptionDistance().cuda()


    def test_step(self, batch, batch_idx) -> None:
        recons, samples, origs = self.sample_images(batch, tofile=False, num_samples=64)
        # print(recons.min(), recons.max(), origs.min(), origs.max())
        self.test_inception.update((255*samples).type(torch.uint8))
        self.test_fid.update(recons.type(torch.uint8), real=False)
        self.test_fid.update(origs.type(torch.uint8), real=True)


    def on_test_epoch_end(self) -> None:
        imean, istd = self.test_inception.compute()
        self.log('test_inception_mean', imean.item())
        self.log('test_inception_stdv', istd.item())
        self.log('test_frechet', self.test_fid.compute().item())


    def sample_images(self, batch=None, tofile=True, num_samples=144):
        # Get sample reconstruction image            
        if batch==None: batch = next(iter(self.trainer.datamodule.sample_dataloader()))
        test_input, test_label = batch
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        recons = self.model.generate(test_input, labels = test_label)
        # normalize to between 0 and 1
        recons = recons - recons.min()
        recons = recons / recons.max()
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
            return recons, samples, test_input
        except Warning:
            if tofile:
                pass
            return recons, None, test_input

    def configure_optimizers(self):
        return self.model.configure_optimizers(self.params)
