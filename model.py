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
import torchvision.transforms.functional as F
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from random import sample, choices
import math
import scipy.stats as ss
import statistics

class Model(pl.LightningModule):
    def __init__(self,
                 model,
                 params: dict,) -> None:
        super(Model, self).__init__()
        self.model = model
        self.params = params
        S1funcs = {'random': self.randomS1, 'rarest': self.rarestS1, 'loss': self.lossS1, 'none': None}
        try: #clustering setup
            self.S1func = S1funcs[self.params['S1func']]
            self.k = self.params['k']
            self.clusters = [] # predicted cluster for each point in each epoch
            self.availS1 = [] # 1hot S1 points that are available in a given epoch (starts with 1st epoch)
            self.num_clusters = 2
        except:
            self.S1func = None

        try: 
            self.active = self.params['active'] #whether to use active learning
            self.edmu = torch.zeros((self.model.latent_dims, self.params['max_epochs']))
            self.edvr = torch.zeros(self.edmu.shape)
            self.help = []
        except: self.active = False
        
        self.val_l1difference = torch.nn.L1Loss()
        # self.val_inception = InceptionScore().cuda()
        # self.val_fid = FrechetInceptionDistance().cuda()

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
        # kmeans = MiniBatchKMeans(n_clusters=self.num_clusters, random_state=1, batch_size=batch_size, reassignment_ratio=0.9,
        #                         max_no_improvement=None, )
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=1, )

        X = None
        for batch, i in fit_loader:
            latent = self.model.to_latent(batch.cuda()) #.cpu().detach().numpy()
            # kmeans.partial_fit(latent)
            # X.append(latent)
            if type(X) == type(None):
                X=latent.cpu().detach().numpy()
            else:
                X = np.concatenate([X, latent.cpu().detach().numpy()])
        
        # X = torch.cat(X).cpu().detach().numpy()
        X = X
        kmeans.fit(X)

        predbatches = [] #will be list of batch-size ndarray vectors
        for batch, i in pred_loader:
            latent = self.model.to_latent(batch.cuda()).cpu().detach().numpy()
            predbatches.append(kmeans.predict(latent))

        # concatenate preds 
        predictions = np.concatenate(predbatches) #cluster of each point
        ids, counts = np.unique(predictions[self.trainer.datamodule.indicesS0+self.trainer.datamodule.indicesS1==1], return_counts=True)
        
        v = torch.zeros((len(pred_loader.dataset) ,))
        if ids.shape[0] != self.num_clusters: #if not all clusters are present in predictions, non-present ones are the rarest
            missing = set(range(self.num_clusters)) - set(ids)
            for m in missing: #not efficient but there will only be one or two such m. revisit later
                v[predictions==m] = 1
            self.log('worst_cluster', m)
        else: #all are present so select the one with fewest occurences
            rarest = np.argmin(counts) #index (=group id) of rarest group
            v[predictions==rarest] = 1
            self.log('worst_cluster', rarest)

        self.clusters.append(predictions)
        self.analyzeClusters(predictions)
        return v


    def lossS1(self):
        fit_loader = self.trainer.datamodule.train_dataloader()
        pred_loader = self.trainer.datamodule.train_all_dataloader()
        batch_size = self.trainer.datamodule.train_batch_size
        kmeans = MiniBatchKMeans(n_clusters=self.num_clusters, random_state=2, batch_size=batch_size)
        
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
        allids = set(range(self.num_clusters))
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
        self.log('worst_cluster', worst)
        self.clusters.append(predictions)
        return v


    def analyzeClusters(self, clusters): #pass in JUST the available training data
        self.clusters.append(clusters)
        for c in range(self.num_clusters):
            # find examples in cluster c
            cluster = (clusters==c).astype(np.float32)
            clusterS0 = cluster[self.trainer.datamodule.indicesS0]
            clusterS1 = cluster[self.trainer.datamodule.indicesS1]
            self.log(f'cluster{c}_all', cluster.sum())
            self.log(f'cluster{c}_S0', clusterS0.sum())
            self.log(f'cluster{c}_availS1', clusterS1.sum())
            self.log(f'cluster{c}_avail', clusterS0.sum() + clusterS1.sum())
            # figure out which labels and other metadata they are, how many are in S0 vs S1 per cluster


    def on_train_epoch_end(self) -> None:
        #https://pytorch-lightning.readthedocs.io/en/stable/guides/data.html#accessing-dataloaders-within-lightningmodule
        # self.log('train_frechet', self.train_fid.compute().item())
        if self.S1func is not None:
            v = self.S1func()
            s1 = self.trainer.datamodule.v_update(v, self.k)  #returns 1hot vector of chosen S1 points
            self.availS1.append(s1)
            self.trainer.reset_train_dataloader(self)
        
        if self.active:
            traindata = self.trainer.datamodule.train_dataset
            n = len(traindata)
            # choose random subset of n samples
            randy = torch.rand(size=(len(traindata), ))
            indices = [i.item() for i in torch.topk(randy, n)[1]] #actual index numbers
            inputdata = torch.utils.data.Subset(traindata, indices)
            input = DataLoader(inputdata, 
                                batch_size=64, # revisit
                                num_workers=4,
                                shuffle=False,
                                pin_memory=True)

            # get each example's distribution info for current epoch
            mulist = []
            logvarlist = []
            for b, _ in input:
                res = self.model.forward(b.cuda())
                mulist.append(res[2])
                logvarlist.append(res[3])
            exmu = torch.cat(mulist).cpu().detach() # n x latent_dims
            exvar = math.e ** torch.cat(logvarlist).cpu().detach() # n x latent_dims

            # calculate each dimension's mixture distribution
            D = self.model.latent_dims
            for dim in range(D): # get distrib info for each dimension - probably could tensorize this
                mean = exmu[:,dim].sum()/n #mixture distribution
                v = (exvar[:,dim] + exmu[:,dim]**2).sum()/n - mean**2
                self.edmu[dim, self.trainer.current_epoch] = mean
                self.edvr[dim, self.trainer.current_epoch] = v

            # help a dimension
            if self.trainer.current_epoch > 0 and self.trainer.current_epoch <= 32:
                # change = torch.abs((self.edvr[:, self.trainer.current_epoch]-self.edvr[:, self.trainer.current_epoch-1]) / self.edvr[:, self.trainer.current_epoch-1])
                # help = change.argmax().item()
                varvar = statistics.variance(self.edvr[:, self.trainer.current_epoch].numpy())
                varmean = statistics.mean(self.edvr[:, self.trainer.current_epoch].numpy())
                help = torch.abs(self.edvr[:, self.trainer.current_epoch] - varvar).argmax().item()
                print(help, 'chosen to help')
                self.help.append(help)

                # get S1 sample
                s1data = self.trainer.datamodule.train_dataset_all
                k = min([10000, len(s1data)])
                randy = torch.rand(size=(len(s1data), ))
                randy[self.trainer.datamodule.indicesS0] = 0
                randy[self.trainer.datamodule.indicesS1] = 0
                indices = [i.item() for i in torch.topk(randy, k)[1]] #actual index numbers
                inputdata = torch.utils.data.Subset(s1data, indices)
                input = DataLoader(inputdata, 
                                    batch_size=64, # revisit
                                    num_workers=4,
                                    shuffle=False,
                                    pin_memory=True)

                # find exs' latent distributions in dimension dim
                mulist = []
                logvarlist = []
                for b, _ in input:
                    res = self.model.forward(b.cuda())
                    mulist.append(res[2].cpu())
                    logvarlist.append(res[3].cpu())
                exmu = torch.cat(mulist)[:, help].detach() 
                exvar = math.e ** torch.cat(logvarlist)[:, help].detach() 

                exmu_sort, exmu_sortinds = torch.sort(exmu) # sort ex mus, keeping track of original ordering
                # transform to z-scores
                helpmean = self.edmu[dim, self.trainer.current_epoch]
                helpstdev = self.edvr[dim, self.trainer.current_epoch].abs()**0.5
                exz = (exmu_sort - helpmean) / helpstdev

                q = 250 # number of samples to add (revisit)
                if self.edvr[help, self.trainer.current_epoch] < varmean: # increase its variance by adding outliers
                    chosen = self.help_outliers(exmu_sortinds, exz, q) # returns indices of 'inputdata'
                else:
                    chosen = self.help_mean(exmu_sortinds, exz, q)
                add = torch.Tensor(indices).index_select(0, torch.Tensor(chosen).int()) # actual index numbers from s1data of chosen examples
                onehot = torch.zeros(size=(len(s1data), ), dtype=torch.long)
                onehot[add.long()]=1
                self.trainer.datamodule.v_add(onehot, q)
                self.trainer.reset_train_dataloader(self)
            # print(len(self.trainer.datamodule.train_dataset))


    def help_distributed(self, exind, exz, q):
        # sample according to normal distribution defined by dim's mixture distribution's mean/var
        # dim is dimension to help, exs is examples' z-scores in that dimension
        cum_weights = ss.norm.cdf(exz) # normal function CDF
        chosen = choices(exind, cum_weights=cum_weights, k=q) # caution: this samples with replacement
        return list(set(chosen))

    def help_outliers(self, exind, exz, q):
        # sample from extremes of the dim's mixture distribution
        chosen = torch.cat([exind[:q//2], exind[:-q//2]], dim=0) # works since it's sorted
        return chosen.float()

    def help_mean(self, exind, exz, q):
        # find points closest to mean (z of 0)
        chosen = torch.topk(-torch.absolute(exz), k=q)[1].cpu()
        return chosen.float()

    def help_random(self, exind, exz, q):
        chosen = torch.topk(torch.rand_like(exz))[1]
        return chosen.float()


    # def on_train_end(self):
    #     if self.active:
    #         with open(os.path.join(self.logger.save_dir, 'helped.csv'), 'w') as f:
    #             f.write(','.join([l.item() for l in list(self.help)])+'\n')


    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels, optimizer_idx=optimizer_idx) #[output, input, intermediates...]
        val_loss = self.model.loss_function(*results,
                                            **self.params, #real_img.shape[0]/ self.num_val_imgs, #for VAEs
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        # self.val_fid.update(results[0].type(torch.uint8), real=False)
        # self.val_fid.update(results[1].type(torch.uint8), real=True)

        # samples = self.model.sample(64, self.curr_device)
        # self.val_inception.update((255*samples).type(torch.uint8))

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True, on_step=False, on_epoch=True)
        self.log("val_L1_distance", self.val_l1difference(results[0], results[1]), on_step=False, on_epoch=True)


    def on_validation_epoch_end(self) -> None:
        samples = self.sample_images()[1] #saves images to file
        # imean, istd = self.val_inception.compute()
        # self.log('val_inception_mean', imean.item())
        # self.log('val_inception_stdv', istd.item())
        # self.log('val_frechet', self.val_fid.compute().item())


    def on_test_start(self) -> None:
        self.test_inception = InceptionScore().cuda()
        self.test_fid = FrechetInceptionDistance().cuda()
        self.test_l1difference = torch.nn.L1Loss()
        

    def test_step(self, batch, batch_idx) -> None:
        recons, samples, origs = self.sample_images(batch, tofile=False, num_samples=64)
        # print(recons.min(), recons.max(), origs.min(), origs.max())
        self.test_inception.update((255*samples).type(torch.uint8))
        self.test_fid.update(recons.type(torch.uint8), real=False)
        self.test_fid.update(origs.type(torch.uint8), real=True)
        self.log("test_L1_distance", self.test_l1difference(origs, recons))


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
                            os.path.join(self.logger.save_dir , 
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
                                os.path.join(self.logger.save_dir , 
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
