import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA, MNIST


# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self):
        pass
    
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    
    def _check_integrity(self) -> bool:
        return True
    

class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"        
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, 0.0 # dummy datat to prevent breaking 


class Dataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_name: str,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        num_samples: int = 0, #0=use full dataset size. Num points in S0
        S1: bool = False, #whether to add additional points from S1
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs
    ): 
        super().__init__()
        
        self.name = data_name
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_samples = num_samples
        self.S1 = S1
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        if self.name.lower() == 'oxfordpets':
                train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                      transforms.CenterCrop(self.patch_size),
                                                      transforms.ToTensor(),
                                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
                
                val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                    transforms.CenterCrop(self.patch_size),
                                                    transforms.ToTensor(),
                                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

                self.train_dataset_all = OxfordPets(
                    self.data_dir,
                    split='train',
                    transform=train_transforms,
                )
                
                self.val_dataset = OxfordPets(
                    self.data_dir,
                    split='val',
                    transform=val_transforms,
                )

                
        elif self.name.lower() == 'celeba':
            train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.CenterCrop(148),
                                                transforms.Resize(self.patch_size),
                                                transforms.ToTensor(),])
            
            val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.CenterCrop(148),
                                                transforms.Resize(self.patch_size),
                                                transforms.ToTensor(),])
            
            self.train_dataset_all = MyCelebA(
                self.data_dir,
                split='train',
                transform=train_transforms,
                download=True,
            )
            
            # Replace CelebA with your dataset
            self.val_dataset = MyCelebA(
                self.data_dir,
                split='test',
                transform=val_transforms,
                download=True,
            )


        elif self.name.lower() == 'mnist':
            trans = transforms.Compose([transforms.Resize(self.patch_size),
                                        transforms.Grayscale(3),
                                        transforms.ToTensor() ])
            self.train_dataset_all = MNIST(self.data_dir, train=True, transform=trans, download=True)
            self.val_dataset = MNIST(self.data_dir, train=False, transform=trans, download=True)

        if self.num_samples > 0:
            # generate random list and take top num_samples values as 1, all others as 0. These are S0 indices
            randy = torch.rand(size=(len(self.train_dataset, )))
            self.indicesS0 = torch.zeros(size=(len(self.train_dataset)))
            self.indicesS0[torch.topk(randy, self.num_samples)] = 1 #one-hot

            self.S0 = torch.utils.data.Subset(self.train_dataset_all, self.indicesS0.nonzero(as_tuple=True).tolist())
            self.train_dataset = self.S0 #initially just use S0
            print('using random S0 subset consisting of', len(self.train_dataset), 'training examples')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )


class superdata(LightningDataModule):
    """The data is comprised of two sets, S_0 and S_1. S_0 is all points currently available
    to model training and S_1 is points that are not currently available"""

    def __init__(
        self,
        data_name: str,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        num_samples: int = 0, #0=use full dataset size. S0 size
        S1: bool = False, #whether to add additional points from S1
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs
    ):
        super().__init__()
        
        self.name = data_name
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory #whether to use GPU

        self.data = Dataset(data_name, data_path, train_batch_size, val_batch_size, 0, patch_size, num_workers, pin_memory)

    def setup(self, stage: Optional[str] = None) -> None:
        self.data.setup()
        
        
        # make another list with S0 indices' elements set to -Inf, all others by 1. Multiply by random numbers and choose top k to get 
        # indices of S1 to add to S0
 