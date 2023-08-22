'''
    Author: Theodor Cheslerean-Boghiu
    Date: May 26th 2023
    Version 1.0
'''
import numpy as np
from typing import Union, Any, Optional, Callable

import pandas as pd

import torch
import torch.utils.data as data
import pytorch_lightning as pl
from torchvision.transforms import Normalize, Compose

from tiatoolbox.models.dataset.classification import WSIPatchDataset

class HistoDataset(data.Dataset):
    def __init__(self, 
                 root: str,
                 transform: Optional[Callable] =None,
                 resolution: float =1,
                 min_mask_ratio: float=0.5,
                 train: bool =True,
                 ):
        super().__init__()
        
        self.root = root
        self.transform = transform
        self.train = train
        self.resolution = resolution
        self.min_mask_ratio = min_mask_ratio
        
        self.dataframe = pd.read_csv(root + 'metadata.csv')
        self.num_classes = len(self.dataframe['label'].unique())
        
        self.train_set = self.dataframe.groupby('label').sample(frac=0.6, random_state=200).reset_index()
        self.val_set = self.dataframe.drop(self.train_set.index).reset_index()
        
        self.dataframe = self.train_set
        if not train:
            self.dataframe = self.val_set
            
            
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        
        img_path = self.root + "slides/" + row['img_path']
        
        label = row['label']
        
        ds = WSIPatchDataset(
            img_path=img_path,
            mode="wsi",
            patch_input_shape=[256, 256],
            stride_shape=[256, 256],
            auto_get_mask=True,
            min_mask_ratio=self.min_mask_ratio,
            resolution=self.resolution,
            units="baseline",
            # preproc_func=preproc_func
        )
        
        patch_stack = (np.transpose(np.stack([patch['image'] for patch in ds]), (0,3,1,2)) / 255.0).astype(np.float32)
        patch_stack = torch.from_numpy(patch_stack)

        if self.transform:
            patch_stack = self.transform(patch_stack)

        return patch_stack, torch.from_numpy(np.array(label).astype(np.int64))
      
class WrapperDataset(pl.LightningDataModule):
    """Main implementation of the full dataset
    
    Generates splits at initialization. 
    Generates the DataLoaders.

    Attributes:
        root (str): Directory path for the data
        batch_size (int): Integer denoting the batch size needed by the Dataloader constructor
        transform_train (albumentations.Compose): Composition of augmentation operations for the training set
        transform_test (albumentations.Compose): Composition of augmentation operations for the training set. Will only contain the normalization operation.
        train_data: Training Dataset
        val_data: Validation Dataset
        test_data: Test Dataset (#TODO: for now it is just the validation split)
        num_classes (int): Total number of classes
    """    

    def __init__(self,
                 root: str=None,
                 batch_size: int=16,
                 resolution: float=1,
                 transforms: Optional[Callable] =None):
        """_summary_

        Args:
            root (str, optional): _description_. Directory path for the data
            batch_size (int, optional): _description_. Defaults to 16.
        """        
        super().__init__()
        
        self.batch_size = batch_size
        self.root = root

        self.transform_train = transforms

        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]

        self.transform_test = Compose(
            [
                Normalize(mean=norm_mean, std=norm_std),
            ])
        
        self.train_data = HistoDataset(
            self.root,
            train=True,
            transform=self.transform_train,
            resolution=resolution)
        
        self.val_data = HistoDataset(
            self.root,
            train=False,
            transform=self.transform_test,
            resolution=resolution)
        
        self.num_classes = self.train_data.num_classes
        
        
            
    def prepare_data(self):
        pass

    def train_dataloader(self):
        return data.DataLoader(self.train_data,
                                           batch_size=self.batch_size,
                                           num_workers=(self.batch_size if self.batch_size <= 32 else 32),
                                           shuffle=True,
                                           pin_memory=True,
                                           drop_last=True)

    def val_dataloader(self):
        return data.DataLoader(self.val_data,
                                           batch_size=self.batch_size,
                                           num_workers=(self.batch_size if self.batch_size <= 32 else 32),
                                           shuffle=True,
                                           pin_memory=True,
                                           drop_last=True)
        
    def predict_dataloader(self):
        return data.DataLoader(self.val_data,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           drop_last=False)
