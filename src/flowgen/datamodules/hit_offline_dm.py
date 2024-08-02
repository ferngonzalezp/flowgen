import numpy as np
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.interpolate import interpn
import h5py as h5
import os
from time import sleep
from typing import List
from lightning.pytorch.utilities import CombinedLoader
import re

def create_data_list(data_dirs: List, seq_len: int):
        file_list = {}
        it_list_total = []
        case = 0
        
        for i in range(len(data_dirs)):
            data_path = data_dirs[i]
            it_list= []
            sim = 0
            for root, dirs, files in os.walk(data_path):
                if 'domain' in root:
                        it_list_sim = []
                        file_list[str(sim + case)] = {}
                        for name in files:
                            if all(x in name for x in ['data','h5']):
                                file_name = os.path.join(root, name)
                                iteration = name.replace('data_','')
                                iteration = float(iteration.replace('.h5',''))
                                file_list[str(sim  + case)].update({'{}'.format(iteration): file_name})
                                it_list_sim.append(iteration)
                        sim += 1
                        it_list_sim.sort()
                        it_list.append(np.array(it_list_sim))
            case = sim * (i+1)
            it_list = np.stack(it_list)
            it_list = it_list[...,:seq_len*(it_list.shape[1]//seq_len)].reshape(it_list.shape[0],-1,seq_len)
            it_list_total.append(it_list)
            
        #it_list_total = np.concatenate(it_list_total)
            
        return [file_list, it_list_total]

class hitDataset(Dataset):
     def __init__(self, data_list):
         
        file_list, it_list = data_list
        self.it_list = it_list
        self.file_list = file_list

     def __len__(self):
        return len(self.it_list[0][0]) * len(self.file_list)

     def __getitem__(self, idx):
            idx_sim = idx // len(self.it_list[0][0])
            idx_seq = idx % (len(self.it_list[0][0]))
            idx_case = idx // (self.__len__() // len(self.it_list))
            field= []
            time=[]
            sims_per_case = len(self.file_list) // len(self.it_list)
            
            for it in self.it_list[idx_case][idx_sim - idx_case*sims_per_case][idx_seq]:
                file = h5.File(self.file_list[str(idx_sim)][str(it)])
                var = []
                for key in file['primes'].keys():
                    var.append(file['primes'][key][:])
                field.append(np.stack(var))
                time.append(it)
            field = np.stack(field, -1, dtype=np.float32)
            time = np.stack(time, -1, dtype=np.float32)
            return field, idx, idx_case, time

class hitOfflineDataModule(L.LightningDataModule):
    def __init__(self, val_dirs: List[str], data_dir: List[str], batch_size: int = 16, seq_len: List[int] = [10, 100]):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.val_dirs = val_dirs
        self.val_data=[]
        for i in range(len(self.val_dirs)):
            self.val_data.append(create_data_list(self.val_dirs[i:i+1],self.seq_len[1]))
        self.train_data= create_data_list(self.data_dir,seq_len[0])

    def prepare_data(self):
        None

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage in ["fit", "validate"]:
            self.train_ds = hitDataset(self.train_data)
            self.val_ds =  []
            
            for i in range(len(self.val_data)):
                self.val_ds.append(hitDataset(self.val_data[i]))

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.val_ds =  []
            for i in range(len(self.val_data)):
                self.val_ds.append(hitDataset(self.val_data[i]))
        if stage == "predict":
            self.val_ds =  []
            for i in range(len(self.val_data)):
                self.val_ds.append(hitDataset(self.val_data[i]))

    def train_dataloader(self):
        dl = DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=4, shuffle=True)
        return dl

    def val_dataloader(self):
        dl = []
        for i in range(len(self.val_ds)):
            dl.append(DataLoader(self.val_ds[i], batch_size=self.batch_size, num_workers=0))
        return CombinedLoader(dl, "sequential")

    def test_dataloader(self):
        dl = []
        for i in range(len(self.val_ds)):
            dl.append(DataLoader(self.val_ds[i], batch_size=self.batch_size, num_workers=0))
        return CombinedLoader(dl, "sequential")

    def predict_dataloader(self):
        dl = []
        for i in range(len(self.val_ds)):
            dl.append(DataLoader(self.val_ds[i], batch_size=self.batch_size, num_workers=0))
        return CombinedLoader(dl, "sequential")