from torch.utils.data import Dataset
from typing import List
from hit_offline_dm import create_data_list, hitDataset
from server_baseclass import adios2DataModule
from lightning.pytorch.utilities import CombinedLoader
from torch.utils.data import DataLoader

class hitDataModule(adios2DataModule):

     def __init__(self, val_dirs, seq_len: List[int] = [10, 100], data_dir: str = "./train", 
                 batch_size: int = 16, reservoir_treshold: int = 128,
                 var_names = ["density", "velocityX", "velocityY", "velocityZ", "pressure", "temperature", "time"], 
                 adios_cfg: str = "./adios2.xml", target_dims = [32,32,32]):

        super().__init__(data_dir, 
                 batch_size, reservoir_treshold ,
                 var_names, 
                 adios_cfg , target_dims)

        self.val_dirs = val_dirs
        self.val_data = []
        for i in range(len(self.val_dirs)):
            self.val_data.append(create_data_list(self.val_dirs[i:i+1],seq_len[1]))
    
     def setup(self, stage: str):
        
        super().setup(stage)

        self.val_ds =  []
        for i in range(len(self.val_data)):
            self.val_ds.append(hitDataset(self.val_data[i]))
    
     def val_dataloader(self):
        dl = []
        for i in range(len(self.val_ds)):
            dl.append(DataLoader(self.val_ds[i], batch_size=5, num_workers=0))
        return CombinedLoader(dl, "sequential")
            