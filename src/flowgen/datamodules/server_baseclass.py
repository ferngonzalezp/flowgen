import torch
import numpy as np
import lightning as L
from torch.utils.data import DataLoader, Dataset, RandomSampler
import adios2.bindings as adios2
import os
from typing import List
from mpi4py import MPI
from tensordict.tensordict import TensorDict

class adios2StreamDataset(Dataset):
    def __init__(self, data_dir: str, 
                 reservoir_treshold: int, 
                 var_names: List[str], 
                 target_dims: List[int], 
                 engines: List, io: any):
        super().__init__()
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.reservoir_treshold = reservoir_treshold
        self.data_dir = data_dir

        self.var_names = var_names
        self.end_of_stream = False
        self.target_dims =  target_dims
        self.io = io

        self.engines = engines
        self.engines_status = [True] * len(engines)

        seen = []
        unseen = []
        self.reservoir = TensorDict.load_memmap("./reservoir")
            
    def __len__(self):
        return self.reservoir.size(0)
    
    def __getitem__(self, idx):

        field_dict = self.reservoir[idx].to_dict()
        self.reservoir.set_at_("seen", 1, idx)
        field = []
        time = None
        for varname in self.var_names:
            if varname =='time':
                time = field_dict[varname]
            else:
                field.append(field_dict[varname])
        field = torch.stack(field).type(dtype=torch.float32)

        #if self.rank == 0:
        #        self.get_new_data()
        
        return [field, idx, time]
        
    def get_new_data(self):

        StepNotFound = None
        var = dict()
        nprocessed = 0
        #self.reservoir = self.reservoir.clone()
        while nprocessed < self.reservoir_treshold:
            n_engine = 0
            for engine in self.engines:
                try:
                    step_status = engine.BeginStep(mode=adios2.StepMode.Read,timeoutSeconds=0.0)
                except:
                    #SEARCH NEW SIMULATIONS
                                        try:
                                            sst_files = []
                                            for root, dirs, files in os.walk(self.data_dir):
                                                for file in files:
                                                    if file.endswith(".sst"):
                                                        sst_files.append(os.path.join(root, file.replace(".sst","")))
                                                    for i in range(len(sst_files)):
                                                            self.engines[n_engine] = (self.io.Open(os.path.relpath(sst_files[i]), adios2.Mode.Read))
                                                            self.engines_status[n_engine] = True
                                        except:
                                               pass
                else:
                    if step_status == adios2.StepStatus.OK:
                            for i in range(len(self.var_names)):
                                var_id = self.io.InquireVariable(self.var_names[i])
                                if nprocessed == 0:
                                    var[self.var_names[i]] = np.zeros(var_id.Count(), dtype=var_id.Type())
                                engine.Get(var_id, var[self.var_names[i]])
                            engine.EndStep()
                            n = self.reservoir["loss"].argmin()
                            file_processed = False
                            while file_processed==False:
                                #if self.reservoir[n]["seen"] == 1:
                                        for i in range(len(self.var_names)):
                                            self.reservoir.set_at_(self.var_names[i], torch.from_numpy(var[self.var_names[i]]), n)
                                            self.reservoir.set_at_("seen", 0, n)
                                            self.reservoir.set_at_("loss", 1.0, n)
                                        file_processed = True
                                #else:
                                #        n = self.reservoir["loss"].argmin()
                            nprocessed += 1
                    elif step_status == adios2.StepStatus.EndOfStream: 
                                    engine.Close()
                                    self.engines_status[n_engine] = False
                                    StepNotFound = True
                    else:
                        if n_engine == len(self.engines) - 1:
                            StepNotFound = True
                            
                n_engine += 1
            #if n_engine == len(self.engines) - 1:
            #                StepNotFound = True
            if StepNotFound == True:
                break
        del var
        #self.reservoir.memmap("./reservoir")
        if sum(self.engines_status) == 0 & sum(self.reservoir["seen"])==self.reservoir_treshold:
            self.end_of_stream = True

class adios2DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, 
                 batch_size: int, reservoir_treshold: int ,
                 var_names : List[str], 
                 adios_cfg: str , target_dims : List[int]):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.reservoir_treshold = reservoir_treshold
        self.adios_cfg = adios_cfg
        self.var_names = var_names
        self.target_dims = target_dims

        #Initialize ADIOS2
        self.comm = MPI.COMM_WORLD
        rank = self.comm.Get_rank()
        if rank ==0:
            self.adios = adios2.ADIOS(self.adios_cfg)
            self.io     = self.adios.DeclareIO("readerIO")
            sst_files = []
            for root, dirs, files in os.walk(self.data_dir):
                for file in files:
                    if file.endswith(".sst"):
                        sst_files.append(os.path.join(root, file.replace(".sst","")))
            engines = []
            for i in range(len(sst_files)):
                engines.append(self.io.Open(os.path.relpath(sst_files[i]), adios2.Mode.Read))
            
            self.engines = engines
        else:
             self.engines = []
             self.io =  None

    def prepare_data(self):
        print("Filling reservoir...")
        nprocessed = 0
        var = dict()
        reservoir =  TensorDict({}, batch_size=[self.reservoir_treshold])
        while nprocessed < self.reservoir_treshold:
            for n_engine in range(len(self.engines)):
                try:
                    engine = self.engines[n_engine]
                    engine.BeginStep(mode=adios2.StepMode.Read,timeoutSeconds=200.0)
                    for i in range(len(self.var_names)):
                        var_id = self.io.InquireVariable(self.var_names[i])
                        if nprocessed == 0:
                            var[self.var_names[i]] = np.zeros(var_id.Count(), dtype=var_id.Type())
                            reservoir[self.var_names[i]] = torch.zeros((self.reservoir_treshold,*var_id.Count()))
                        reservoir["seen"] = torch.zeros(self.reservoir_treshold, dtype=torch.int)
                        reservoir["loss"] = torch.zeros(self.reservoir_treshold, dtype=torch.float)
                        engine.Get(var_id, var[self.var_names[i]])
                    engine.EndStep()

                    for i in range(len(self.var_names)):
                        reservoir.set_at_(self.var_names[i], torch.from_numpy(var[self.var_names[i]]), nprocessed)
                    nprocessed += 1
                except:
                            try:
                                engine.Close()
                            except:
                                #SEARCH NEW SIMULATIONS
                                        try:
                                            sst_files = []
                                            for root, dirs, files in os.walk(self.data_dir):
                                                for file in files:
                                                    if file.endswith(".sst"):
                                                        sst_files.append(os.path.join(root, file.replace(".sst","")))
                                                    for i in range(len(sst_files)):
                                                            self.engines[n_engine] = (self.io.Open(os.path.relpath(sst_files[i]), adios2.Mode.Read))
                                        except:
                                                pass
        del var
        print("Reservoir Full!")
        reservoir.memmap("./reservoir")

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_ds = adios2StreamDataset(self.data_dir, self.reservoir_treshold, self.var_names, self.target_dims, self.engines, self.io)

    def train_dataloader(self):
        dataloader = DataLoader(self.train_ds, batch_size=self.batch_size, 
                          sampler = RandomSampler(self.train_ds, replacement=True), 
                          num_workers=0)
        return dataloader