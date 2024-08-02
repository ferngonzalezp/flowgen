import torch
import lightning as L
import numpy as np
from flowgen import hitDataModule
from time import sleep
from mpi4py import MPI

reservoir = 10
dm = hitDataModule(val_dirs = ["/scratch/cfd/gonzalez/hit_online_training/04_HIT_decay/case1", "/scratch/cfd/gonzalez/hit_online_training/04_HIT_decay/case2", 
                               "/scratch/cfd/gonzalez/hit_online_training/04_HIT_decay/case3"], seq_len=(10, 100), data_dir="train_online", batch_size=4, reservoir_treshold=reservoir, target_dims=[32,32,32])
comm = MPI.COMM_WORLD
dm.prepare_data()
dm.setup(stage='fit')

class get_new_data_callback(L.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.datamodule.use_patches:
            trainer.datamodule.train_ds.seen = trainer.datamodule.train_ds.unseen
            trainer.datamodule.train_ds.unseen = []
            trainer.datamodule.train_ds.get_new_data()

epoch = 0
while dm.train_ds.end_of_stream != True:
    for i, batch in enumerate(dm.train_dataloader()):
        print("epoch: ", epoch," it: ",i, batch[0].shape, batch[0].type(), batch[0].device, batch[-1].shape)
        print("N seen: ", sum(dm.train_ds.reservoir["seen"]), " N unseen: ",  reservoir - sum(dm.train_ds.reservoir["seen"]))
        print(dm.train_ds.end_of_stream)
    #dm.train_ds.seen = dm.train_ds.unseen
    #dm.train_ds.unseen = []
    #if comm.Get_rank() == 0:
    #    dm.train_ds.get_new_data()
    print("Validating...")
    for batch, batch_idx, dataloader_idx in dm.val_dataloader():
            print("epoch: ", epoch," it: ",dataloader_idx, batch[0].shape, batch[0].type(), batch[0].device)
    epoch += 1
sleep(4)