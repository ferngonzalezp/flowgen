import numpy as np
import lightning as L
from torch.utils.data import DataLoader, Dataset
from scipy.interpolate import interpn
import h5py as h5
import os
from time import sleep
from typing import List

def map2mesh(u, target_dims, dtype=np.float32):
    nf = u.shape[0]
    nt = u.shape[-1]
    lx, ly, lz = u.shape[-4], u.shape[-3], u.shape[-2]
    
    u_interp = np.zeros((nf,*target_dims,nt), dtype=dtype)
    
    for i in range(nf):
        for j in range(nt):
            nx = np.linspace(1,lx,lx)
            ny = np.linspace(1,ly,ly)
            nz = np.linspace(1,lz,lz)
            td = target_dims
            xyz = np.meshgrid(np.linspace(1,lx,td[0]),
                               np.linspace(1,ly,td[1]),
                               np.linspace(1,lz,td[2]), indexing='ij')

            xyz = np.stack( xyz , -1)

            u_interp[i,...,j] = interpn((nx, ny, nz), u[i,...,j], xyz, method='linear').astype(dtype)
    
    return u_interp 

def extract_patches(file, patch_id, patch_id_list, n_levels):
    idx = patch_id_list[patch_id]
    xmin = idx[0].min()
    xmax = idx[0].max()
    ymin = idx[1].min()
    ymax = idx[1].max()
    zmin = idx[2].min()
    zmax = idx[2].max()
    
    u = file['u'][:]
    v = file['v'][:]
    w = file['w'][:]
    patch = [np.stack([u[idx[0], idx[1], idx[2]],v[idx[0], idx[1], idx[2]],
                             w[idx[0], idx[1], idx[2]]]).astype(np.float32)]
    Nx, Ny, Nz = file['geometry'].shape
    nx, ny, nz = (patch[0].shape[-3], patch[0].shape[-2], patch[0].shape[-1])
    for l in range(1,n_levels+1):
        for i in range(0,Nx,Nx//(2//l)):
            if xmin >= i & xmax <= i + Nx//(2//l):
                xs = i; xe = xs + Nx//(2//l)
        for i in range(0,Ny,Ny//(2//l)):
            if ymin >= i & ymax <= i + Ny//(2//l):
                ys = i; ye = ys + Ny//(2//l)
        for i in range(0,Nz,Nz//(2//l)):
            if zmin >= i & zmax <= i + Nz//(2//l):
                zs = i; ze = zs + Nz//(2//l)
        
        idx = np.meshgrid(np.arange(xs,xe,dtype=int),
                                     np.arange(ys,ye, dtype=int),
                                     np.arange(zs,ze, dtype=int),indexing='ij')
        
        field = np.stack([u[idx[0], idx[1], idx[2]],v[idx[0], idx[1], idx[2]],
                                w[idx[0], idx[1], idx[2]]])
        patch.append(map2mesh(field[..., np.newaxis], [nx, ny, nz], np.float32)[...,0])
    
    patch = np.concatenate(patch, 0)
    del u,v,w
    return patch

def create_data_list(data_path: str, seq_len: int, use_patches: bool = False, p: int = 5):
        file_list = {}
        it_list = []
        for root, dirs, files in os.walk(data_path):
            for name in files:
                if all(x in name for x in ['sol','h5']):
                    file_name = os.path.join(data_path, name)
                    iteration = name.replace('sol_','')
                    iteration = int(iteration.replace('.h5',''))
                    file_list[str(iteration)] = {'fname':file_name}
                    it_list.append(iteration)
        it_list.sort()
        it_list = np.array(it_list)
        it_list = it_list.reshape(-1,seq_len)
        patch_id_list = [None]
        if use_patches:
            patch_id_list = []
            file = h5.File(file_list[str(it_list[0,0])]['fname'], 'r')
            mesh = file['geometry'][:]
            Nx, Ny, Nz = mesh.shape
            nx, ny, nz = (Nx//p, Ny//p, Nz//p)
            
            
            for i in range(0,Nx,nx):
                    xs = i; xe = xs + nx
                    for j in range(0,Ny,ny):
                        ys = j; ye = ys + ny
                        for k in range(0,Nz,nz):
                            zs = k; ze = zs + nz

                            patch_id_list.append(np.meshgrid(np.arange(xs,xe,dtype=int),
                                                 np.arange(ys,ye, dtype=int),
                                                 np.arange(zs,ze, dtype=int),indexing='ij'))
            
        return [file_list, it_list, patch_id_list]

class atmoDataset(Dataset):
     def __init__(self, data_list, target_dims: List[str], use_patches: bool = False, p: int = 5):
         
        file_list, it_list, patch_id_list = data_list
        self.it_list = it_list
        self.file_list = file_list
        self.target_dims = target_dims
        self.patch_id_list = patch_id_list
        self.use_patches = use_patches
        self.p = p

     def __len__(self):
        return len(self.it_list) * len(self.patch_id_list)

     def __getitem__(self, idx):
            
        if self.use_patches:
            id_it = idx // len(self.patch_id_list)
            id_patch = idx % len(self.patch_id_list)
            field = []
            for it in self.it_list[id_it]:
                file = h5.File(self.file_list[str(it)]['fname'], 'r')
                field.append(extract_patches(file, id_patch, self.patch_id_list, 2))
            field = np.stack(field, -1)
            field = {'velocity': field}
        else:
            field = []
            for it in self.it_list[idx]:
                file = h5.File(self.file_list[str(it)]['fname'], 'r')
                u = file['u'][:]
                v = file['v'][:]
                w = file['w'][:]
                field.append(np.stack([u,v,w]))
            field = np.stack(field, -1)
            field = {'velocity': map2mesh(field, self.target_dims, np.float32)}
            del u,v,w
        return field

class atmoOfflineDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size: int = 16, seq_len: int = 5, 
                 target_dims: List[str] = [64,64,64], use_patches: bool = False, p: int = 5):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.target_dims = target_dims
        self.seq_len = seq_len
        self.use_patches = use_patches
        self.p = p
        self.train_data = create_data_list(data_dir+'/train', seq_len, self.use_patches, self.p)
        self.val_data = create_data_list(data_dir+'/val', seq_len, self.use_patches, self.p)

    def prepare_data(self):
        None

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_ds = atmoDataset(self.train_data, self.target_dims, self.use_patches, self.p)
            self.val_ds = atmoDataset(self.val_data, self.target_dims, self.use_patches, self.p)
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.val_ds = atmoDataset(self.val_data, self.target_dims, self.use_patches, self.p)
        if stage == "predict":
            self.val_ds = atmoDataset(self.val_data, self.target_dims, self.use_patches, self.p)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)