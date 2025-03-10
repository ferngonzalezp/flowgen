import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.fft import fftn
from numpy import sqrt,zeros,arange, ones
from torch import conj
from numpy import pi
from scipy.signal import convolve
import random

# def compute_tke_spectrum(u, lx = 6.283185307179586, ly = 6.283185307179586, 
#                          lz=6.283185307179586, axes: tuple = None, one_dimensional = False):
#     """
#     Given a velocity field u this function computes the kinetic energy
#     spectrum of that velocity field in spectral space. This procedure consists of the
#     following steps:
#     1. Compute the spectral representation of u using a fast Fourier transform.
#     This returns uf (the f stands for Fourier)
#     2. Compute the point-wise kinetic energy Ef (kx, ky, kz) = 1/2 * (uf)* conjugate(uf)
#     3. For every wave number triplet (kx, ky, kz) we have a corresponding spectral kinetic energy
#     Ef(kx, ky, kz). To extract a one dimensional spectrum, E(k), we integrate Ef(kx,ky,kz) over
#     the surface of a sphere of radius k = sqrt(kx^2 + ky^2 + kz^2). In other words
#     E(k) = sum( E(kx,ky,kz), for all (kx,ky,kz) such that k = sqrt(kx^2 + ky^2 + kz^2) ).

#     Parameters:
#     -----------
#     u: 5D array Bs x components x Nx x Ny x Nz
#       The velocity field tensor.
#     lx: float
#       The domain size in the x-direction.
#     ly: float
#       The domain size in the y-direction.
#     lz: float
#       The domain size in the z-direction.
#     smooth: boolean
#       A boolean to smooth the computed spectrum for nice visualization.
#     """
#     nx = len(u[0,0,:, 0, 0])
#     ny = len(u[0,0,0, :, 0])
#     nz = len(u[0,0,0, 0, :])

#     nt = nx * ny * nz
#     n = max(nx, ny, nz)  # int(np.round(np.power(nt,1.0/3.0)))

#     uh = fftn(u, dim = axes) / nt

#     # tkeh = zeros((nx, ny, nz))

#     tkeh = 0.5 * (torch.sum(uh * conj(uh), dim=1)).real

#     length = max(lx, ly, lz)

#     knorm = 2.0 * pi / length

#     kxmax = nx / 2
#     kymax = ny / 2
#     kzmax = nz / 2

#     wave_numbers = arange(0, n)
#     tke_spectrum = torch.zeros((u.shape[0],len(wave_numbers)))
#     kx = np.concatenate((np.arange(0,kxmax), abs(np.arange(-kxmax, 0))))
#     ky = np.concatenate((np.arange(0,kymax), abs(np.arange(-kymax, 0))))
#     kz = np.concatenate((np.arange(0,kzmax), abs(np.arange(-kzmax, 0))))

#     Kx, Ky, Kz = np.meshgrid(kx, ky, kz)

#     k_matrix = np.round((Kx**2 + Ky**2 + Kz**2)**0.5)

#     if one_dimensional == True:
#       for i in range(len(wave_numbers)):
#         tke_spectrum[:,i] = tkeh[:,np.where(k_matrix == wave_numbers[i])].sum(dim=tuple(range(1,len(tkeh.shape)+1)))

#       tke_spectrum = tke_spectrum / knorm
#     else:
#       tke_spectrum = tkeh

#     knyquist = knorm * min(nx, ny, nz) / 2

#     return knyquist, wave_numbers * knorm, tke_spectrum

def compute_tke_spectrum(u, lx=6.283185307179586, ly=6.283185307179586,
                         lz=6.283185307179586, axes: tuple = None, one_dimensional=False):
    """
    Given a velocity field u this function computes the kinetic energy
    spectrum of that velocity field in spectral space, using native PyTorch operations for fully batched computation.
    """
    nx = len(u[0, 0, :, 0, 0])
    ny = len(u[0, 0, 0, :, 0])
    nz = len(u[0, 0, 0, 0, :])
    nt = nx * ny * nz
    n = max(nx, ny, nz)
    uh = fftn(u, dim=axes) / nt
    tkeh = 0.5 * (torch.sum(uh * conj(uh), dim=1)).real  # Shape: [batch, nx, ny, nz]
    length = max(lx, ly, lz)
    knorm = 2.0 * pi / length
    kxmax = nx // 2
    kymax = ny // 2
    kzmax = nz // 2
    wave_numbers = torch.arange(0, n, device=tkeh.device)
    
    if one_dimensional:
        # Create k_matrix
        kx = torch.cat((torch.arange(0, kxmax, device=tkeh.device), 
                        torch.abs(torch.arange(-kxmax, 0, device=tkeh.device))))
        ky = torch.cat((torch.arange(0, kymax, device=tkeh.device), 
                        torch.abs(torch.arange(-kymax, 0, device=tkeh.device))))
        kz = torch.cat((torch.arange(0, kzmax, device=tkeh.device), 
                        torch.abs(torch.arange(-kzmax, 0, device=tkeh.device))))
        
        # Creating meshgrid in PyTorch
        kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing='ij')
        k_matrix = torch.round((kx**2 + ky**2 + kz**2)**0.5).long()  # Shape: [nx, ny, nz]
        
        # Get the batch size
        batch_size = u.shape[0]
        
        # Initialize output tensor
        #tke_spectrum = torch.zeros((batch_size, n), device=tkeh.device, requires_grad=True)
        tke_spectrum = []
        
        # One-hot encode k_matrix for each wave number (most efficient native PyTorch approach)
        # This avoids loops over wave numbers and batches
        for k in range(n):
            # Create binary mask for this wave number
            mask = (k_matrix == k)
            # If no elements match this wave number, skip
            if not mask.any():
                continue
            # Apply mask to all batches at once using broadcasting
            masked_tkeh = tkeh * mask.unsqueeze(0)
            # Sum over spatial dimensions for each batch
            tke_spectrum.append(masked_tkeh.sum(dim=(1, 2, 3)))
        
        tke_spectrum = torch.stack(tke_spectrum, dim=-1) / knorm
    else:
        tke_spectrum = tkeh
    
    knyquist = knorm * min(nx, ny, nz) / 2
    return knyquist, wave_numbers * knorm, tke_spectrum

def nrmse_loss(x, y, p = 2, w=None, reduction=True):
    err =  torch.abs(x-y)**p
    norm_y = torch.abs(y)**p
    dims_sum = tuple(range(2,(len(x.shape))))
    if len(x.shape) ==  6:
        err =  torch.sum(err, dim=dims_sum[:-1])
        norm_y =  torch.sum(norm_y, dim=dims_sum[:-1])

        epsilon = 1e-8

        loss = torch.sum(err/(norm_y + epsilon), dim=-1)
    else:   
        err =  torch.sum(err, dim=dims_sum)
        norm_y =  torch.sum(norm_y, dim=dims_sum)

        epsilon = 1e-8

        loss = err / (norm_y + epsilon)

    if w:
        assert len(w) == loss.shape[1], "Weight shape must be the same as num features"
        loss = torch.tensor(w, device=x.device).unsqueeze(0) * loss

    if reduction:   
        loss = torch.mean(loss.sum(dim=1))
    else:
        pass
    return loss

def rH1loss(x, y, w=None, reduction=True):
    grad_x = torch.cat(torch.gradient(x, dim=tuple(range(2,len(x.shape)-1))), 1)
    grad_y = torch.cat(torch.gradient(y, dim=tuple(range(2,len(x.shape)-1))), 1)
    
    return nrmse_loss(grad_x, grad_y, w=w, reduction=reduction)

def spec_loss(x, y, N_bins=8, w=None, reduction=True):
    
    if len(x.shape) ==  4:
        B,C,H,W = x.shape
        knyquist, k, x_spec = compute_tke_spectrum(x, axes=(2,3), one_dimensional=True)
        knyquist, k, y_spec = compute_tke_spectrum(y, axes=(2,3), one_dimensional=True)
        K = k.shape[-1]
        step = K//N_bins
        loss = 0.0
        for n in range(0, K, step):
            energy_ratio = (x_spec[:,n:n+step] + 1e-12) /(y_spec[:,n:n+step] + 1e-12)
            loss += torch.mean((1 - energy_ratio)**2)
        loss = loss / N_bins
    elif len(x.shape) ==  5:
        B, C, H, W, D = x.shape
        knyquist, k, x_spec = compute_tke_spectrum(x, axes=(2,3,4), one_dimensional=True)
        knyquist, k, y_spec = compute_tke_spectrum(y, axes=(2,3,4), one_dimensional=True)
        K = k.shape[-1]
        step = K//N_bins
        loss = 0.0
        for n in range(0, K, step):
            energy_ratio = (x_spec[:,n:n+step] + 1e-12) /(y_spec[:,n:n+step] + 1e-12)
            loss += torch.mean((1 - energy_ratio)**2)
        loss = loss / N_bins
    elif len(x.shape) ==  6:
        x = (x - x.mean(dim=(2,3,4), keepdim=True)) / (x.std(dim=(2,3,4), keepdim=True) + 1e-8)
        y = (y - y.mean(dim=(2,3,4), keepdim=True)) / (y.std(dim=(2,3,4), keepdim=True) + 1e-8)
        B, C, H, W, D, T = x.shape
        loss_t = 0.0
        loss = [0.0 for t in range(T)]
        for t in range(T):
            knyquist, k, x_spec = compute_tke_spectrum(x[...,t], axes=(2,3,4), one_dimensional=True)
            knyquist, k, y_spec = compute_tke_spectrum(y[...,t], axes=(2,3,4), one_dimensional=True)
            K = k.shape[-1]
            step = K//N_bins
            for n in range(0, K, step):
                #energy_ratio = (x_spec[:,n:n+step] + 1e-8) /(y_spec[:,n:n+step] + 1e-8)
                #loss[t] = loss[t] + torch.mean((1 - energy_ratio)**2)
                loss[t] = loss[t] + torch.sum((x_spec[:,n:n+step] - y_spec[:,n:n+step])**2)/(torch.sum(y_spec[:,n:n+step]**2)+1e-8)
            loss[t] = loss[t] / N_bins
            loss_t = loss_t + loss[t]
        #loss_t = loss_t / T
    return loss_t
    
def energy_loss(x,y, w=None, reduction=True):
    kx = 0.5 * torch.sum(x[:,3:].std(dim=(2,3,4), keepdim=True)**2, dim=1, keepdim=True)
    ky = 0.5 * torch.sum(y[:,3:].std(dim=(2,3,4), keepdim=True)**2, dim=1, keepdim=True)
    return nrmse_loss(kx, ky, w=w, reduction=reduction)

def state_reg(x, y=None, w = 1.0, r = 4.4642857142857135):
    rho = x[:,0:1]
    P = x[:,1:2]
    T = x[:,2:3]
    return w * torch.mean((P-rho*r*T)**2)

def autocorrelation(x):
    """
    Compute the autocorrelation of a tensor along the last dimension (T).
    Assumes input shape (B, C, T).
    """
    B, C, T = x.shape
    x = x - x.mean(dim=-1, keepdim=True)  # Zero-mean normalization

    # Reshape input for grouped convolution
    x_reshaped = rearrange(x, '(n b) c t -> n (b c) t', n=1) # Shape (1, B * C, T), remains grouped by batch

    # Flip x along the time axis to create the kernel
    x_flip = x #x.flip(-1)  # Shape (B, C, T)
    x_flip = rearrange(x_flip, '(n b) c t -> (b c) n t', n=1)  # Shape (1, B * C, T)

    # Perform grouped 1D convolution (each channel is convolved independently)
    autocorr = F.conv1d(x_reshaped, x_flip, padding=T-1, groups=C*B)
    autocorr = rearrange(autocorr, 'n (b c) t -> (n b) c t', b=B, c=C, n=1)
    autocorr = autocorr[:, :, T-1:]  # Keep only valid lags

    # Normalize by variance
    def f(x):
        if x == 0:
            return 1
        else:
            return x
    if x.shape[-1] > 0:
        norm_factor = ((T-1) * x.std(dim=-1, keepdim=True) ** 2 + 1e-8)
    else:
        norm_factor = x[..., -1:] + 1e-8
    return autocorr / norm_factor

def corr_loss(x, y):
    x = rearrange(x, 'b c x y z t -> b c (x y z) t')
    y = rearrange(y, 'b c x y z t -> b c (x y z) t')
    x = x.std(dim=-2)
    y = y.std(dim=-2)   
    R_x = autocorrelation(x)
    R_y = autocorrelation(y)
    return F.mse_loss(R_x, R_y)

def temp_diff_loss(x,y):
    dx = abs(x[...,1:] - x[...,:-1])**2
    dy = abs(y[...,1:] - y[...,:-1])**2
    return nrmse_loss(dx, dy, reduction=False).mean()