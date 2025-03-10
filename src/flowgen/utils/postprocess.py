from matplotlib import pyplot as plt
import torch
from jaxfluids.post_process import load_data, create_contourplot
from flowgen.utils.loss import nrmse_loss
import numpy as np
from torch.fft import fftn
from numpy import sqrt,zeros,arange, ones
from torch import conj
from numpy import pi
from scipy.signal import convolve
import random
import os
import pandas as pd

def movingaverage(interval, window_size):
    window = ones(int(window_size)) / float(window_size)
    return convolve(interval, window)

def find_h5_directory(parent_dir="."):
    """
    Find directories containing .h5 files within the parent directory.
    
    Args:
        parent_dir (str): Path to the parent directory (defaults to current directory)
    
    Returns:
        str: Path to the directory containing .h5 files, or None if not found
    """
    try:
        # Walk through directory tree
        for root, dirs, files in os.walk(parent_dir):
            # Check if any file ends with .h5
            if any(f.endswith('.h5') for f in files):
                return root
        
        print("No directory with .h5 files found")
        return None
        
    except Exception as e:
        print(f"Error while searching for .h5 files: {e}")
        return None


# ------------------------------------------------------------------------------
def compute_tke_spectrum(u, lx = 6.283185307179586, ly = 6.283185307179586, 
                         lz=6.283185307179586, axes: tuple = None, one_dimensional = False):
    """
    Given a velocity field u this function computes the kinetic energy
    spectrum of that velocity field in spectral space. This procedure consists of the
    following steps:
    1. Compute the spectral representation of u using a fast Fourier transform.
    This returns uf (the f stands for Fourier)
    2. Compute the point-wise kinetic energy Ef (kx, ky, kz) = 1/2 * (uf)* conjugate(uf)
    3. For every wave number triplet (kx, ky, kz) we have a corresponding spectral kinetic energy
    Ef(kx, ky, kz). To extract a one dimensional spectrum, E(k), we integrate Ef(kx,ky,kz) over
    the surface of a sphere of radius k = sqrt(kx^2 + ky^2 + kz^2). In other words
    E(k) = sum( E(kx,ky,kz), for all (kx,ky,kz) such that k = sqrt(kx^2 + ky^2 + kz^2) ).

    Parameters:
    -----------
    u: 5D array Bs x components x Nx x Ny x Nz
      The velocity field tensor.
    lx: float
      The domain size in the x-direction.
    ly: float
      The domain size in the y-direction.
    lz: float
      The domain size in the z-direction.
    smooth: boolean
      A boolean to smooth the computed spectrum for nice visualization.
    """
    nx = len(u[0,0,:, 0, 0])
    ny = len(u[0,0,0, :, 0])
    nz = len(u[0,0,0, 0, :])

    nt = nx * ny * nz
    n = max(nx, ny, nz)  # int(np.round(np.power(nt,1.0/3.0)))

    uh = fftn(u, dim = axes) / nt

    # tkeh = zeros((nx, ny, nz))

    tkeh = 0.5 * (torch.sum(uh * conj(uh), dim=1)).real

    length = max(lx, ly, lz)

    knorm = 2.0 * pi / length

    kxmax = nx / 2
    kymax = ny / 2
    kzmax = nz / 2

    wave_numbers = arange(0, n)
    tke_spectrum = torch.zeros((u.shape[0],len(wave_numbers)))
    kx = np.concatenate((np.arange(0,kxmax), abs(np.arange(-kxmax, 0))))
    ky = np.concatenate((np.arange(0,kymax), abs(np.arange(-kymax, 0))))
    kz = np.concatenate((np.arange(0,kzmax), abs(np.arange(-kzmax, 0))))

    Kx, Ky, Kz = np.meshgrid(kx, ky, kz)

    k_matrix = np.round((Kx**2 + Ky**2 + Kz**2)**0.5)

    if one_dimensional == True:
      for i in range(len(wave_numbers)):
        tke_spectrum[:,i] = tkeh[:,np.where(k_matrix == wave_numbers[i])].sum(dim=tuple(range(1,len(tkeh.shape)+1)))

      tke_spectrum = tke_spectrum / knorm
    else:
      tke_spectrum = tkeh

    knyquist = knorm * min(nx, ny, nz) / 2

    return knyquist, wave_numbers * knorm, tke_spectrum

def plot_tkespec_1d(knyquist, wavenumbers, tkespec, title):
    plt.rc("font", size=10, family='serif')

    fig = plt.figure(figsize=(4, 2.8), dpi=200, constrained_layout=False)
    l = []
    for i in range(len(tkespec)):
      l1, = plt.semilogy(wavenumbers, tkespec[i]['spec'], tkespec[i]['color'], label=tkespec[i]['label'], markersize=4, markerfacecolor='w', markevery=2)
      l.append(l1)
    #plt.axis([0.9, 100, 1e-18, 100])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.axvline(x=knyquist, linestyle='--', color='black')
    plt.xlabel('$\kappa$ (1/m)')
    plt.ylabel('$E(\kappa)$ (m$^3$/s$^2$)')
    plt.grid()
    plt.gcf().tight_layout()
    plt.title(title + str(len(wavenumbers)) + 'x' + str(len(wavenumbers)))
    plt.legend(handles=l, loc=3)
    return fig

def plot_fields(field,pred,varname,times):
    plt.rc("font", size=16, family='serif')
    rel_ae = torch.sqrt(abs(field-pred)**2/(abs(field)**2 + 1e-7))
    N = field.shape[-1]
    idx = np.random.randint(field.shape[0])
    time_steps = np.arange(10,N,(N-10)//3)
    fig, axs = plt.subplots(3,4,figsize=(20,10))
    fig.suptitle('{} - Ground truth vs. Prediction'.format(varname),fontsize=24)
    ax = axs[0,0]
    cm = ax.imshow(field[idx,...,time_steps[0]], cmap='jet')
    fig.colorbar(cm, ax = axs[0,0])
    ax.set_title('t = {:.2f} $ t / \\tau$'.format(times[time_steps[0]]/0.85))
    ax.set_ylabel('Ground truth')
    ax = axs[0,1]
    cm = ax.imshow(field[idx,...,time_steps[1]], cmap='jet')
    fig.colorbar(cm, ax = axs[0,1])
    ax.set_title('t = {:.2f} $ t / \\tau$'.format(times[time_steps[1]]/0.85))
    ax = axs[0,2]
    cm = ax.imshow(field[idx,...,time_steps[2]], cmap='jet')
    fig.colorbar(cm, ax = axs[0,2])
    ax.set_title('t = {:.2f} $ t / \\tau$'.format(times[time_steps[2]]/0.85))
    ax = axs[0,3]
    cm = ax.imshow(field[idx,...,time_steps[3]], cmap='jet')
    fig.colorbar(cm, ax = axs[0,3])
    ax.set_title('t = {:.2f} $ t / \\tau$'.format(times[time_steps[3]]/0.85))
    ax = axs[1,0]
    cm = ax.imshow(pred[idx,...,time_steps[0]], cmap='jet')
    fig.colorbar(cm, ax = axs[1,0])
    ax.set_ylabel('Prediction')
    ax = axs[1,1]
    cm = ax.imshow(pred[idx,...,time_steps[1]], cmap='jet')
    fig.colorbar(cm, ax = axs[1,1])
    ax = axs[1,2]
    cm = ax.imshow(pred[idx,...,time_steps[2]], cmap='jet')
    fig.colorbar(cm, ax = axs[1,2])
    ax = axs[1,3]
    cm = ax.imshow(pred[idx,...,time_steps[3]], cmap='jet')
    fig.colorbar(cm, ax = axs[1,3])
    ax = axs[2,0]
    cm = ax.imshow(rel_ae[idx,...,time_steps[0]], cmap='jet')
    fig.colorbar(cm, ax = axs[2,0])
    ax.set_ylabel('Error')
    ax = axs[2,1]
    cm = ax.imshow(rel_ae[idx,...,time_steps[1]], cmap='jet')
    fig.colorbar(cm, ax = axs[2,1])
    ax = axs[2,2]
    cm = ax.imshow(rel_ae[idx,...,time_steps[2]], cmap='jet')
    fig.colorbar(cm, ax = axs[2,2])
    ax = axs[2,3]
    cm = ax.imshow(rel_ae[idx,...,time_steps[3]], cmap='jet')
    fig.colorbar(cm, ax = axs[2,3])
    plt.close()
    return fig

def mae(x, y, p = 1):
    err = torch.mean(abs(x-y)**p, dim=tuple(range(2,len(x.shape)-1)))
    norm_y = torch.mean(abs(y)**p, dim=tuple(range(2,len(y.shape)-1))) + 1e-8
    loss = torch.mean(err/norm_y, dim = tuple(range(2)))
    return loss

class Postprocess:
    
    def __init__(self, dataloader, model, savepath, device, val_dir, dtype='single'):
        
        self.dataloader = dataloader
        self.model = model
        self.savepath = savepath
        if os.path.exists(self.savepath) != True:
          os.mkdir(self.savepath)
        self.device = device
        self.val_dir = find_h5_directory(val_dir)
        if not self.val_dir:
          self.val_dir =  val_dir
        if dtype == 'half':
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        
        self.results = dict()
        for dl_idx in range(len(self.dataloader.iterables)):
            self.results[dl_idx] = dict()
            self.results[dl_idx]['rmse'] = []
            self.results[dl_idx]['rms'] = dict()
            self.results[dl_idx]['rms']['real'] = 0.0
            self.results[dl_idx]['rms']['pred'] = 0.0
            self.results[dl_idx]['k'] = dict()
            self.results[dl_idx]['k']['real'] = 0.0
            self.results[dl_idx]['k']['pred'] = 0.0
            self.results[dl_idx]['mae'] = []
            self.results[dl_idx]['spectrum'] = dict()
            self.results[dl_idx]['spectrum']['real'] = []
            self.results[dl_idx]['spectrum']['pred'] = []
            self.results[dl_idx]['mean'] = dict()
            self.results[dl_idx]['mean']['real'] = []
            self.results[dl_idx]['mean']['pred'] = []
            
    def run(self, unroll_steps):
        super().__init__()
        cumsum = dict()
        cumsum['real'] = [0.0] * len(self.dataloader.iterables)
        cumsum['pred'] = [0.0] * len(self.dataloader.iterables)
        N = [0.0] * len(self.dataloader.iterables)
        
        for batch, batch_idx, dataloader_idx in self.dataloader:
            with torch.no_grad():
                y_pred = []
                y = batch[0]
                labels = batch[2]
                time = batch[-1].to(self.device)
                input = y[...,0].to(self.device)
                for i in range(y.shape[-1]-1):
                    with torch.autocast(device_type="cuda", dtype=self.dtype):
                      pred = self.model(input, time[...,i], time[...,i+1])
                    y_pred.append(pred)
                    if (i+1) % unroll_steps ==0 :
                      input = y[...,i+1].to(self.device)
                    else:
                      input = pred
                y_pred = torch.stack(y_pred, dim=-1)
                
                results = self.results[dataloader_idx]
                    
                cumsum['pred'][dataloader_idx] += torch.sum(y_pred.cpu(), dim=0)
                cumsum['real'][dataloader_idx] += torch.sum(y[...,1:], dim=0).cpu()
                #Compute rho rms
                results['rms']['pred'] += torch.sum(y_pred.std(dim=(2,3,4)).cpu(), dim=0)
                results['rms']['real'] += torch.sum(y[...,1:].std(dim=(2,3,4)).cpu(), dim=0)
                #compute kinetic energy
                results['k']['pred'] += torch.sum(0.5 * torch.sum(y_pred[:,3:].std(dim=(2,3,4)).cpu()**2, dim=1, keepdim=True), dim=0)
                results['k']['real'] += torch.sum(0.5 * torch.sum(y[:,3:,...,1:].std(dim=(2,3,4)).cpu()**2, dim=1, keepdim=True), dim=0)

                N[dataloader_idx] += y.shape[0]
                
                results['rmse'].append(nrmse_loss(y_pred.cpu(), y[...,1:]))
                results['mae'].append(mae(y_pred.cpu(), y[...,1:]))

                #calculate spectrum
                knyquist, wave_numbers, tke_spectrum = compute_tke_spectrum(y[:,3:,...,50].to(self.device),axes=(2,3,4),
                                                                            one_dimensional=True)
                results['spectrum']['real'].append(tke_spectrum.cpu())
                knyquist, wave_numbers, tke_spectrum = compute_tke_spectrum(y_pred[:,3:,...,49],axes=(2,3,4),
                                                                            one_dimensional=True)
                results['spectrum']['pred'].append(tke_spectrum.cpu())
                self.results[dataloader_idx] = results

        for dl_idx in range(len(self.dataloader.iterables)):

            results = self.results[dl_idx]
            
            results['mean']['real'] = cumsum['real'][dl_idx]/N[dl_idx]
            results['mean']['pred'] = cumsum['pred'][dl_idx]/N[dl_idx]
            results['rms']['real'] =  results['rms']['real']/N[dl_idx]
            results['rms']['pred'] =  results['rms']['pred']/N[dl_idx]
            results['k']['real'] =  results['k']['real']/N[dl_idx]
            results['k']['pred'] =  results['k']['pred']/N[dl_idx]
            results['rmse'] = torch.mean(torch.stack(results['rmse']))
            results['mae'] = torch.mean(torch.stack(results['mae']), dim=0)

            results['spectrum']['real'] = torch.mean(torch.cat(results['spectrum']['real']), dim=0)
            results['spectrum']['pred'] = torch.mean(torch.cat(results['spectrum']['pred']), dim=0)
            
            self.results[dl_idx] = results

        # PLOT
        data_ref_1 = np.loadtxt("./reference_data/spyropoulos_case1.txt", skiprows=3)
        data_ref_2 = np.loadtxt("./reference_data/spyropoulos_case2.txt", skiprows=3)
        data_ref_3 = np.loadtxt("./reference_data/spyropoulos_case3.txt", skiprows=3)

        quantities = ["density", "pressure", "temperature", "velocityX", "velocityY", "velocityZ"]
        cell_centers, cell_sizes, times, data_dict = load_data(self.val_dir, quantities)

        fig, ax = plt.subplots(figsize=(5,5))
        times = times[1:100]
        for idx in range(len(self.dataloader.iterables)):
            if idx == len(self.dataloader.iterables)-1:
                label = 'LES'
            else:
                label=None
                
            ax.plot((times-times[0])/0.85, self.results[idx]['rms']['real'][0], 'c', label=label)
            ax.plot((times-times[0])/0.85, self.results[idx]['rms']['pred'][0], '--', markevery=5, linewidth=3, 
                    label='TFNO_case{}'.format(idx+1))

        if 'HIT_LES_COMP' in self.val_dir:
          ax.plot(data_ref_1[:,0], data_ref_1[:,1], linestyle="None", marker="o", markersize=4, mfc="black", mec="black")
          ax.plot(data_ref_2[:,0], data_ref_2[:,1], linestyle="None", marker="o", markersize=4, mfc="black", mec="black")
          ax.plot(data_ref_3[:,0], data_ref_3[:,1], linestyle="None", marker="o", markersize=4, mfc="black", mec="black", label="DNS")
          ax.set_ylim([0, 0.16])
          ax.set_yticks([0, 0.05, 0.1, 0.15])
          ax.text(0.7, 0.15, "Case 1", transform=ax.transAxes, fontsize=12,
              verticalalignment='top')
          ax.text(0.7, 0.35, "Case 2", transform=ax.transAxes, fontsize=12,
                  verticalalignment='top')
          ax.text(0.7, 0.55, "Case 3", transform=ax.transAxes, fontsize=12,
                  verticalalignment='top')
        ax.set_xlabel(r"$t / \tau$")
        ax.set_ylabel(r"$\rho_{rms}$")
        ax.set_xlim([0, 5])
        ax.set_box_aspect(1)
        ax.legend()
        #ax.vlines(times[ctff[0]]/0.85,0,0.05, linestyles='dashed')
        #ax.vlines(times[ctff[1]]/0.85,0.05,0.1, linestyles='dashed')
        #ax.vlines(times[ctff[2]]/0.85,0.125,0.20, linestyles='dashed')

        plt.savefig(self.savepath+'/rho_rms.png')

        #Plot kinetic energy decay
        fig, ax = plt.subplots(figsize=(5,5))
        for idx in range(len(self.dataloader.iterables)):
                
            ax.plot((times-times[0])/0.85, self.results[idx]['k']['real'][0], '-', label='LES_case{}'.format(idx+1))
            ax.plot((times-times[0])/0.85, self.results[idx]['k']['pred'][0], '--', markevery=5, linewidth=3, 
                    label='TFNO_case{}'.format(idx+1))

        #ax.set_ylim([0, 0.16])
        #ax.set_yticks([0, 0.05, 0.1, 0.15])
        #ax.text(0.7, 0.15, "Case 1", transform=ax.transAxes, fontsize=12,
        #    verticalalignment='top')
        #ax.text(0.7, 0.35, "Case 2", transform=ax.transAxes, fontsize=12,
        #        verticalalignment='top')
        #ax.text(0.7, 0.55, "Case 3", transform=ax.transAxes, fontsize=12,
        #        verticalalignment='top')
        ax.set_xlabel(r"$t / \tau$")
        ax.set_ylabel(r"$\overline{u'u'}$")
        #ax.set_xlim([0, 5])
        #ax.set_box_aspect(1)
        ax.legend()

        plt.savefig(self.savepath+'/kinetic_energy.png')
        
        #Plot MAE
        fig, ax = plt.subplots(figsize=(5,5))
        for idx in range(len(self.dataloader.iterables)):
                
            ax.plot(times/0.85, self.results[idx]['mae'], '-', label='TFNO_case{}'.format(idx+1))
        ax.set_xlabel(r"$t / \tau$")
        ax.set_ylabel("r-MAE")
        ax.legend()
        plt.savefig(self.savepath+'/mae.png')
        MAE = 0.0
        for idx in range(len(self.dataloader.iterables)):
          MAE += self.results[idx]['mae'].mean().unsqueeze(0)
        df = pd.DataFrame({'MAE': MAE/(idx+1)})
        df.to_csv(self.savepath+'/results.csv')

        for dl_idx in range(len(self.dataloader.iterables)):
            results = self.results[dl_idx]
            E = [{'spec':  results['spectrum']['pred'], 'color': '--', 'label':'predicted'},
            {'spec':  results['spectrum']['real'], 'color': '-', 'label':'LES'}]
            fig = plot_tkespec_1d(knyquist, wave_numbers, E, 'TKE Spectrum case {} '.format(dl_idx+1))
            fig.savefig(self.savepath+'/spectrum_case_{}.png'.format(dl_idx+1))

            fields = ['density', 'pressure', 'temperature', 'u','v','w']
            for i, field in enumerate(fields):
                plots_case = plot_fields(results['mean']['real'][i],
                                          results['mean']['pred'][i],field,(times-times[0])/0.85)
                plots_case.savefig(self.savepath+"/{}_case{}.png".format(field,dl_idx+1))

