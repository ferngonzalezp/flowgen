import torch
import matplotlib.pyplot as plt
import numpy as np
import lightning as L
from flowgen import hitOfflineDataModule
from flowgen.models.VAE import SpatioTemporalVAETrainer as VAE
from flowgen.utils.scaler import FeatureScaler
from lightning.pytorch.plugins.environments import MPIEnvironment
import os
import yaml
from flowgen.utils.loss import nrmse_loss
import pandas as pd
from argparse import ArgumentParser
import umap
from flowgen.models.RevIN import RevIN

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

def plot_fields(field, pred, varname):
    plt.rc("font", size=16, family='serif')
    N = field.shape[-1]
    idx = np.random.randint(field.shape[0])
    time_steps = np.arange(N)
    fig, axs = plt.subplots(2,N,figsize=(5 * N,10))
    fig.suptitle('{} - Ground truth vs. Prediction'.format(varname),fontsize=24)
    for t in range(N):
        ax = axs[0,t]
        cm = ax.imshow(field[idx,...,time_steps[t]], cmap='jet')
        fig.colorbar(cm, ax = axs[0,t])
        ax.set_title('t = {:.2f}'.format(time_steps[t]))
        if t == 0:
            ax.set_ylabel('Ground truth')
    
        ax = axs[1,t]
        cm = ax.imshow(pred[idx,...,time_steps[t]], cmap='jet')
        fig.colorbar(cm, ax = axs[1,t])
        if t == 0:
            ax.set_ylabel('Prediction')

    return fig

def evaluate_latent_space(model, data_loader, latent_dim, save_path, scaler, val_dir):
    """
    Evaluate the latent space of a trained model.
    
    Args:
        model (torch.nn.Module): The trained VAE model.
        data_loader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        latent_dim (int): Dimensionality of the latent space.
    
    Returns:
        dict: Dictionary of evaluation metrics.
    """
    model.eval()
    
    metrics = {}
    all_mu = []
    all_logvar = []
    all_recon_losses = []
    original_data = [[]]*3
    reconstructed_data = [[]]*3
    num_samples_vis = 3
    k = []
    Ma =[]
    Mach_list = [0.2, 0.4, 0.6]
    results = dict()
    for dl_idx in range(len(data_loader.iterables)):
        results[dl_idx] = dict()
        results[dl_idx]['rho_rms'] =  dict()
        results[dl_idx]['rho_rms'] = {'orig': [], 'recon': []}
        results[dl_idx]['k_t'] = dict()
        results[dl_idx]['k_t'] = {'orig': [], 'recon': []}
    
    criterion = torch.nn.MSELoss()

    # Analyze the latent space
    with torch.no_grad():
        for batch, batch_idx, dataloader_idx in data_loader:
            k_trajectory = {'orig': [], 'recon': []}
            rho_rms_trajectory = {'orig': [], 'recon': []}
            for t in range(0,batch[0].shape[-1],5):
                data = batch[0][...,t:t+5].to(next(model.parameters()).device)  # Move to correct device
                time = batch[-1][...,t:t+5].to(next(model.parameters()).device)  # Move to correct device
                recon_data, mu, logvar = model(data, time)

                # Collect metrics
                all_mu.append(mu.cpu().numpy())
                all_logvar.append(logvar.cpu().numpy())
                all_recon_losses.append(nrmse_loss(x=recon_data, y=data, reduction=None).mean().item())
                k.append((0.5 * torch.sum(torch.mean(data[:,3:]**2, dim=(2,3,4)), dim=(1,2))/ data.shape[-1]).cpu().numpy())
                Ma.append(np.stack([Mach_list[dataloader_idx]] * data.shape[0]))

                k_trajectory['orig'].append(0.5 * torch.sum(torch.mean(data[:,3:]**2, dim=(2,3,4)), dim=1).cpu())
                k_trajectory['recon'].append(0.5 * torch.sum(torch.mean(recon_data[:,3:]**2, dim=(2,3,4)), dim=1).cpu())

                rho_rms_trajectory['orig'].append(data[:,0].std(dim=(1,2,3)).cpu())
                rho_rms_trajectory['recon'].append(recon_data[:,0].std(dim=(1,2,3)).cpu())

                #Randomly saves data snapshots
                save_data = np.random.randint(2, size=1).astype(np.bool)
                if save_data and len(original_data[dataloader_idx]) < num_samples_vis:
                    original_data[dataloader_idx].append(data.cpu().numpy())
                    reconstructed_data[dataloader_idx].append(recon_data.cpu().numpy())
    
            
            k_trajectory['orig'] = torch.cat(k_trajectory['orig'], dim=-1)
            k_trajectory['recon'] = torch.cat(k_trajectory['recon'], dim=-1)

            rho_rms_trajectory['orig'] = torch.cat(rho_rms_trajectory['orig'], dim=-1)
            rho_rms_trajectory['recon'] = torch.cat(rho_rms_trajectory['recon'], dim=-1)

            results[dataloader_idx]['k_t']['orig'].append(k_trajectory['orig'])
            results[dataloader_idx]['k_t']['recon'].append(k_trajectory['recon'])

            results[dataloader_idx]['rho_rms']['orig'].append(rho_rms_trajectory['orig'])
            results[dataloader_idx]['rho_rms']['recon'].append(rho_rms_trajectory['recon'])

            ##Randomly saves data snapshots
            ##save_data = np.random.randint(2, size=1).astype(np.bool)
            #if save_data and len(original_data) < num_samples_vis:
            #    original_data.append(data.cpu().numpy())
            #    reconstructed_data.append(recon_data.cpu().numpy())
    
    # Concatenate all batches
    all_mu = np.concatenate(all_mu, axis=0)
    all_logvar = np.concatenate(all_logvar, axis=0)
    k = np.concatenate(k, axis=0)
    Ma = np.concatenate(Ma, axis=0)

    for idx in range(len(data_loader.iterables)):
        results[idx]['k_t']['orig'] = torch.mean(torch.cat(results[idx]['k_t']['orig'], dim=0), dim=0)
        results[idx]['k_t']['recon'] = torch.mean(torch.cat(results[idx]['k_t']['recon'], dim=0), dim=0)

        results[idx]['rho_rms']['orig'] = torch.mean(torch.cat(results[idx]['rho_rms']['orig'], dim=0), dim=0)
        results[idx]['rho_rms']['recon'] = torch.mean(torch.cat(results[idx]['rho_rms']['recon'], dim=0), dim=0)


    original_data = np.concatenate([np.concatenate(original_data[i]) for i in range(len(original_data))])
    reconstructed_data = np.concatenate([np.concatenate(reconstructed_data[i]) for i in range(len(reconstructed_data))])

    # Variance metrics
    mu_variance = np.var(all_mu, axis=0).mean()  # Average variance of latent means
    logvar_mean = np.mean(all_logvar)           # Average log variance
    logvar_variance = np.var(all_logvar)        # Variance of log variance

    # Reconstruction quality
    avg_reconstruction_loss = np.mean(all_recon_losses)
    #avg_distance = compute_latent_diversity(all_mu)
    
    # Metrics summary
    metrics['mu_variance'] = mu_variance
    metrics['logvar_mean'] = logvar_mean
    metrics['logvar_variance'] = logvar_variance
    metrics['reconstruction_loss'] = avg_reconstruction_loss
    #metrics['avg_distance'] = avg_distance
    
    print(f"Latent Space Metrics:\n"
          f"  - Mean Variance: {mu_variance:.4f}\n"
          f"  - Logvar Mean: {logvar_mean:.4f}\n"
          f"  - Logvar Variance: {logvar_variance:.4f}\n"
          f"  - Avg Reconstruction Loss: {avg_reconstruction_loss:.4f}\n"
          #f"  - Avg pariwise distance: {avg_distance:.4f}\n"
          )
    
    # Visualization
    visualize_latent_space(all_mu, latent_dim, save_path, k, Ma)
    visualize_reconstructed(original_data, reconstructed_data, save_path)

    data_ref_1 = np.loadtxt("/scratch/cfd/gonzalez/HIT_LES_COMP/reference_data/spyropoulos_case1.txt", skiprows=3)
    data_ref_2 = np.loadtxt("/scratch/cfd/gonzalez/HIT_LES_COMP/reference_data/spyropoulos_case2.txt", skiprows=3)
    data_ref_3 = np.loadtxt("/scratch/cfd/gonzalez/HIT_LES_COMP/reference_data/spyropoulos_case3.txt", skiprows=3)
    quantities = ["density", "pressure", "temperature", "velocityX", "velocityY", "velocityZ"]

    fig, ax = plt.subplots(figsize=(5,5))
    times = np.linspace(start=0,stop=6,num=100)
    for idx in range(len(data_loader.iterables)):
        if idx == len(data_loader.iterables)-1:
                label = 'LES'
        else:
                label=None
        ax.plot(times, results[idx]['rho_rms']['orig'], 'c', label=label)
        ax.plot(times, results[idx]['rho_rms']['recon'], '--', markevery=5, linewidth=3, 
        label='VAE_case{}'.format(idx+1))
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
    plt.savefig(os.path.join(save_path, 'rho_rms.png'))

    #Plot kinetic energy decay
    fig, ax = plt.subplots(figsize=(5,5))
    for dl_idx in range(len(data_loader.iterables)):        
        ax.plot(times, results[dl_idx]['k_t']['orig'], '-', label='LES_case{}'.format(dl_idx+1))
        ax.plot(times, results[dl_idx]['k_t']['recon'], '--', markevery=5, linewidth=3, 
                    label='VAE_case{}'.format(dl_idx+1))

    ax.set_xlabel(r"$t / \tau$")
    ax.set_ylabel(r"$\overline{u'u'}$")

    ax.legend()

    plt.savefig(os.path.join(save_path, 'kinetic_energy.png'))
        
    return metrics

def visualize_latent_space(mu, latent_dim, save_path, energy, Ma, num_samples=1000):
    """
    Visualize the latent space using a scatter plot or pairwise dimensions.

    Args:
        mu (np.ndarray): Latent means of shape (N, latent_dim).
        latent_dim (int): Dimensionality of the latent space.
        num_samples (int): Number of samples to visualize.
    """
    if latent_dim > 2:
        print("Latent space has more than 2 dimensions; visualizing 2 dimensional embedding.")
        reducer = umap.UMAP()
        mu = np.sum(mu, axis=1)
        embedding = reducer.fit_transform(mu)
        print(embedding.shape)
    samples = embedding[:num_samples]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc1 = axes[0].scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=5, c=energy, cmap='jet')
    fig.colorbar(sc1, ax=axes[0], label="Kinetic Energy")
    axes[0].set_title("Latent Space Visualization")
    axes[0].set_xlabel("Latent embedding 1")
    axes[0].set_ylabel("Latent embedding 2")
    axes[0].grid()
    sc2 = axes[1].scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=5, c=Ma, cmap='tab10')
    fig.colorbar(sc2, ax=axes[1], label="Initial Ma")
    axes[1].set_title("Latent Space Visualization")
    axes[1].set_xlabel("Latent embedding 1")
    axes[1].set_ylabel("Latent embedding 2")
    axes[1].grid()
    plt.savefig(os.path.join(save_path,'latent_space.png'))
    plt.close()

def visualize_reconstructed(original_data, reconstructed_data, save_path):
    quantities = ["density", "pressure", "temperature", "velocityX", "velocityY", "velocityZ"]
    for b in range(original_data.shape[0]):
        for i, varname in enumerate(quantities):
             fig = plot_fields(original_data[b,i], reconstructed_data[b, i], varname)
             plt.savefig(os.path.join(save_path , f'{varname}_{b}.png'))
             plt.close()

def compute_latent_diversity(mu):
    """
    Compute diversity in the latent space by calculating pairwise distances.

    Args:
        mu (np.ndarray): Latent means of shape (N, latent_dim).

    Returns:
        float: Average pairwise distance.
    """
    from scipy.spatial.distance import pdist, squareform
    latent_dim = mu.shape[-1]
    mu = mu.reshape(-1, latent_dim)
    pairwise_distances = pdist(mu, metric='euclidean')
    avg_distance = np.mean(pairwise_distances)
    print(f"Average Pairwise Distance in Latent Space: {avg_distance:.4f}")
    return avg_distance

def main(args):
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    data_path = args.data_path #Path to dataset location

    train_data_dirs = [data_path+f"train/{case_name}" for case_name in args.cases]
    val_dir = [data_path+f"val/{case_name}" for case_name in args.cases]
    dm = hitOfflineDataModule(val_dirs = val_dir, data_dir = train_data_dirs, batch_size =  1, seq_len=args.seq_len)

    with open(args.vae_params, 'r') as file:
        vae_params = yaml.safe_load(file)

    vae_params.update(dict(seq_len=args.seq_len[0]))
    
    if vae_params['scaler_type'] == 'iterative':
        scaler = FeatureScaler(vae_params['input_dim'])
        scaler = torch.load("HIT_scaler_state.pt")
    else:
        scaler = RevIN(vae_params['input_dim'], affine=False)

    if args.save_path:
         
         
         if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        
         case_name_folder = 'VAE'
         create_directory = True
         i = 1
         save_path = os.path.join(args.save_path, case_name_folder)
         while create_directory:
            if os.path.exists(save_path):
                case_name_folder_new = case_name_folder + "-%d" % i
                save_path = os.path.join(args.save_path, case_name_folder_new)
                i += 1
            else:
                create_directory   = False
         os.mkdir(save_path)
        
    else:
        save_path = os.getcwd()
    
    model = VAE.load_from_checkpoint(args.ckpt_path, 
                                    vae_params=vae_params, 
                                    scaler=scaler, 
                                    learning_rate=1e-3, 
                                    scaler_state_path="HIT_scaler_state.pt", 
                                    load_scaler_state=True)
    model.to(device)

    dm.setup(stage='fit')

    metrics = evaluate_latent_space(model, dm.val_dataloader(), vae_params['latent_dim'], save_path, scaler, val_dir[0])
    
    # Save metrics to CSV
    # Convert metrics dictionary to DataFrame
    metrics_df = pd.DataFrame([metrics])
    
    # Create metrics filename
    metrics_filename = os.path.join(save_path, 'latent_space_metrics.csv')
    
    # Save to CSV
    metrics_df.to_csv(metrics_filename, index=False)
    print(f"Metrics saved to: {metrics_filename}")

if __name__ == "__main__":
     parser = ArgumentParser()
     parser.add_argument("--data_path", type=str)
     parser.add_argument("--vae_params", type=str)
     parser.add_argument("--ckpt_path", type=str)
     parser.add_argument("--seq_len", type=int, nargs='+', default=[5, 5])
     parser.add_argument("--cases", type=str, nargs='+', default=["case1", "case2", "case3"])
     parser.add_argument('--save_path', type=str, default=None)
     args = parser.parse_args()
     main(args)