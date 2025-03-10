import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
import lightning as L
import torch.nn.functional as F
from torch.optim import Adam
from flowgen.models.VAE.embedding import get_1d_sincos_pos_embed_from_grid, RotaryPositionEmbeddingPytorchV2, RotaryPositionEmbeddingPytorchV1
from flowgen.utils.loss import nrmse_loss, spec_loss, state_reg, corr_loss, temp_diff_loss
from flowgen.models.UNETs.block import ConvNextBlockBlock, LayerNorm
from flowgen.models.LRU import LRU
from flowgen.models.VAE.perceiver import PerceiverEncoder, PerceiverDecoder
import os
import matplotlib.pyplot as plt
import numpy as np
import math

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

class hMLP_output(nn.Module):
    """ Patch to Image De-bedding
    """
    def __init__(self, patch_size=8, out_chans=6, embed_dim=768):
        super().__init__()
        if isinstance(patch_size, (list, tuple)):
          self.patch_size = min(patch_size)
        else:
          self.patch_size = patch_size
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        def f(x):
            return 1 if x == 0 else 4
        self.upsampling_blocks = nn.ModuleList([torch.nn.Sequential(
                                    nn.ConvTranspose3d(embed_dim//f(i), embed_dim//4, kernel_size=6, stride=2, padding=2),
                                    LayerNorm(embed_dim//4, eps=1e-6, data_format="channels_first"),
                                    nn.GELU()) for i in range(int(math.log2(self.patch_size)) - 1)]
                                    )
        self.out_proj = nn.ConvTranspose3d(embed_dim//4, out_chans, kernel_size=2, stride=2)

    
    def forward(self, x):
        for module in self.upsampling_blocks:
            x = module(x)
        x = self.out_proj(x)
        return x

class hMLP_stem(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=8, in_chans=6, embed_dim =768):
        super().__init__()
        if isinstance(patch_size, (list, tuple)):
          self.patch_size = min(patch_size)
        else:
          self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        def f(x):
            return in_chans if x == 0 else embed_dim//4
        self.downsampling_blocks = nn.ModuleList([torch.nn.Sequential(
                                    nn.Conv3d(f(i), embed_dim//4, kernel_size=6, stride=2, padding=2),
                                    LayerNorm(embed_dim//4, eps=1e-6, data_format="channels_first"),
                                    nn.GELU()) for i in range(int(math.log2(self.patch_size)) - 1)]
                                    )
        self.out_proj = nn.Conv3d(embed_dim//4, embed_dim, kernel_size=2, stride=2)

    def forward(self, x):
        for module in self.downsampling_blocks:
            x = module(x)
        x = self.out_proj(x)
        return rearrange(x, 'b c w h d -> b (w h d) c') # (B, num_patches, embed_dim)

class CausalDownsamplingLRU(nn.Module):
    def __init__(self, in_channels, out_channels, state_features, temporal_downsampling):
        super().__init__()
        self.lru_block = LRU(in_features=in_channels, out_features=out_channels, state_features=state_features)
        self.temporal_downsampling = temporal_downsampling
    def forward(self, x):
        x = self.lru_block(x)
        return x[:,-self.temporal_downsampling:]

class CausalUpsamplingLRU(nn.Module):
    def __init__(self, in_channels, out_channels, state_features, out_seq):
        super().__init__()
        self.lru_block = LRU(in_features=in_channels, out_features=out_channels, state_features=state_features)
        self.out_seq = out_seq
    def forward(self, x):
        out = []
        for i in range(self.out_seq):
            if i ==0:
                state = None
            else:
                state = self.lru_block.state
            x = self.lru_block(x, state)
            out.append(x[:,-1:])
        out = torch.cat(out, dim=-2)
        return out

def generate_causal_mask(batch_size: int = None, seq_len: int = None, num_heads: int = None, device: str = "cpu") -> torch.Tensor:
    """
    Generate a causal mask for self-attention.
    
    Args:
        batch_size (int, optional): Batch size N. If provided with num_heads, creates a 3D mask
        seq_len (int): Length of the sequence (L and S are the same for self-attention)
        num_heads (int, optional): Number of attention heads. Required if batch_size is provided
        device (str): Device to place the mask on ("cpu" or "cuda")
    
    Returns:
        torch.Tensor: Causal attention mask of shape (L, S) or (N*num_heads, L, S)
    """
    # Create base causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    
    # Convert to boolean mask where True means that position will be masked (prevented from attending)
    mask = mask.bool()
    
    # If batch_size and num_heads are provided, create 3D mask
    if batch_size is not None and num_heads is not None:
        mask = mask.unsqueeze(0).expand(batch_size * num_heads, seq_len, seq_len)
    
    return mask.to(device)

def visualize_reconstructed(original_data, reconstructed_data, save_path):
    quantities = ["density", "pressure", "temperature", "velocityX", "velocityY", "velocityZ"]
    b = 0
    while b < orignal_data.shape[0]:    
        for i, varname in enumerate(quantities):
             fig = plot_fields(original_data[b,i], reconstructed_data[b, i], varname)
             plt.savefig(os.path.join(os.path.join(save_path, 'snapshots') , f'{varname}_{b}.png'))
             plt.close()
        b += 1
        if b >= 1:
            break

class CausalDownsamplingConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, temporal_downsampling):
        super().__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.padding = kernel_size - 2  # Ensure causality
        self.ds_layers = nn.ModuleList([nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size),
                                                      LayerNorm(self.out_channels, eps=1e-6, data_format="channels_first"),
                                                      nn.GELU()) for _ in range(temporal_downsampling-1)])
        self.temporal_downsampling = temporal_downsampling
    def forward(self, x):
        for i in range(self.temporal_downsampling-1):
            x = F.pad(x, (self.padding, 0), 'replicate')
            x = self.ds_layers[i](x)
        return x

class CausalUpsamplingConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, temporal_upsampling):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1  # Ensure causality
        self.us_layers = nn.ModuleList([nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size),
                                                      LayerNorm(out_channels, eps=1e-6, data_format="channels_first"),
                                                      nn.GELU()) for _ in range(temporal_upsampling)])
        self.temporal_upsampling = temporal_upsampling
    
    def forward(self, x):
      out = []
      for i in range(self.temporal_upsampling):
        x = F.pad(x, (max(self.padding-i, 0), 0), 'replicate')
        new_x = x[...,-1:] + self.us_layers[i](x)
        out.append(new_x)
        x = torch.cat([x[...,-i-1:], new_x], dim=-1)
      return torch.cat(out, dim=-1)

# Helper: Patch Embedding for 3D Volumes
class PatchEmbedding3D(nn.Module):
    def __init__(self, input_dim, embed_dim, patch_size, overlap):
        super().__init__()
        stride = [math.floor(patch_dim * overlap) for patch_dim in patch_size]
        self.proj = nn.Conv3d(input_dim, embed_dim, kernel_size=patch_size, stride=stride)
    
    def forward(self, x):
        # x: (B, C, W, H, D)
        x = self.proj(x)  # (B, embed_dim, W//patch, H//patch, D//patch)
        return rearrange(x, 'b c w h d -> b (w h d) c') # (B, num_patches, embed_dim)

# Temporal Patch Embedding
class TemporalPatchEmbedding(nn.Module):
    def __init__(self, embed_dim, time_downsample):
        super().__init__()
        #self.proj = nn.Conv1d(embed_dim, embed_dim, kernel_size=time_downsample, stride=time_downsample)
        self.proj = CausalDownsamplingConv(embed_dim, embed_dim, kernel_size=time_downsample, 
                                            temporal_downsampling=time_downsample)
        #self.proj = CausalDownsamplingLRU(embed_dim, embed_dim, embed_dim*2, time_downsample)
    def forward(self, x):
        # x: (B * num_patches, T, embed_dim)
        x = rearrange(x, 'b t d -> b d t') # (B*num_patches, embed_dim, T)
        x = self.proj(x)  # (B*num_patches, embed_dim, T//time_downsample)
        return x

# Spatial ViT Block
class SpatialViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, latent_shape = [8,8,8]):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.1)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.norm1 = nn.RMSNorm(embed_dim)
        self.norm2 = nn.RMSNorm(embed_dim)
        self.embedding = RotaryPositionEmbeddingPytorchV1(
                        dim = embed_dim//num_heads,
                        rope_dim= "3D",
                        latent_shape = latent_shape,
        )
        self.num_heads = num_heads

    def forward(self, x):
        # Self-attention
        B, L, D = x.shape
        q, k = self.norm1(x), self.norm1(x)
        v = self.norm1(x)

        q = rearrange(q, 'b s (h d) -> b s h d', h=self.num_heads, d=D//self.num_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.num_heads, d=D//self.num_heads)

        q, k = self.embedding(q, k, seq_len = L)

        q = rearrange(q, 'b s h d -> b s (h d)')
        k = rearrange(k, 'b s h d -> b s (h d)')

        x = x + self.attn(q, k, v)[0]
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x

# Temporal ViT Block
class TemporalViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, is_causal: bool = False):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.norm = torch.nn.functional.layer_norm

        self.embedding = RotaryPositionEmbeddingPytorchV1(
                        dim = embed_dim//num_heads,
                        rope_dim= "1D",
                        max_position_embeddings = 5,
        )

        self.num_heads = num_heads
        self.is_causal = is_causal

    def forward(self, x):
        # Self-attention across time
        B, L, D = x.shape
        latent_dim = x.shape[-1]

        q, k = self.norm(x, [latent_dim]), self.norm(x, [latent_dim])
        v = self.norm(x, [latent_dim])

        q = rearrange(q, 'b s (h d) -> b s h d', h=self.num_heads, d=D//self.num_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.num_heads, d=D//self.num_heads)

        q, k = self.embedding(q, k, seq_len = L)

        q = rearrange(q, 'b s h d -> b s (h d)')
        k = rearrange(k, 'b s h d -> b s (h d)')
        if self.is_causal:
            mask = generate_causal_mask(batch_size=B, 
                                        seq_len=L, 
                                        num_heads=self.num_heads, 
                                        device=x.device)
        else:
            mask=None
            
        x = x + self.attn(q, k, v, attn_mask=mask, is_causal=self.is_causal)[0]
        # MLP
        x = x + self.mlp(self.norm(x, [latent_dim]))
        return x

# Encoder
class ViTEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, patch_size, spatial_heads, temporal_heads, mlp_dim, 
    num_spatial_blocks, num_temporal_blocks, time_downsample, overlap=1, latent_shape=[8,8,8], is_causal: bool = False,
    temporal_encoder_type: str = 'hybrid'):
        super().__init__()
        #self.patch_embed = PatchEmbedding3D(input_dim, embed_dim, patch_size, overlap)
        self.patch_embed = hMLP_stem(patch_size = patch_size, in_chans=input_dim, embed_dim =embed_dim)
        self.spatial_blocks = nn.ModuleList([SpatialViTBlock(embed_dim, spatial_heads, mlp_dim, latent_shape) for _ in range(num_spatial_blocks)])
        assert temporal_encoder_type in ['hybrid', 'perceiver']
        if temporal_encoder_type == 'hybrid':
            self.temporal_downsample = TemporalPatchEmbedding(embed_dim, time_downsample)
            self.temporal_blocks = nn.ModuleList([TemporalViTBlock(embed_dim, temporal_heads, mlp_dim, is_causal=is_causal) for _ in range(num_temporal_blocks)])
        elif temporal_encoder_type == 'perceiver':
            self.temporal_blocks = PerceiverEncoder(embed_dim, mlp_dim, time_downsample, 
                                    num_heads=temporal_heads, num_layers=num_temporal_blocks)
        self.embed_dim = embed_dim
        self.temporal_encoder_type = temporal_encoder_type

    def forward(self, x, time):
        # Input shape: (B, C, W, H, D, T)
        B, C, W, H, D, T = x.shape
        #spatial_embeddings = []
        t_pos = []
        # Process each time step independently
        spatial_embed = [self.patch_embed(x[:, :, :, :, :, t]) for t in range(T)]  # (B, num_patches, embed_dim)
        for t in range(T):
            for block in self.spatial_blocks:
                spatial_embed[t] = block(spatial_embed[t])
        for b in range(B):
            t_pos.append(torch.tensor(get_1d_sincos_pos_embed_from_grid(self.embed_dim, time[b].cpu())))
        
        spatial_embeddings = torch.stack(spatial_embed, dim=1)  # (B, T, num_patches, embed_dim)
        
        B, T, num_patches, embed_dim = spatial_embeddings.shape
        # Temporal Attention
        t_pos = torch.stack(t_pos).type_as(spatial_embeddings)
        spatial_embeddings = spatial_embeddings + t_pos[:, :, None, :]
        temporal_embeddings = rearrange(spatial_embeddings, 'b t l d -> (b l) t d')

        if self.temporal_encoder_type == 'hybrid':
            for block in self.temporal_blocks:
                temporal_embeddings = block(temporal_embeddings)
            
            # Temporal Downsampling
            temporal_embeddings = self.temporal_downsample(temporal_embeddings)  # (B * num_patches, T//time_downsample, embed_dim)
            
            temporal_embeddings = rearrange(temporal_embeddings, '(b l) d t -> b l t d', b=B, l=num_patches)
        
        elif self.temporal_encoder_type == 'perceiver':
            temporal_embeddings = self.temporal_blocks(temporal_embeddings)
            temporal_embeddings = rearrange(temporal_embeddings, '(b l) t d -> b l t d', b=B, l=num_patches)

        return temporal_embeddings

        
class ViTDecoder(nn.Module):
    def __init__(self, latent_dim, embed_dim, out_dim, patch_size, spatial_heads, temporal_heads, mlp_dim, 
                 num_spatial_blocks, num_temporal_blocks, seq_len, overlap=1, latent_shape=[8,8,8], is_causal: bool = False,
                 temporal_decoder_type: str = 'hybrid'):
        super().__init__()
        stride = [math.floor(patch_dim * overlap) for patch_dim in patch_size]
        self.latent_to_time = nn.Linear(latent_dim, embed_dim)  # Expand latent to time dimension

        assert temporal_decoder_type in ['hybrid', 'perceiver']
        if temporal_decoder_type == 'hybrid':
            self.temporal_blocks = nn.ModuleList([TemporalViTBlock(embed_dim, temporal_heads, mlp_dim, is_causal=is_causal) 
                                                for _ in range(num_temporal_blocks)])
            #self.temporal_upsample = nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=time_downsample, stride=time_downsample)
            self.temporal_upsample = CausalUpsamplingConv(embed_dim, embed_dim, kernel_size=seq_len,
                                                            temporal_upsampling=seq_len)
        elif temporal_decoder_type == 'perceiver':
            self.temporal_blocks = PerceiverDecoder(embed_dim, seq_len, num_heads=temporal_heads, 
                                                    num_layers=num_temporal_blocks, mlp_dim=mlp_dim)

        #self.temporal_upsample = CausalUpsamplingLRU(embed_dim, embed_dim, embed_dim*2, seq_len)
        self.spatial_blocks = nn.ModuleList([SpatialViTBlock(embed_dim, spatial_heads, mlp_dim, latent_shape) 
                                             for _ in range(num_spatial_blocks)])
        #self.patch_unembed = nn.ConvTranspose3d(embed_dim, out_dim, kernel_size=patch_size, stride=stride)
        self.patch_unembed = hMLP_output(patch_size = patch_size, out_chans=out_dim, embed_dim =embed_dim)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.temporal_decoder_type = temporal_decoder_type

    def forward(self, z, time):
        # z: (B, num_patches, T//t_downsample, latent_dim)
        B, num_patches, T, latent_dim = z.shape
        temporal_embeddings = self.latent_to_time(z) #.flatten(0,1).permute(0, 2, 1)  # (B * num_patches, embed_dim, T//t_downsample)
        if self.temporal_decoder_type == 'hybrid':
            temporal_embeddings = rearrange(temporal_embeddings, 'b l t d -> (b l) d t')
            temporal_embeddings = self.temporal_upsample(temporal_embeddings) #.reshape(B * num_patches, -1, self.embed_dim)  # (B *num_patches, T, embed_dim)
            temporal_embeddings = rearrange(temporal_embeddings, 'bl d t -> bl t d')
            # Temporal processing
            for block in self.temporal_blocks:
                temporal_embeddings = block(temporal_embeddings)
        elif self.temporal_decoder_type == 'perceiver':
            temporal_embeddings = rearrange(temporal_embeddings, 'b l t d -> (b l) t d')
            temporal_embeddings = self.temporal_blocks(temporal_embeddings, time)
        temporal_embeddings = rearrange(temporal_embeddings, '(b l) t d -> b l d t', b=B, l=num_patches)

        # Spatial processing
        #spatial_embeddings = []
        B, num_patches, embed_dim, T = temporal_embeddings.shape
        spatial_embedding = [temporal_embeddings[..., t] for t in range(T)]  # (B, num_patches, embed_dim)
        for t in range(T):
            #spatial_embedding = temporal_embeddings[..., t]  # (B, num_patches, embed_dim)
            for block in self.spatial_blocks:
                spatial_embedding[t] = block(spatial_embedding[t])
            spatial_embedding[t] = rearrange(spatial_embedding[t], 'b (h w d) c -> b c h w d', 
                                    h=round(num_patches**(1/3)), w= round(num_patches**(1/3)), 
                                    d=round(num_patches**(1/3)))
            spatial_embedding[t] = self.patch_unembed(spatial_embedding[t])
            #spatial_embeddings.append(self.patch_unembed(spatial_embedding))
        
        # Reconstruct spatial and temporal dimensions
        reconstructed = torch.stack(spatial_embedding, dim=-1)  # (B, C, W, H, D, T)

        return reconstructed

class VAE4D(nn.Module):
    def __init__(self, input_dim, out_dim, embed_dim, patch_size, spatial_heads, temporal_heads, mlp_dim, 
                 num_spatial_blocks, num_temporal_blocks, time_downsample, overlap, latent_dim, seq_len, temporal_block_type, variational=True):
        super().__init__()
        self.encoder = ViTEncoder(
            input_dim, embed_dim, patch_size, spatial_heads['encoder'], temporal_heads['encoder'], mlp_dim, 
            num_spatial_blocks, num_temporal_blocks, time_downsample, overlap, is_causal=True, 
            temporal_encoder_type=temporal_block_type
        )
        self.variational = variational
        self.fc_mu = nn.Linear(embed_dim, latent_dim)  # Mean vector
        if self.variational:
            self.fc_logvar = nn.Linear(embed_dim, latent_dim)  # Log variance vector
        self.latent_to_time = nn.Linear(latent_dim, embed_dim)  # Map latent back to temporal dimension
        

        self.decoder = ViTDecoder(
            latent_dim=latent_dim, 
            out_dim = out_dim,
            embed_dim=embed_dim,
            patch_size=patch_size,
            spatial_heads=spatial_heads['decoder'],
            temporal_heads=temporal_heads['decoder'],
            mlp_dim=mlp_dim,
            num_spatial_blocks=num_spatial_blocks,
            num_temporal_blocks=num_temporal_blocks,
            seq_len=seq_len,
            overlap=overlap,
            is_causal=True,
            temporal_decoder_type = temporal_block_type
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier normal initialization and zero out biases."""
        def init_weights(m):
            if isinstance(m, (nn.Linear, nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                # Initialize the input projection weights
                nn.init.xavier_normal_(m.in_proj_weight)
                # Initialize the output projection weights
                nn.init.xavier_normal_(m.out_proj.weight)
                # Zero out all biases
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                if m.out_proj.bias is not None:
                    nn.init.zeros_(m.out_proj.bias)

        self.apply(init_weights)
        
        # Initialize fc_logvar biases to -5
        #if self.fc_logvar.bias is not None:
        #    nn.init.constant_(self.fc_logvar.bias, -5.0)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.shape, device=std.device, dtype=std.dtype)
        return mu + eps * std

    def forward(self, x, t):
        # Encode
        latent_vector = self.encoder(x, t)  # (B, num_patches, T//time_downsample, embed_dim)
        B, num_patches, T, embed_dim = latent_vector.shape
        latent_vector =  rearrange(latent_vector, 'b l t d -> b (l t) d')
        mu = self.fc_mu(latent_vector)  # (B, latent_dim)
        if self.variational:
            logvar = self.fc_logvar(latent_vector)  # (B, latent_dim)
            # Sample from latent space
            z = self.reparameterize(mu, logvar)  # (B, latent_dim)
        else:
            logvar = -1 * torch.empty_like(mu)
            z = mu
        # Decode
        z = rearrange(z, 'b (l t) d -> b l t d', l=num_patches, t=T)
        reconstructed = self.decoder(z, t)  # (B, C, W, H, D, T)

        return reconstructed, mu, logvar
        
def calculate_kl_divergence(mu, logvar, free_bits=0.0):
    """
    Compute KL divergence for a latent space with shape (Batch, Len, dim).
    
    Args:
        mu (torch.Tensor): Mean of the posterior, shape (Batch, Len, dim).
        logvar (torch.Tensor): Log variance of the posterior, shape (Batch, Len, dim).
    
    Returns:
        torch.Tensor: Scalar KL divergence, averaged over batch and sequence.
    """
    
    # Compute KL divergence element-wise
    kl = -0.5 * (-logvar.exp() - mu.pow(2) + 1 + logvar)
    
    kl = torch.clamp(kl, min=free_bits)

    # Sum over latent dimensions
    kl_latent = kl.sum(dim=-1)  # Sum over dim

    
    # Mean over batch and sequence dimensions
    kl_mean = kl_latent.mean()  # Average over Batch and Len
    
    return kl_mean

class SpatioTemporalVAETrainer(L.LightningModule):
    def __init__(self, vae_params, scaler, learning_rate=1e-3, scaler_state_path="scaler_state.pt", load_scaler_state=True):
        super().__init__()

        if 'overlap' in vae_params.keys():
            overlap = vae_params['overlap']
        else:
            overlap = 1

        self.vae = VAE4D(
                        input_dim = vae_params['input_dim'], 
                        out_dim = vae_params['out_dim'], 
                        embed_dim = vae_params['embed_dim'], 
                        patch_size = vae_params['patch_size'], 
                        spatial_heads = vae_params['spatial_heads'], 
                        temporal_heads = vae_params['temporal_heads'], 
                        mlp_dim = vae_params['mlp_dim'], 
                        num_spatial_blocks = vae_params['num_spatial_blocks'], 
                        num_temporal_blocks = vae_params['num_temporal_blocks'], 
                        time_downsample = vae_params['time_downsample'], 
                        latent_dim = vae_params['latent_dim'],
                        overlap = overlap,
                        seq_len = vae_params['seq_len'],
                        temporal_block_type = vae_params['temporal_block_type'],
                        variational = vae_params['variational'],
        )
        self.learning_rate = learning_rate
        self.scaler = scaler
        self.scaler_type = vae_params['scaler_type']
        self.scaler_state_path = scaler_state_path
        self.load_scaler_state = load_scaler_state  # Flag to enable/disable loading scaler state
        self.val_loss_avg = [0.0]
        self.beta = 1.0
        self.variational = vae_params['variational']
        if 'beta' in vae_params.keys():
            self.beta = vae_params['beta']
        self.fine_tune_recon = False
        if 'fine_tune_recon' in vae_params.keys():
            self.fine_tune_recon = True
            self.vae.encoder.eval()
            self.vae.fc_logvar.eval()
            self.vae.fc_mu.eval()
            self.vae.latent_to_time.eval()
        
        self.save_hyperparameters()
    
    def forward(self, x, t):
        # Forward pass through the VAE
        input = x.clone()
        if self.scaler_type == 'revin':
            for i in range(input.shape[-1]):
                input[...,i] = self.scaler(input[...,i], i,'norm')
        else:
            input = self.scaler(input)

        reconstructed, mu, logvar = self.vae(input, t)

        if self.scaler_type == 'revin':
            for i in range(reconstructed.shape[-1]):
                reconstructed[...,i] = self.scaler(reconstructed[...,i], i, 'denorm')
            self.scaler.reinit_stats()
        else: 
            reconstructed = self.scaler.denorm(reconstructed)

        return reconstructed, mu, logvar

    def vae_loss_function(self, reconstructed, original, mu, logvar, beta=0.0, free_bits=0.0):

        def annealed_lambda_dynamic(recon_loss, recon_loss_init, lambda_max=0.1, beta=10):
            delta = recon_loss_init - recon_loss  # Improvement in recon loss
            if recon_loss > recon_loss_init:
                return 0.0
            else:
                return lambda_max * (1 - math.exp(-beta * delta))

        # Reconstruction loss (MSE)
        #recon_loss = F.mse_loss(reconstructed, original, reduction="mean")
        recon_loss = nrmse_loss(reconstructed, original, w = [1,1,1,1,1,1], reduction=True)
       # recon_loss = nrmse_loss(reconstructed, original, reduction=True)
        t_loss = temp_diff_loss(reconstructed, original)
        # spectrum loss
        Spec_loss = 0.25 * spec_loss(reconstructed[:,3:], original[:,3:], N_bins=4)
        for i in range(3):
            Spec_loss += 0.25 * spec_loss(reconstructed[:,i:i+1], original[:,i:i+1], N_bins=4)
        # KL Divergence loss
        kl_loss = 0.0
        if self.variational:
            kl_loss = calculate_kl_divergence(mu, logvar, free_bits=free_bits)
        w_spec = annealed_lambda_dynamic(recon_loss.item(), 0.1, 0.01, beta=20)
        rmse = nrmse_loss(reconstructed, original, reduction=False).mean().item()
        return recon_loss + beta * kl_loss + 0.1 * Spec_loss + 1.0 * t_loss, (recon_loss, kl_loss, t_loss, Spec_loss, rmse)

    def ft_loss_function(self, reconstructed, original):
        # Reconstruction loss (MSE)
        recon_loss = nrmse_loss(reconstructed, original, reduction=True)
        # KL Divergence loss
        spectrum_loss = spec_loss(reconstructed, original)
        #state_regularization = state_reg(reconstructed)
        t_loss = temp_diff_loss(reconstructed, original)
        aux = dict(spectrum_loss = spectrum_loss,
                   #state_regularization = state_regularization,
                   t_loss = t_loss,
                   recon_loss =  recon_loss)
        return recon_loss + spectrum_loss + correlation_loss, aux

    def training_step(self, batch, batch_idx):
        time = batch[-1]
        input = batch[0]
        reconstructed, mu, logvar = self(input, time)
        beta_target = self.beta
        beta = min(beta_target, beta_target * self.trainer.current_epoch / 1)
        if self.fine_tune_recon:
            loss, aux = self.ft_loss_function(reconstructed, input)
            self.log("train_loss", loss.item(), prog_bar=True)
            self.log("train_recon_loss", aux['recon_loss'].item(), prog_bar=True)
            self.log("spectrum_loss", aux['spectrum_loss'].item(), prog_bar=True)
            #self.log("state_regularization", aux['state_regularization'].item(), prog_bar=True)
            self.log("temporal_loss", aux['temporal_loss'].item(), prog_bar=True)
            self.log("latent_mu_var", mu.var().item(), prog_bar=True)
            self.log("latent_logvar_mean", logvar.mean().item(), prog_bar=True)
        else:
            loss, aux = self.vae_loss_function(reconstructed, input, mu, logvar, beta=beta, free_bits=0.0)
            self.log("train_loss", loss, prog_bar=True)
            self.log("train_recon_loss", aux[0], prog_bar=True)
            self.log("nrmse", aux[-1], prog_bar=True)
            self.log("spec_loss", aux[-2].item(), prog_bar=True)
            self.log("temporal_loss", aux[-3].item(), prog_bar=True)
            self.log("train_kl_loss", aux[1], prog_bar=True)
            self.log("latent_mu_var", mu.var().item(), prog_bar=True)
            self.log("latent_logvar_mean", logvar.mean().item(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        time = batch[-1]
        input =  batch[0]
        reconstructed, mu, logvar = self(input, time)
        loss, aux = self.vae_loss_function(reconstructed, input, mu, logvar)
        recon_loss = nrmse_loss(x=reconstructed, y=input, reduction=None).mean()
        if len(self.val_loss_avg) >= dataloader_idx +1 :
            self.val_loss_avg[dataloader_idx] += recon_loss
        else:
            self.val_loss_avg.append(recon_loss)
            
        if not self.trainer.sanity_checking:
            self.log("val_loss", loss, prog_bar=True)
            self.log("val_recon_loss", recon_loss, prog_bar=True)
            self.log("val_kl_loss", aux[1], prog_bar=True)
            if batch_idx==self.batch_to_save and self.snapshot_saved < self.num_val_dl :
                self.visualize_reconstructed(input.cpu().numpy(), reconstructed.cpu().numpy(), self.trainer.log_dir, dataloader_idx)
                self.snapshot_saved += 1

    def configure_optimizers(self):
        if self.fine_tune_recon:
            steps = self.trainer.max_steps
            optimizer = torch.optim.AdamW(self.vae.decoder.parameters(), lr=self.learning_rate, weight_decay=1e-4)
            scheduler1 =  torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-4, total_iters=1000)
            scheduler2 =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=1e-6)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[1000])
            lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "step",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "val_loss",
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": None,
            }

        else:
            steps = self.trainer.max_epochs
            optimizer = torch.optim.AdamW(self.vae.parameters(), lr=self.learning_rate, weight_decay=1e-2)
            scheduler1 =  torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-4, total_iters=10)
            scheduler2 =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=0.01 * self.learning_rate)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[10])
            lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": None,
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": False,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": None,
            }
        return {"optimizer": optimizer, 
                 "lr_scheduler": lr_scheduler_config,
                 }

    def on_fit_start(self):
        self.num_val_dl = len(iter(self.trainer.datamodule.val_dataloader()))

        if self.scaler_type == 'iterative':
            # Check if a saved scaler state exists and load it if enabled
            if self.load_scaler_state and os.path.exists(self.scaler_state_path):
                self.scaler = torch.load(self.scaler_state_path)
                print(f"Scaler state loaded from {self.scaler_state_path}")
            else:
                # Fit the scaler if state loading is disabled or the state doesn't exist
                print("Fitting scaler from training data...")
                for i, batch in enumerate(self.trainer.datamodule.train_dataloader()):
                    if i == 0:
                        self.scaler.fit(batch[0])
                    else:
                        self.scaler.update_stats(batch[0])

                # Save the scaler state for future use
                torch.save(self.scaler, self.scaler_state_path)
                print(f"Scaler state saved to {self.scaler_state_path}")
    
    def on_validation_epoch_start(self):
        self.snapshot_saved = 0
        self.batch_to_save = np.random.randint(self.trainer.num_val_batches)[0]
    
    def visualize_reconstructed(self, original_data, reconstructed_data, save_path, idx):
        quantities = ["density", "pressure", "temperature", "velocityX", "velocityY", "velocityZ"]
        snapshots_per_batch = 1

        if not os.path.exists(os.path.join(save_path, 'snapshots')):
            try:
                os.mkdir(os.path.join(save_path, 'snapshots'))
            except:
                pass

        for b in range(snapshots_per_batch):   
            for i, varname in enumerate(quantities):
                fig = plot_fields(original_data[b,i], reconstructed_data[b, i], varname)
                plt.savefig(os.path.join(os.path.join(save_path, 'snapshots') , f'{varname}_{b}_case{idx + 1}.png'))
                plt.close()
