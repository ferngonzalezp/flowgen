from VAE import VAE4D
import torch
from einops import repeat

# Define the VAE hyperparameters
input_dim = 6           # Single input channel for CFD data (e.g., scalar fields like velocity magnitude)
out_dim = 6
embed_dim = 64          # Latent embedding dimension
patch_size = (4, 4, 4)  # Spatial patch size
spatial_heads = 4       # Number of attention heads for spatial blocks
temporal_heads = 2      # Number of attention heads for temporal blocks
mlp_dim = 128           # MLP hidden dimension
num_spatial_blocks = 2  # Number of spatial attention blocks
num_temporal_blocks = 2 # Number of temporal attention blocks
time_downsample = 5     # Temporal downsampling factor
latent_dim = 32         # Latent space dimension

# Create a VAE instance
vae = VAE4D(
    input_dim=input_dim,
    out_dim = out_dim,
    embed_dim=embed_dim,
    patch_size=patch_size,
    spatial_heads=spatial_heads,
    temporal_heads=temporal_heads,
    mlp_dim=mlp_dim,
    num_spatial_blocks=num_spatial_blocks,
    num_temporal_blocks=num_temporal_blocks,
    time_downsample=time_downsample,
    latent_dim=latent_dim
)

device =  torch.device('cuda')

vae.to(device)
x =  torch.rand(5,6,32,32,32,5, device=device)
t = torch.linspace(0, 1, steps=5)
t = repeat(t, 'l -> b l', b=5)

# Forward pass through the VAE
reconstructed, mu, logvar = vae(x, t)

# Print shapes to verify
print(f"Input Shape: {x.shape}")
print(f"Reconstructed Shape: {reconstructed.shape}")
print(f"Latent Mean Shape: {mu.shape}")
print(f"Latent LogVar Shape: {logvar.shape}")