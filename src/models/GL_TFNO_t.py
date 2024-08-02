import torch.nn as nn
import torch.nn.functional as F
import torch
from neuralop.layers.spectral_convolution import SpectralConv
from timeEmbedding import time_embedding
from torch.nn.utils.parametrizations import spectral_norm

class timeCondIN(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.norm = nn.GroupNorm(1,channels, affine=False)

        self.gamma = nn.Conv1d(channels, channels, 1, bias=True)
        self.beta = nn.Conv1d(channels, channels, 1, bias=True)

    def forward(self, x, t):
        bs, c, *dims = x.shape
        gamma = self.gamma(t.reshape(bs,c,-1)).view(bs,c,*[1]*len(dims))
        beta = self.beta(t.reshape(bs,c,-1)).view(bs,c,*[1]*len(dims))
        x = gamma * self.norm(x) + beta
        return x

class SpatialAttentionWithMemory(nn.Module):
    def __init__(self, channels, spatial_kernel_size=3, memory_size=5):
        super().__init__()
        self.channels = channels
        self.spatial_kernel_size = spatial_kernel_size
        self.memory_size = memory_size
        
        self.query = nn.Conv3d(channels, channels, 1)
        self.key = nn.Conv3d(channels, channels, 1)
        self.value = nn.Conv3d(channels, channels, 1)
        
        # Memory for temporal information
        self.memory_key = nn.Parameter(torch.randn(1, channels, memory_size, 1, 1, 1))
        self.memory_value = nn.Parameter(torch.randn(1, channels, memory_size, 1, 1, 1))
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Conv3d(channels * 2, channels, 1),
            nn.Sigmoid()
        )
        
        # Memory update mechanism
        self.memory_update = nn.GRUCell(channels, channels)
        
    def forward(self, x, prev_memory=None):
        B, C, H, D, W = x.shape
        
        # Pad the input for local attention
        padding = self.spatial_kernel_size // 2
        x_padded = F.pad(x, (padding, padding, padding, padding, padding, padding))
        
        # Create spatial patches
        B, C, H, D, W = x_padded.shape
        patches = F.unfold(x_padded.reshape(B*C, 1, H, D, W), 
                           kernel_size=self.spatial_kernel_size).reshape(B, C, -1, H, D, W)
        
        # Compute attention
        q = self.query(x).reshape(B, C, -1)
        k_spatial = self.key(patches.reshape(B, C, -1, H, D, W)).reshape(B, C, -1, H*D*W)
        v_spatial = self.value(patches.reshape(B, C, -1, H, D, W)).reshape(B, C, -1, H*D*W)
        
        # Include memory in key and value
        k = torch.cat([self.memory_key.expand(B, -1, -1, H, D, W).reshape(B, C, -1, H*D*W), k_spatial], dim=2)
        v = torch.cat([self.memory_value.expand(B, -1, -1, H, D, W).reshape(B, C, -1, H*D*W), v_spatial], dim=2)
        
        attn = torch.einsum('bcn,bcmn->bcmn', q, k) / (C ** 0.5)
        attn = F.softmax(attn, dim=2)
        
        out = torch.einsum('bcmn,bcmn->bcn', attn, v)
        out = out.reshape(B, C, H, D, W)
        
        # Apply gating mechanism
        gate = self.gate(torch.cat([x, out], dim=1))
        out = gate * out + (1 - gate) * x
        
        # Update memory
        if prev_memory is None:
            prev_memory = torch.zeros(B*C, self.channels, device=x.device)
        
        memory_update = out.mean(dim=[2,3,4]).reshape(B*C, -1)
        new_memory = self.memory_update(memory_update, prev_memory)
        
        return out, new_memory.reshape(B, C, -1)

class FNO_block(nn.Module):
    
    def __init__(self, in_channels, out_channels, modes, factorization,
                        rank ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.factorization = factorization
        self.rank = rank
        
        self.spec_conv = SpectralConv(in_channels=in_channels,
                                     out_channels=out_channels,
                                     rank=rank,
                                     n_modes=modes,
                                     factorization=factorization,
                                     implementation='factorized',
                                     n_layers=1)
        
        self.mlp = nn.Sequential(nn.Conv1d(out_channels, out_channels//2, 1, bias=True),
                                           nn.GELU(),
                                           nn.Conv1d(out_channels//2, out_channels, 1,
                                                     bias=True))
        self.norm1 = nn.GroupNorm(1,out_channels)
        self.norm2 = timeCondIN(out_channels)

        self.spatial_attention = SpatialAttentionWithMemory(out_channels)

    def forward(self,x,t,prev_memory):
        
        bs, c, *dims = x.shape
        skip1 = x
        skip2 = nn.functional.gelu(x)
        x = self.spec_conv(x, 0, output_shape=None)
        x = nn.functional.gelu(self.norm1(x) + skip1)
        x = self.norm2(self.mlp(x.reshape(bs,c,-1)).reshape((bs,c,*dims)), t)

        # Apply spatial attention with memory
        x, new_memory = self.spatial_attention(x, prev_memory)
        x = x + skip2
        return x, new_memory


class GL_TFNO_t(nn.Module):
    
    def __init__(self, modes = (16,16,16), precision='full', factorization='tucker', 
                 rank=0.42, layers=4, hidden_dim=64, in_channels=6, out_channels=6):
        
        super().__init__()
        
        self.modes = modes
        self.precision = precision
        self.factorization = factorization
        self.rank = rank
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Initialize memory
        self.memory = None
        
        self.spec_conv = nn.ModuleList([FNO_block(hidden_dim, hidden_dim, modes, factorization,
                        rank) for i in range(layers)])
        
        #self.lifting = nn.Conv1d(in_channels + 1, hidden_dim,1)
        self.lifting = nn.Sequential(nn.Conv3d(in_channels+1, in_channels+1, 1, groups=in_channels+1),
                                     nn.GroupNorm(in_channels+1, in_channels+1),
                                     nn.GELU(),
                                     nn.Conv3d(in_channels+1, hidden_dim, 1))
        #self.projection = nn.ConvTranspose1d(hidden_dim, out_channels,1)
        self.projection = nn.Sequential(nn.Conv3d(hidden_dim, hidden_dim*out_channels, 1, groups=hidden_dim),
                                     nn.GroupNorm(hidden_dim*out_channels, hidden_dim*out_channels),
                                     nn.GELU(),
                                     nn.Conv3d(hidden_dim*out_channels, out_channels, 1, groups=out_channels))
        
        self.time_embedding = time_embedding(0.1, hidden_dim)
        
        self.time_mlp_0 = nn.Sequential(spectral_norm(nn.Linear(in_features=hidden_dim*2, out_features=64, 
                                                                    bias=True)),
                                      nn.GELU(),
                                      spectral_norm(nn.Linear(in_features=64, out_features=1,
                                                                   bias=True)))
        self.time_mlp_1 = nn.Sequential(spectral_norm(nn.Linear(in_features=hidden_dim*2, out_features=hidden_dim, 
                                                                    bias=True)),
                                      nn.GELU(),
                                      spectral_norm(nn.Linear(in_features=hidden_dim, out_features=hidden_dim,
                                                                   bias=True)))
    def forward(self,x,t0,t):
        
        bs, c, *dims = x.shape
        t0 = self.time_embedding(t0)
        t = self.time_embedding(t)
        t0 = self.time_mlp_0(t0)
        t = self.time_mlp_1(t)

        t0 = t0.reshape(bs,1,*[1]*len(dims)).repeat(1,1,*dims)
        x = torch.cat((x,t0), dim=1)
        #x = self.lifting(x.reshape(bs,c + 1,-1)).reshape(bs,self.hidden_dim,*dims)
        x = self.lifting(x)
        if self.memory is None:
            self.memory = [None] * len(self.spec_conv)
        
        for i, _ in enumerate(self.spec_conv):
            x, self.memory[i] = self.spec_conv[i](x, t, self.memory[i])
        
        #x = self.projection(x.reshape(bs,self.hidden_dim,-1)).reshape(bs,self.out_channels,*dims)
        x = self.projection(x)
        return x
    
    def reset_memory(self):
        self.memory = None
    
    def detach_memory(self):
        if self.memory is not None:
            self.memory = [m.detach() if m is not None else None for m in self.memory]
    