import torch
import torch.nn as nn
from torch.nn import functional as F
from flowgen.models.UNETs.block import ConvNextBlockBlock, LayerNorm
from flowgen.models.timeEmbedding import time_embedding
from torch.nn.utils.parametrizations import spectral_norm
import xformers.ops as xops
from xformers.components.positional_embedding.rotary import RotaryEmbedding
from einops import rearrange
from functools import partial

compile_mode = 'default'

@partial(torch.compile, mode=compile_mode)
def compiled_cross_attn(q, k, v):
        #return xops.memory_efficient_attention(q, k, v)
        with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
            return F.scaled_dot_product_attention(q, k, v)


@partial(torch.compile, mode=compile_mode)
def compiled_self_attn(q, k, v):
        #return xops.memory_efficient_attention(q, k, v)
        with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
            return F.scaled_dot_product_attention(q, k, v)

class hMLP_stem(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=(8,8,8), in_chans=6, embed_dim =768):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.in_proj = torch.nn.Sequential(
            *[nn.Conv3d(in_chans, embed_dim//4, kernel_size=2, stride=2, bias=False),
            LayerNorm(embed_dim//4, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv3d(embed_dim//4, embed_dim, kernel_size=2, stride=2, bias=False),
            LayerNorm(embed_dim, eps=1e-6, data_format="channels_first"),
            #nn.GELU(),
            #nn.Conv3d(embed_dim//4, embed_dim, kernel_size=2, stride=2, bias=False),
            #LayerNorm(embed_dim, eps=1e-6, data_format="channels_first"),
            ]
            )
    
    def forward(self, x):
        x = self.in_proj(x)
        return x
    
class hMLP_output(nn.Module):
    """ Patch to Image De-bedding
    """
    def __init__(self, patch_size=(8,8,8), out_chans=6, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.out_proj = torch.nn.Sequential(
            *[nn.ConvTranspose3d(embed_dim, embed_dim//4, kernel_size=2, stride=2, bias=False),
            LayerNorm(embed_dim//4, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.ConvTranspose3d(embed_dim//4, out_chans, kernel_size=2, stride=2),
            #LayerNorm(embed_dim//4, eps=1e-6, data_format="channels_first"),
            #nn.GELU(),
            ])
        #out_head = nn.ConvTranspose3d(embed_dim//4, out_chans, kernel_size=2, stride=2)
        #self.out_kernel = nn.Parameter(out_head.weight)
        #self.out_bias = nn.Parameter(out_head.bias)
    
    def forward(self, x):
        x = self.out_proj(x)#.flatten(2).transpose(1, 2)
        #x = F.conv_transpose3d(x, self.out_kernel, self.out_bias, stride=2)
        return x

class MHCA(nn.Module):
    def __init__(self, embed_dim, x_feats, skip_feats, num_heads):
        super().__init__()
        self.k_embed = hMLP_stem(in_chans=x_feats, embed_dim=embed_dim)
        self.q_embed = hMLP_stem(in_chans=skip_feats, embed_dim=embed_dim)
        self.v_embed = hMLP_stem(in_chans=x_feats, embed_dim=embed_dim)
        self.out_proj = hMLP_output(out_chans=skip_feats, embed_dim=embed_dim)
        self.rope = RotaryEmbedding(embed_dim)
        self.num_heads = num_heads
        self.norm_skip = LayerNorm(skip_feats, eps=1e-6, data_format='channels_first')
        self.norm_x = LayerNorm(x_feats, eps=1e-6, data_format='channels_first')
        self.norm_out = LayerNorm(embed_dim, eps=1e-6, data_format='channels_first')

    def forward(self, x, skip):
        res = skip
        skip =  self.norm_skip(skip)
        x = self.norm_x(x)
        q, k = self.q_embed(skip), self.k_embed(x)
        v = self.v_embed(x)
        bs, c, *dim_q = q.shape
        _, _, *dim_kv = k.shape
        
        q = rearrange(q, 'b c h w d -> b (h w d) c').contiguous()
        k = rearrange(k, 'b c h w d -> b (h w d) c').contiguous()
        v = rearrange(v, 'b c h w d -> b (h w d) c').contiguous()

        _, k = self.rope(k, k)
        _, q = self.rope(q, q)
        _, bs, l, c = q.shape

        q, k, v = (rearrange(q.squeeze(0), 'b (h w d) (nh c) -> b nh (h w d) c', nh=self.num_heads, h=dim_q[0], w=dim_q[1], d=dim_q[2]).contiguous(), 
                  rearrange(k.squeeze(0), 'b (h w d) (nh c) -> b nh (h w d) c', nh=self.num_heads, h=dim_kv[0], w=dim_kv[1], d=dim_kv[2]).contiguous(),  
                  rearrange(v, 'b (h w d) (nh c) -> b nh (h w d) c', nh=self.num_heads, h=dim_kv[0], w=dim_kv[1], d=dim_kv[2]).contiguous())

        #proj = xops.memory_efficient_attention(q, k, v)
        proj = compiled_cross_attn(q, k, v)
        proj = rearrange(proj, 'b nh (h w d) c -> b (c nh) h w d', h=dim_q[0], w=dim_q[1], d=dim_q[2]).contiguous()
        proj = self.norm_out(proj)
        proj =  self.out_proj(proj)
        return proj + res

class MHSA(nn.Module):
    def __init__(self, embed_dim, x_feats, num_heads):
        super().__init__()
        self.k_embed = nn.Sequential(
            nn.Linear(x_feats, embed_dim//4),
            LayerNorm(embed_dim//4, eps=1e-6, data_format="channels_last"),
            nn.GELU(),
            nn.Linear(embed_dim//4, embed_dim),
            LayerNorm(embed_dim, eps=1e-6, data_format="channels_last"),
            nn.GELU(),
        )
        self.q_embed = nn.Sequential(
            nn.Linear(x_feats, embed_dim//4),
            LayerNorm(embed_dim//4, eps=1e-6, data_format="channels_last"),
            nn.GELU(),
            nn.Linear(embed_dim//4, embed_dim),
            LayerNorm(embed_dim, eps=1e-6, data_format="channels_last"),
            nn.GELU(),
        )
        self.v_embed = nn.Sequential(
            nn.Linear(x_feats, embed_dim//4),
            LayerNorm(embed_dim//4, eps=1e-6, data_format="channels_last"),
            nn.GELU(),
            nn.Linear(embed_dim//4, embed_dim),
            LayerNorm(embed_dim, eps=1e-6, data_format="channels_last"),
            nn.GELU(),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//4),
            LayerNorm(embed_dim//4, eps=1e-6, data_format="channels_last"),
            nn.GELU(),
            nn.Linear(embed_dim//4, x_feats),
        )
        self.rope = RotaryEmbedding(embed_dim)
        self.num_heads = num_heads

        self.norm_x = LayerNorm(x_feats, eps=1e-6, data_format='channels_last')
        self.norm_out = LayerNorm(embed_dim, eps=1e-6, data_format='channels_last')

    def forward(self, x):
        bs, c, *dim = x.shape
        #x = x.permute(0,2,3,4,1).reshape(bs, -1, c).contiguous()
        res = x
        x = rearrange(x, 'b c h w d -> b (h w d) c').contiguous()
        x =  self.norm_x(x)
        q, k = self.q_embed(x), self.k_embed(x)
        v = self.v_embed(x)
        q, k = self.rope(q, k)
        _, bs, l, c = q.shape

        q, k, v = (rearrange(q.squeeze(0), 'b l (nh c) -> b nh l c', nh=self.num_heads).contiguous(), 
                  rearrange(k.squeeze(0), 'b l (nh c) -> b nh l c', nh=self.num_heads).contiguous(),  
                  rearrange(v, 'b l (nh c) -> b nh l c', nh=self.num_heads).contiguous())
        #proj = xops.memory_efficient_attention(q, k, v)
        proj = compiled_self_attn(q, k, v)
        proj = rearrange(proj, 'b nh (h w d) c -> b (h w d) (nh c)', nh=self.num_heads, h=dim[0], w=dim[1], d=dim[2]).contiguous()
        proj =  self.norm_out(proj)
        proj = self.out_proj(proj)
        proj = rearrange(proj, 'b (h w d) c -> b c h w d', h=dim[0], w=dim[1], d=dim[2]).contiguous()

        return proj + res


class TimeConditionedLayerNorm(nn.Module):
    # Implementation of time-conditioned layer normalization
    def __init__(self, channels, eps=1e-6, data_format="channels_first"):
        super().__init__()

        self.norm = LayerNorm(channels, eps=eps, data_format=data_format)

        self.gamma = nn.Conv1d(channels, channels, 1, bias=True)
        self.beta = nn.Conv1d(channels, channels, 1, bias=True)

    def forward(self, x, t):
        bs, c, *dims = x.shape
        gamma = self.gamma(t.reshape(bs,c,-1)).view(bs,c,*[1]*len(dims))
        beta = self.beta(t.reshape(bs,c,-1)).view(bs,c,*[1]*len(dims))
        x = gamma * self.norm(x) + beta
        return x

class Attention_Unet(nn.Module):
    def __init__(self, in_channels, out_channels, dims=[16, 32, 32, 16], depths=[2,2,2,2], num_heads=4, embed_dim=768):
        super().__init__()
        self.encoder = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.decoder = nn.ModuleList()
        
        self.stem = nn.Conv3d(in_channels, dims[0], kernel_size=5, padding=2)
        self.norm_input = TimeConditionedLayerNorm(dims[0], eps=1e-6, data_format="channels_first")

        self.self_attention = MHSA(embed_dim,dims[-1],num_heads)

        for i, (dim, depth) in enumerate(zip(dims[:-1], depths[:-1])):
            stage = nn.Sequential(
                *[ConvNextBlockBlock(dim=dim) for j in range(depth)],
            )
            downsample = nn.ModuleList(
                [TimeConditionedLayerNorm(dim, eps=1e-6, data_format="channels_first"),
                nn.Conv3d(dim, dims[i+1], kernel_size=2, stride=2)]
            )
            self.encoder.append(nn.ModuleList([stage, downsample]))
        
        for i, (dim, depth) in enumerate(zip(reversed(dims[1:]), reversed(depths[1:]))):
            stage = nn.Sequential(
                *[ConvNextBlockBlock(dim=max(1,2-j) * dim, out_dim=dim) for j in range(depth)],)
            upsample = nn.ModuleList(
                    [TimeConditionedLayerNorm(dim, eps=1e-6, data_format="channels_first"),
                    nn.ConvTranspose3d(dim, dims[i+1], kernel_size=2, stride=2)]
            )
            attn_gate = MHCA(embed_dim, dim, dim, num_heads)                
            self.decoder.append(nn.ModuleList([stage, upsample, attn_gate]))
        
        self.output_layer = nn.Sequential(
            nn.Conv3d(dims[-1], dims[-1], kernel_size=5, padding=2, groups=dims[-1]),
            LayerNorm(dims[-1], eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv3d(dims[-1], out_channels, kernel_size=1),
        )
        hidden_dim = dims[0]
        self.time_embedding = time_embedding(0.1, hidden_dim)
        
        self.time_mlp_0 = nn.Sequential(nn.Linear(in_features=hidden_dim*2, out_features=hidden_dim, 
                                                                    bias=True),
                                      nn.GELU(),
                                      nn.Linear(in_features=hidden_dim, out_features=hidden_dim,
                                                                   bias=True))
        self.time_mlp_1 = nn.Sequential(nn.Linear(in_features=hidden_dim*2, out_features=hidden_dim, 
                                                                    bias=True),
                                      nn.GELU(),
                                      nn.Linear(in_features=hidden_dim, out_features=hidden_dim,
                                                                   bias=True))

    def forward(self, x, *time):

        bs, c, *dims = x.shape
        
        t0, t = time
        t0 = self.time_embedding(t0)
        t = self.time_embedding(t)
        t0 = self.time_mlp_0(t0)
        t = self.time_mlp_1(t)
        x = self.stem(x)
        x = self.norm_input(x, t0)
        # Encoder
        skip_connections = []
        for enc, ds in self.encoder:
            x = enc(x)
            x = ds[0](x,t)
            #save skip connection
            skip_connections.append(x)
            #Downsample
            x = ds[1](x)
        x = self.self_attention(x)
        
        # Decoder
        for (dec, us, attn_gate), skip in zip(self.decoder, reversed(skip_connections)):
            
            skip = attn_gate(x, skip)
            
            x = us[0](x, t)
            x = us[1](x)
            
            x = torch.cat((x, skip), dim=1)
            x = dec(x)

        x = self.output_layer(x)

        return x
    
if __name__ == "__main__":
    device = torch.device('cuda')
    batch_size = 2
    x = torch.rand(batch_size,6,32,32,32, device=device)
    t0 = torch.rand(batch_size, device=device)
    t1 = t0 + 0.1
    model = Attention_Unet(6,6, dims=[128,128,128,128], depths=[2,2,2,2])
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'total number of params: {pytorch_total_params}')
    model.to(device)
    pred = model(x, t0, t1)

    print(pred.shape)

    print(pred.mean(), pred.std())



