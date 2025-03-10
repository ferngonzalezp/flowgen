import torch
import torch.nn as nn
from flowgen.models.VAE.embedding import get_1d_sincos_pos_embed_from_grid, RotaryPositionEmbeddingPytorchV2, RotaryPositionEmbeddingPytorchV1
from einops import rearrange

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

class PerceiverEncoder(nn.Module):
    def __init__(self, embed_dim, mlp_dim, latent_len, num_heads=4, num_layers=3):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(latent_len, embed_dim))  # Learnable latents

        # Iterative Cross-Attention (Latents <-> Input)
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        # Self-Attention in Latents (Latents <-> Latents)
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, activation='gelu',
                                        dim_feedforward=mlp_dim, batch_first=True, dropout=0.1)
            for _ in range(num_layers)
        ])

        self.norms_cross = nn.ModuleList([nn.RMSNorm(embed_dim) for _ in range(num_layers)])
        #self.norms_self = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])

        self.num_heads = num_heads

    def forward(self, x):
        """
        x: (batch, time, embed_dim)  -> Full temporal sequence
        Returns: (batch, latent_len, embed_dim)  -> Compressed latent representation
        """
        B, L, D = x.shape
        latents = self.latents.expand(B, -1, -1)  # Expand latents for batch
        
        # Iterative Cross-Attention + Self-Attention
        for i in range(len(self.cross_attn_layers)):
            # Cross-Attention (Latents attend to Input with shared KV)
            latents = latents + self.cross_attn_layers[i](latents, x, x)[0]
            latents = self.norms_cross[i](latents)
            
            # Self-Attention (Latents refine among themselves)
            latents = latents + self.self_attn_layers[i](latents)
            #latents = self.norms_self[i](latents)

        return latents

class PerceiverDecoder(nn.Module):
    def __init__(self, embed_dim, num_output_queries, num_heads=4, num_layers=3, mlp_dim=256):
        super().__init__()
        self.output_queries = nn.Parameter(torch.randn(num_output_queries, embed_dim))  # Learnable queries

        # Iterative Cross-Attention (Queries <-> Latents)
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=0.1)
            for _ in range(num_layers)
        ])

        # Self-Attention in Output Queries (Refinement)
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, activation='gelu',
                                        dim_feedforward=mlp_dim, batch_first=True, dropout=0.1)
            for _ in range(num_layers)
        ])

        self.norms_cross = nn.ModuleList([nn.RMSNorm(embed_dim) for _ in range(num_layers)])
        #self.norms_self = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads

    def forward(self, latents, time):
        """
        latents: (batch, num_latents, dim_latent)  -> Compressed latent representation
        Returns: (batch, num_output_queries, dim_output)  -> Decoded output
        """
        batch_size = latents.shape[0]
        L = self.output_queries.shape[0]
        queries = self.output_queries.expand(batch_size, -1, -1)  # Expand queries for batch
        t_pos =[]
        for b in range(time.shape[0]):
            t_pos.append(torch.tensor(get_1d_sincos_pos_embed_from_grid(self.embed_dim, time[b].cpu())))
        t_pos = torch.stack(t_pos).type_as(queries)
        t_pos = t_pos.unsqueeze(1).expand(-1, batch_size//t_pos.shape[0], -1, -1)
        queries = queries + rearrange(t_pos, 'b n l d -> (b n) l d')

        mask = generate_causal_mask(batch_size=batch_size, 
                                        seq_len=L, 
                                        num_heads=self.num_heads, 
                                        device=latents.device)

        # Iterative Refinement
        for i in range(len(self.cross_attn_layers)):
            # Cross-Attention: Output queries extract information from latents
            
            queries = queries + self.cross_attn_layers[i](queries, latents, latents)[0]
            queries = self.norms_cross[i](queries)

            # Self-Attention: Queries refine among themselves
            queries = queries + self.self_attn_layers[i](queries, src_mask = None)
            #queries = self.norms_self[i](queries)

        return queries