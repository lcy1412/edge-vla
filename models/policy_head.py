"""
Action Chunking Policy Head for EdgeVLA.
Predicts a continuous multi-step trajectory chunk:
A \in R^{K \times D} where K is the chunk horizon (e.g., K=8) and D is the action dimension (e.g., 7-DoF: dx, dy, dz, droll, dpitch, dyaw, gripper).
"""

import torch
import torch.nn as nn

class ActionChunkingMLPHead(nn.Module):
    """
    Direct Multi-Layer Perceptron Action Chunking Head.
    Input: Multi-modal fusion feature (B, fusion_dim)
    Output: Action chunk (B, chunk_size, action_dim)
    """
    def __init__(self, in_features=512, chunk_size=8, action_dim=7, hidden_dim=512):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, chunk_size * action_dim)
        )
        
        # Tanh activation for normalized actions in [-1, 1]
        self.tanh = nn.Tanh()

    def forward(self, x):
        B = x.shape[0]
        flat_actions = self.net(x)
        actions = flat_actions.view(B, self.chunk_size, self.action_dim)
        # Apply tanh for spatial/velocity limits
        return self.tanh(actions)

class ActionChunkingTransformerHead(nn.Module):
    """
    Transformer-based Action Chunking Decoder (ACT-style).
    Uses learned query tokens to attend to multi-modal fused conditioning tokens.
    """
    def __init__(self, embed_dim=256, chunk_size=8, action_dim=7, nhead=4, num_layers=2):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        
        # Learnable action queries for K steps
        self.action_queries = nn.Parameter(torch.randn(1, chunk_size, embed_dim) * 0.02)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.action_proj = nn.Linear(embed_dim, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, conditioning_tokens):
        """
        conditioning_tokens: (B, num_tokens, embed_dim)
        """
        B = conditioning_tokens.shape[0]
        queries = self.action_queries.expand(B, -1, -1)
        hidden = self.decoder(tgt=queries, memory=conditioning_tokens)
        actions = self.action_proj(hidden)
        return self.tanh(actions)

if __name__ == "__main__":
    head = ActionChunkingMLPHead(in_features=512, chunk_size=8, action_dim=7)
    dummy_feat = torch.randn(2, 512)
    actions = head(dummy_feat)
    print(f"MLP Head Action Chunk Output Shape: {actions.shape}")
