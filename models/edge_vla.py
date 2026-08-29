"""
Integrated EdgeVLA Policy Model.
Combines Lightweight Vision Encoder, Language Encoder, Cross-Modal Fusion, and Action Chunking Head.
"""

import torch
import torch.nn as nn
from models.vision_encoder import LightweightVisionEncoder
from models.language_encoder import LightweightLanguageEncoder
from models.policy_head import ActionChunkingMLPHead, ActionChunkingTransformerHead

class EdgeVLA(nn.Module):
    """
    EdgeVLA: Fast and Lightweight Vision-Language-Action Policy.
    """
    def __init__(
        self,
        embed_dim=256,
        action_dim=7,
        chunk_size=8,
        vocab_size=1000,
        max_seq_len=16,
        head_type="mlp",
        use_proprioception=True,
        proprio_dim=7
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.use_proprio = use_proprioception
        self.head_type = head_type
        
        # 1. Vision Backbone
        self.vision_encoder = LightweightVisionEncoder(embed_dim=embed_dim)
        
        # 2. Language Backbone
        self.language_encoder = LightweightLanguageEncoder(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            embed_dim=embed_dim
        )
        
        # 3. Proprioception Projector (Current Robot Arm State)
        if use_proprioception:
            self.proprio_proj = nn.Sequential(
                nn.Linear(proprio_dim, embed_dim // 2),
                nn.SiLU(inplace=True),
                nn.Linear(embed_dim // 2, embed_dim)
            )
            fusion_in_dim = embed_dim * 3
        else:
            fusion_in_dim = embed_dim * 2
            
        # 4. Cross-Modal Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim * 2, embed_dim * 2)
        )
        
        # 5. Policy Head
        if head_type == "transformer":
            self.policy_head = ActionChunkingTransformerHead(
                embed_dim=embed_dim,
                chunk_size=chunk_size,
                action_dim=action_dim
            )
        else:
            self.policy_head = ActionChunkingMLPHead(
                in_features=embed_dim * 2,
                chunk_size=chunk_size,
                action_dim=action_dim
            )

    def forward(self, rgb_image, language_ids, proprio_state=None):
        """
        Forward pass for training and inference.
        Args:
            rgb_image: Tensor (B, 3, 224, 224) normalized to [-1, 1] or [0, 1]
            language_ids: LongTensor (B, seq_len)
            proprio_state: Tensor (B, proprio_dim) [optional]
        Returns:
            predicted_actions: (B, chunk_size, action_dim)
        """
        # Vision features
        v_feat = self.vision_encoder(rgb_image, return_spatial_tokens=False) # (B, embed_dim)
        
        # Language features
        l_feat, _ = self.language_encoder(language_ids)                      # (B, embed_dim)
        
        if self.use_proprio:
            if proprio_state is None:
                # Default zero proprioception if not provided
                proprio_state = torch.zeros(rgb_image.shape[0], 7, device=rgb_image.device, dtype=rgb_image.dtype)
            p_feat = self.proprio_proj(proprio_state)                        # (B, embed_dim)
            fused_input = torch.cat([v_feat, l_feat, p_feat], dim=-1)
        else:
            fused_input = torch.cat([v_feat, l_feat], dim=-1)
            
        # Cross-modal fused representation
        fused_rep = self.fusion(fused_input)                                 # (B, embed_dim * 2)
        
        # Predict multi-step action chunk
        if self.head_type == "transformer":
            tokens = fused_rep.unsqueeze(1) # (B, 1, embed_dim*2) -> project if needed
            actions = self.policy_head(tokens)
        else:
            actions = self.policy_head(fused_rep)                            # (B, chunk_size, action_dim)
            
        return actions

def build_edge_vla(variant="base", chunk_size=8):
    """
    Factory function for EdgeVLA model variants.
    """
    if variant == "tiny":
        return EdgeVLA(embed_dim=128, chunk_size=chunk_size, head_type="mlp")
    elif variant == "base":
        return EdgeVLA(embed_dim=256, chunk_size=chunk_size, head_type="mlp")
    else:
        raise ValueError(f"Unknown variant: {variant}")

if __name__ == "__main__":
    model = build_edge_vla("base", chunk_size=8)
    B = 2
    dummy_img = torch.randn(B, 3, 224, 224)
    dummy_lang = torch.randint(0, 50, (B, 16), dtype=torch.long)
    dummy_prop = torch.randn(B, 7)
    
    out_actions = model(dummy_img, dummy_lang, dummy_prop)
    print(f"EdgeVLA Forward Test Passed! Output Action Chunk Shape: {out_actions.shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params / 1e6:.2f} M")
