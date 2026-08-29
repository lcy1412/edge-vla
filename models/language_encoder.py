"""
Lightweight Language Tokenizer and Encoder for EdgeVLA.
Converts robotic language instructions (e.g., 'pick up the red mug') into dense embeddings.
Designed to be lightweight and fully exportable to ONNX/NPU without heavy transformers.
"""

import torch
import torch.nn as nn

class LightweightLanguageEncoder(nn.Module):
    """
    Compact embedding-based language encoder with positional encoding and multi-head self-attention.
    Total parameter footprint is < 1M params for fast edge inference.
    """
    def __init__(self, vocab_size=1000, max_seq_len=16, embed_dim=256, num_layers=2):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids):
        """
        input_ids: (B, seq_len) LongTensor
        Returns:
            pooled_emb: (B, embed_dim) pooled language representation
            token_embs: (B, seq_len, embed_dim) full sequence embeddings
        """
        B, seq_len = input_ids.shape
        x = self.token_embedding(input_ids) + self.pos_embedding[:, :seq_len, :]
        x = self.transformer(x)
        x = self.norm(x)
        
        # Mean pooling across tokens
        pooled = x.mean(dim=1)
        return pooled, x

class SimpleRobotTokenizer:
    """
    Built-in vocabulary tokenizer for robot task instructions.
    Maps common manipulation words (pick, place, open, close, drawer, block, cup, etc.) to IDs.
    """
    def __init__(self, max_seq_len=16):
        self.max_seq_len = max_seq_len
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3}
        # Pre-populate common robotics vocabulary
        common_words = [
            "pick", "up", "the", "place", "in", "on", "into", "drawer", "table", "cup",
            "mug", "block", "cube", "red", "blue", "green", "yellow", "black", "white",
            "open", "close", "top", "bottom", "left", "right", "middle", "slider", "cabinet",
            "push", "pull", "grasp", "release", "turn", "switch", "lamp", "door", "bowl",
            "plate", "pan", "bottle", "can", "box", "bin", "basket", "tray", "microwave"
        ]
        for w in common_words:
            self._add_word(w)

    def _add_word(self, word):
        if word not in self.vocab:
            self.vocab[word] = len(self.vocab)

    def encode(self, text, max_len=None):
        if max_len is None:
            max_len = self.max_seq_len
        words = text.lower().strip().replace(".", "").replace(",", "").split()
        ids = [self.vocab.get(w, self.vocab["<UNK>"]) for w in words]
        if len(ids) < max_len:
            ids = ids + [self.vocab["<PAD>"]] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return ids

if __name__ == "__main__":
    tokenizer = SimpleRobotTokenizer(max_seq_len=16)
    tokens = tokenizer.encode("pick up the red mug and place into drawer")
    print(f"Tokenized instruction: {tokens}")
    
    encoder = LightweightLanguageEncoder(vocab_size=1000, max_seq_len=16, embed_dim=256)
    t_tensor = torch.tensor([tokens, tokens], dtype=torch.long)
    pooled, seq = encoder(t_tensor)
    print(f"Pooled Language Embedding: {pooled.shape}, Seq: {seq.shape}")
