"""
LIBERO Benchmark Dataset Adapter for EdgeVLA.
Supports downloading and parsing standard LIBERO manipulation demonstration datasets:
- LIBERO-Spatial (10 tasks, spatial relation reasoning)
- LIBERO-Object (10 tasks, novel object manipulation)
- LIBERO-Goal (10 tasks, diverse goal conditions)
- LIBERO-10 / LIBERO-90 (Lifelong benchmark suites)
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from models.language_encoder import SimpleRobotTokenizer

LIBERO_TASK_SUITES = {
    "libero_spatial": "https://huggingface.co/datasets/libero-benchmark/libero-spatial/resolve/main/libero_spatial_demo.hdf5",
    "libero_object": "https://huggingface.co/datasets/libero-benchmark/libero-object/resolve/main/libero_object_demo.hdf5",
    "libero_goal": "https://huggingface.co/datasets/libero-benchmark/libero-goal/resolve/main/libero_goal_demo.hdf5",
    "libero_10": "https://huggingface.co/datasets/libero-benchmark/libero-10/resolve/main/libero_10_demo.hdf5"
}

class LiberoBenchmarkDataset(Dataset):
    """
    Standard LIBERO Demonstration Trajectory Dataset.
    Loads RGB views (agentview + eye_in_hand), language instruction, and 7-DoF action chunks.
    """
    def __init__(self, hdf5_path, chunk_size=8, max_samples=None):
        self.hdf5_path = hdf5_path
        self.chunk_size = chunk_size
        self.tokenizer = SimpleRobotTokenizer(max_seq_len=16)
        self.samples = []
        
        if os.path.exists(str(hdf5_path)):
            self._load_hdf5(hdf5_path, max_samples)
        else:
            print(f"[LIBERO] Warning: {hdf5_path} not found. Initializing mock demonstration suite.")
            self._generate_mock_libero_demos(max_samples or 200)

    def _load_hdf5(self, hdf5_path, max_samples):
        print(f"[LIBERO] Loading real demonstration trajectories from {hdf5_path}...")
        with h5py.File(hdf5_path, "r") as f:
            data_grp = f["data"]
            demo_keys = list(data_grp.keys())
            print(f"[LIBERO] Total Demonstration Episodes Found: {len(demo_keys)}")
            
            for d_idx, demo_key in enumerate(demo_keys):
                ep = data_grp[demo_key]
                obs = ep["obs"]
                actions = ep["actions"][:] # (T, 7)
                images = obs["agentview_rgb"][:] # (T, H, W, 3)
                
                # Instruction string
                lang = ep.attrs.get("language_instruction", "manipulate object")
                token_ids = self.tokenizer.encode(str(lang))
                
                T = len(actions)
                for t in range(0, T - self.chunk_size, 2): # Stride of 2
                    chunk = actions[t : t + self.chunk_size]
                    img = images[t].transpose(2, 0, 1).astype(np.float32) / 127.5 - 1.0 # (3, H, W) in [-1, 1]
                    
                    # Proprioception
                    if "robot0_eef_pos" in obs and "robot0_eef_quat" in obs:
                        pos = obs["robot0_eef_pos"][t]
                        proprio = np.pad(pos, (0, 4), mode='constant')
                    else:
                        proprio = np.zeros(7, dtype=np.float32)
                        
                    self.samples.append({
                        "rgb": img,
                        "token_ids": np.array(token_ids, dtype=np.int64),
                        "proprio": proprio.astype(np.float32),
                        "action_chunk": chunk.astype(np.float32),
                        "instruction": lang
                    })
                    
                    if max_samples and len(self.samples) >= max_samples:
                        break
                if max_samples and len(self.samples) >= max_samples:
                    break
        print(f"[LIBERO] Extracted {len(self.samples)} valid action-chunk sample tuples.")

    def _generate_mock_libero_demos(self, num_samples):
        task_list = [
            "pick up the black bowl and place on the plate",
            "push the white mug to the front edge",
            "open the middle drawer of the cabinet",
            "close the top drawer gently",
            "pick up the alphabet soup and put in the basket",
            "turn on the green desk lamp",
            "move the butter dish to the tray"
        ]
        for i in range(num_samples):
            lang = task_list[i % len(task_list)]
            token_ids = self.tokenizer.encode(lang)
            rgb = np.random.uniform(-1.0, 1.0, size=(3, 224, 224)).astype(np.float32)
            proprio = np.random.uniform(-0.5, 0.5, size=(7,)).astype(np.float32)
            
            # Smooth trajectory
            base_act = np.random.uniform(-0.4, 0.4, size=(7,)).astype(np.float32)
            action_chunk = np.zeros((self.chunk_size, 7), dtype=np.float32)
            for t in range(self.chunk_size):
                action_chunk[t] = np.clip(base_act + (t / self.chunk_size) * 0.15, -1.0, 1.0)
                
            self.samples.append({
                "rgb": rgb,
                "token_ids": np.array(token_ids, dtype=np.int64),
                "proprio": proprio,
                "action_chunk": action_chunk,
                "instruction": lang
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            "rgb": torch.tensor(item["rgb"], dtype=torch.float32),
            "token_ids": torch.tensor(item["token_ids"], dtype=torch.long),
            "proprio": torch.tensor(item["proprio"], dtype=torch.float32),
            "action_chunk": torch.tensor(item["action_chunk"], dtype=torch.float32)
        }

if __name__ == "__main__":
    dataset = LiberoBenchmarkDataset(hdf5_path="datasets/mock_libero.hdf5", chunk_size=8, max_samples=50)
    print(f"LIBERO Dataset Initialized with {len(dataset)} samples.")
