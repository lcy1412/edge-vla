"""
ONNX Export script for EdgeVLA Policy.
Exports the PyTorch model to ONNX format with static tensor shapes optimized for Ascend NPU compilation.
"""

import os
import argparse
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.edge_vla import build_edge_vla

def export_to_onnx(checkpoint_path, output_onnx_path, variant="base", chunk_size=8):
    print(f"[Export] Loading EdgeVLA ({variant}) from checkpoint: {checkpoint_path}...")
    model = build_edge_vla(variant=variant, chunk_size=chunk_size)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print("[Export] Checkpoint weights successfully loaded.")
    else:
        print("[Export] Warning: No checkpoint found, exporting with initialized weights.")
        
    model.eval()
    
    # Dummy inputs with fixed batch=1 for embedded NPU latency optimization
    dummy_rgb = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    dummy_lang = torch.zeros(1, 16, dtype=torch.long)
    dummy_prop = torch.zeros(1, 7, dtype=torch.float32)
    
    os.makedirs(os.path.dirname(os.path.abspath(output_onnx_path)), exist_ok=True)
    
    print(f"[Export] Exporting to ONNX: {output_onnx_path}...")
    torch.onnx.export(
        model,
        (dummy_rgb, dummy_lang, dummy_prop),
        output_onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["rgb_image", "language_ids", "proprio_state"],
        output_names=["predicted_action_chunk"]
    )
    print(f"[Export] Successfully exported ONNX model to: {output_onnx_path}")
    
    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(output_onnx_path)
        onnx.checker.check_model(onnx_model)
        print("[Export] ONNX model structure verification: SUCCESS!")
    except Exception as e:
        print(f"[Export] ONNX verification note: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/home/HwHiAiUser/robot_ws/weights/best_edge_vla.pt")
    parser.add_argument("--output", type=str, default="/home/HwHiAiUser/robot_ws/weights/edge_vla.onnx")
    parser.add_argument("--variant", type=str, default="base")
    parser.add_argument("--chunk_size", type=int, default=8)
    args = parser.parse_args()
    
    export_to_onnx(args.checkpoint, args.output, args.variant, args.chunk_size)
