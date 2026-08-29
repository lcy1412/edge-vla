"""
Training script for EdgeVLA policy on Robot Demonstration Datasets.
Uses imitation learning (Behavioral Cloning) with L1/L2 Action Loss + Temporal Smoothness Regularization.
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.edge_vla import build_edge_vla
from datasets.robot_dataset import get_dataloader

def train_edge_vla(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Initializing EdgeVLA Training on Device: {device} | Variant: {args.variant}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 1. Dataset & DataLoader
    train_loader = get_dataloader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        chunk_size=args.action_chunk_size,
        num_samples=args.num_samples,
        is_synthetic=args.synthetic,
        shuffle=True
    )
    
    # 2. Build Model
    model = build_edge_vla(variant=args.variant, chunk_size=args.action_chunk_size)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] EdgeVLA Total Trainable Parameters: {total_params / 1e6:.2f} M")
    
    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # 4. Losses
    l1_loss_fn = nn.L1Loss()
    l2_loss_fn = nn.MSELoss()
    
    loss_history = []
    start_time = time.time()
    
    print(f"[Train] Starting {args.epochs} Training Epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_l1 = 0.0
        epoch_smoothness = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            rgb = batch["rgb"].to(device)
            token_ids = batch["token_ids"].to(device)
            proprio = batch["proprio"].to(device)
            gt_actions = batch["action_chunk"].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_actions = model(rgb, token_ids, proprio)
            
            # Action Reconstruction Loss
            l1_loss = l1_loss_fn(pred_actions, gt_actions)
            l2_loss = l2_loss_fn(pred_actions, gt_actions)
            
            # Temporal Smoothness Regularization between consecutive timesteps in chunk
            # \Delta A = A_{t+1} - A_t
            if args.action_chunk_size > 1:
                pred_diff = pred_actions[:, 1:, :] - pred_actions[:, :-1, :]
                gt_diff = gt_actions[:, 1:, :] - gt_actions[:, :-1, :]
                smooth_loss = l1_loss_fn(pred_diff, gt_diff)
            else:
                smooth_loss = torch.tensor(0.0, device=device)
                
            total_loss = l1_loss + 0.5 * l2_loss + 0.1 * smooth_loss
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_l1 += l1_loss.item()
            epoch_smoothness += smooth_loss.item()
            
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        avg_l1 = epoch_l1 / len(train_loader)
        loss_history.append(avg_loss)
        
        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            print(f"[Epoch {epoch:02d}/{args.epochs:02d}] Total Loss: {avg_loss:.4f} | Action L1 Error: {avg_l1:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    total_training_time = time.time() - start_time
    print(f"[Train] Training Completed in {total_training_time:.2f}s!")
    
    # Save Model Checkpoint
    checkpoint_path = os.path.join(args.save_dir, "best_edge_vla.pt")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "loss_history": loss_history
    }, checkpoint_path)
    print(f"[Train] Model checkpoint saved to: {checkpoint_path}")
    
    # Plot Training Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, 'b-o', label='Imitation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('EdgeVLA Training Convergence Curve')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plot_path = os.path.join(args.results_dir, "training_loss_curve.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Train] Training curve plot saved to: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EdgeVLA Policy")
    parser.add_argument("--variant", type=str, default="base", choices=["tiny", "base"])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true", default=True)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--action_chunk_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--save_dir", type=str, default="/home/HwHiAiUser/robot_ws/weights")
    parser.add_argument("--results_dir", type=str, default="/home/HwHiAiUser/robot_ws/results")
    
    args = parser.parse_args()
    train_edge_vla(args)
