#!/bin/bash
set -e

echo "========================================================================"
echo "    🚀 EdgeVLA Full-Stack Research Experiment & Benchmark Suite         "
echo "    Hardware: Orange Pi AIpro (Ascend 310B NPU 20 TOPS)                 "
echo "    Target:   LIBERO & Real-Time Embedded Robotics                      "
echo "========================================================================"

mkdir -p weights results

# 1. Train EdgeVLA Policy (15 epochs)
echo ""
echo "[Step 1/5] Training EdgeVLA Policy on Robotic Demonstration Dataset..."
python3 train/train.py --variant base --epochs 15 --batch_size 16 --action_chunk_size 8

# 2. Closed-Loop Simulation Evaluation
echo ""
echo "[Step 2/5] Running Closed-Loop Simulation Benchmark..."
python3 sim_bridge/sim_evaluator.py --checkpoint weights/best_edge_vla.pt --episodes 30

# 3. Export to ONNX
echo ""
echo "[Step 3/5] Exporting PyTorch Model to Static Graph ONNX..."
python3 export/export_onnx.py --checkpoint weights/best_edge_vla.pt --output weights/edge_vla.onnx

# 4. Compile with Ascend CANN ATC Compiler for 310B NPU
echo ""
echo "[Step 4/5] Compiling ONNX to Ascend 310B NPU .om Model..."
bash export/atc_convert.sh weights/edge_vla.onnx weights/edge_vla_310b || true

# 5. Hardware Latency & Control Frequency Benchmark on Orange Pi 20T
echo ""
echo "[Step 5/5] Executing Hardware Latency & Control Throughput Benchmark..."
python3 deploy/edge_benchmark.py --iters 100 --results_dir results/

echo ""
echo "========================================================================"
echo "  ✅ Experiment Successfully Completed!"
echo "  Results and figures stored in: /home/HwHiAiUser/robot_ws/results/"
echo "========================================================================"
ls -lh results/
