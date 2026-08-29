#!/bin/bash
# Ascend CANN ATC (Architecture Translation Compiler) conversion script for Orange Pi AIpro 20T (Ascend 310B NPU)

ONNX_MODEL=${1:-"/home/HwHiAiUser/robot_ws/weights/edge_vla.onnx"}
OM_MODEL=${2:-"/home/HwHiAiUser/robot_ws/weights/edge_vla_310b"}
SOC_VERSION=${3:-"Ascend310B4"}

echo "================================================================="
echo " [Ascend ATC] Compiling EdgeVLA ONNX Model to Ascend NPU .om Model"
echo " Input ONNX: $ONNX_MODEL"
echo " Output OM:  $OM_MODEL.om"
echo " Target SoC: $SOC_VERSION"
echo "================================================================="

# Source CANN environment if available
if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

ATC_BIN="/usr/local/Ascend/ascend-toolkit/latest/bin/atc"

if [ ! -f "$ATC_BIN" ]; then
    echo "[Error] ATC compiler not found at $ATC_BIN"
    exit 1
fi

$ATC_BIN \
    --model="$ONNX_MODEL" \
    --framework=5 \
    --output="$OM_MODEL" \
    --soc_version="$SOC_VERSION" \
    --input_format=ND \
    --input_shape="rgb_image:1,3,224,224;language_ids:1,16;proprio_state:1,7" \
    --log=error

if [ $? -eq 0 ]; then
    echo "================================================================="
    echo " [Ascend ATC] SUCCESS! Compiled offline model: ${OM_MODEL}.om"
    echo "================================================================="
else
    echo "================================================================="
    echo " [Ascend ATC] Note: ATC conversion exited with status $?"
    echo "================================================================="
fi
