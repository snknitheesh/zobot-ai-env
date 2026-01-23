#!/bin/bash
# Script to display key installed packages

# Get versions
PYTHON_VER=$(python3 --version 2>&1 | awk '{print $2}')
CONDA_VER=$(conda --version 2>&1 | awk '{print $2}')
PKL_VER=$(pkl --version 2>&1 | head -1)
PIXI_VER=$(pixi --version 2>&1 | head -1 | awk '{print $2}')
UV_VER=$(uv --version 2>&1 | awk '{print $2}')
CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | sed 's/,//' || echo "13.0.2")
TORCH_VER=$(pip3 show torch 2>/dev/null | grep "^Version:" | awk '{print $2}')
TORCHVISION_VER=$(pip3 show torchvision 2>/dev/null | grep "^Version:" | awk '{print $2}')
TF_VER=$(pip3 show tensorflow 2>/dev/null | grep "^Version:" | awk '{print $2}')
NUMPY_VER=$(pip3 show numpy 2>/dev/null | grep "^Version:" | awk '{print $2}')
SCIPY_VER=$(pip3 show scipy 2>/dev/null | grep "^Version:" | awk '{print $2}')

echo "=========================================="
echo "== SYSTEM INFO =="
echo "=========================================="
echo "Python:          ${PYTHON_VER}"
echo "Conda:           ${CONDA_VER}"
echo "Pixi:            ${PIXI_VER}"
echo "UV:              ${UV_VER}"
echo "PKL:             ${PKL_VER}"
echo "CUDA:            ${CUDA_VER}"
echo ""
echo "PyTorch:         ${TORCH_VER}"
echo "TorchVision:     ${TORCHVISION_VER}"
echo "TensorFlow:      ${TF_VER}"
echo "NumPy:           ${NUMPY_VER}"
echo "SciPy:           ${SCIPY_VER}"
echo ""
echo "ROS Distro:      ${ROS_DISTRO:-jazzy}"
echo "RMW:             ${RMW_IMPLEMENTATION:-rmw_zenoh_cpp}"
echo ""
echo "=========================================="
echo ""
