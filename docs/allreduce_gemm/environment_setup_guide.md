# Environment Setup Guide - AllReduce GEMM Implementation

## Overview

This document provides complete environment setup instructions for the AllReduce GEMM implementation project. Follow these steps to reproduce the working development environment.

## Prerequisites

### System Requirements
**Phase 1 & 1.5 (Development)**:
- **GPU**: NVIDIA RTX 4090 (SM89) or similar for development and testing
- **CUDA**: Version 12.1+
- **OS**: Linux (tested on Ubuntu)
- **Python**: 3.12+ recommended

**Phase 2 (Production)**:
- **GPU**: 2+ NVIDIA H100/H200 (SM90+) for full AllReduce GEMM functionality
- **NVLS**: NVLink Sharp support required for optimal performance
- **Environment**: Docker container or native Linux with CUDA runtime

### Required Tools
- `git` with submodule support
- `python3` and virtual environment support (`venv` or `conda`)
- `ninja-build` (for JIT compilation)
- NVIDIA CUDA Toolkit

## Environment Setup Options

### Option 1: Native Linux Environment

#### Repository Setup
```bash
# Clone the repository (if not already done)
git clone https://github.com/your-username/flashinfer.git
cd flashinfer
```

#### Initialize Git Submodules
**Critical Step**: CUTLASS and other dependencies are git submodules that must be initialized:
```bash
git submodule update --init --recursive
```

**Verification**:
```bash
# These directories should now contain files:
ls -la ./3rdparty/cutlass/include/cutlass/ | head -5
ls -la ./3rdparty/spdlog/include/
```

#### Create Python Virtual Environment
```bash
# Using venv (recommended)
python -m venv venv_allreduce
source venv_allreduce/bin/activate

# OR using conda
conda create -n allreduce python=3.12
conda activate allreduce
```

### Option 2: Docker Environment (Recommended for Phase 2)

#### Docker Setup for SM90+ Hardware
For Phase 2 development on H100/H200 systems, Docker provides a consistent environment:

```dockerfile
# Example Dockerfile for Phase 2 development
FROM nvcr.io/nvidia/pytorch:24.08-py3

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Clone and setup FlashInfer
COPY . /workspace/flashinfer/
WORKDIR /workspace/flashinfer

# Initialize submodules
RUN git submodule update --init --recursive

# Install FlashInfer in development mode
RUN pip install -e .

# Set CUDA architecture for Phase 2 (SM90)
ENV TORCH_CUDA_ARCH_LIST=9.0
```

#### Run Docker Container
```bash
# For multi-GPU Phase 2 development
docker run --gpus all -it --rm \
    -v $(pwd):/workspace/flashinfer \
    your-flashinfer-image:latest

# Inside container, verify environment
cd /workspace/flashinfer
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

#### Activate Virtual Environment
```bash
source venv_test/bin/activate
```

**Verify activation**:
```bash
python --version  # Should show: Python 3.12.3
which python      # Should show: /home/jason/repos/.../venv_test/bin/python
```

### 3. Python Dependencies Installation

#### Install PyTorch with CUDA Support
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Critical**: Use CUDA 12.1 index for compatibility.

#### Clear any problematic pip configurations
If you encounter CodeArtifact or other index issues:
```bash
pip config unset global.index-url 2>/dev/null || true
pip config list  # Should show minimal configuration
```

#### Install FlashInfer in Development Mode
```bash
PIP_INDEX_URL=https://pypi.org/simple/ pip install -e .
```

**Installation verification**:
```bash
python -c "import flashinfer; print('FlashInfer imported successfully')"
```

### 4. Environment Verification

#### Test Phase 1 Implementation
Run the complete test to verify environment is working:
```bash
TORCH_CUDA_ARCH_LIST=8.9 python ./docs/allreduce_gemm/test_allreduce_gemm_phase1.py
```

**Expected output**:
```
=== Phase 1 AllReduce GEMM Test ===
1. Testing module import...                    ✅ Module import successful
2. Checking CUDA availability...               ✅ CUDA available, device: NVIDIA GeForce RTX 4090
3. Creating test tensors...                    ✅ Test tensors created: A[512, 512], B[512, 512]
4. Testing basic interface (single GPU)...     ✅ Basic interface test passed: output shape [512, 512]
5. Verifying result correctness...             ⚠️  Result differs from expected (expected for Phase 1 stub)
6. Testing workspace allocation...             ✅ Workspace allocated: 1605640 bytes
7. Testing with explicit workspace...          ✅ Explicit workspace test passed

🎉 Phase 1 tests completed successfully!
```

## Environment Configuration Details

### Key Environment Variables

#### TORCH_CUDA_ARCH_LIST
**Purpose**: Specifies CUDA architectures for compilation
**Required Value**: `8.9` (for RTX 4090 SM89 architecture)
**Usage**: `TORCH_CUDA_ARCH_LIST=8.9` before Python commands

#### PIP_INDEX_URL
**Purpose**: Ensures PyPI is used instead of internal repositories
**Required Value**: `https://pypi.org/simple/`
**Usage**: `PIP_INDEX_URL=https://pypi.org/simple/ pip install ...`

### JIT Cache Management

FlashInfer uses a JIT compilation cache that sometimes needs clearing:
```bash
# Clear JIT cache if needed
rm -rf /home/jason/.cache/flashinfer/
```

**When to clear cache**:
- After significant code changes to CUDA files
- When compilation errors persist after fixes
- When switching between different CUDA architectures

### Directory Structure Verification

After setup, verify this directory structure exists:
```
/home/jason/repos/trtllm-flashinfer-dir/flashinfer/
├── venv_test/                           # Python virtual environment
├── 3rdparty/
│   ├── cutlass/include/cutlass/         # CUTLASS headers (should have files)
│   └── spdlog/include/                  # spdlog headers (should have files)
├── flashinfer/comm/
│   └── trtllm_allreduce_gemm.py        # Python JIT interface
├── csrc/
│   └── trtllm_allreduce_gemm_wrapper.cu # CUDA implementation
├── docs/allreduce_gemm/
│   ├── test_allreduce_gemm_phase1.py   # Working test script
│   └── phase1_5_cutlass_integration_plan.md
└── csrc/nv_internal/tensorrt_llm/cutlass_extensions/include/ # TensorRT-LLM extensions
```

## Troubleshooting Common Issues

### 1. Git Submodule Issues
**Problem**: Empty `3rdparty/cutlass/` directory
**Solution**:
```bash
git submodule update --init --recursive
# If still empty, try:
git submodule update --force --recursive
```

### 2. PyTorch Installation Issues
**Problem**: CUDA not available after PyTorch installation
**Solution**: Reinstall with explicit CUDA version:
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 3. FlashInfer Compilation Issues
**Problem**: Build fails with permission or index errors
**Solution**: Use explicit PyPI index:
```bash
PIP_INDEX_URL=https://pypi.org/simple/ pip install -e .
```

### 4. JIT Compilation Failures
**Problem**: Ninja build failures or template errors
**Solution**: Clear cache and retry:
```bash
rm -rf /home/jason/.cache/flashinfer/
TORCH_CUDA_ARCH_LIST=8.9 python [your_test_script]
```

### 5. CUDA Architecture Mismatches
**Problem**: Architecture not supported errors
**Solution**: Verify and set correct architecture:
```bash
# Check your GPU architecture
nvidia-smi --query-gpu=name,compute_cap --format=csv

# For RTX 4090 (compute capability 8.9):
export TORCH_CUDA_ARCH_LIST=8.9
```

## Phase-Specific Environment Notes

### Phase 1 (Complete)
- Virtual environment with PyTorch CUDA support ✅
- Basic FlashInfer installation ✅
- Simple stub implementation working ✅

### Phase 1.5 (Complete)
- CUTLASS submodules initialized ✅
- CUTLASS template compilation working ✅
- TensorRT-LLM extensions integration successful ✅

### Phase 2 (Future)
- TensorRT-LLM source files integration
- Complex CUTLASS extension compilation
- Multi-GPU testing environment (if available)

## Environment Persistence

### Activation Script
Create a convenience script for easy environment activation:
```bash
# File: activate_env.sh
#!/bin/bash
cd /home/jason/repos/trtllm-flashinfer-dir/flashinfer
source venv_test/bin/activate
export TORCH_CUDA_ARCH_LIST=8.9
export PIP_INDEX_URL=https://pypi.org/simple/
echo "AllReduce GEMM development environment activated"
echo "Current directory: $(pwd)"
echo "Python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
```

Usage:
```bash
chmod +x activate_env.sh
source activate_env.sh
```

## Verification Checklist

Before proceeding with Phase 1.5, verify:
- [ ] Virtual environment activated and working
- [ ] Git submodules initialized (CUTLASS files present)
- [ ] PyTorch with CUDA support installed
- [ ] FlashInfer installed in development mode
- [ ] Phase 1 test passes completely
- [ ] JIT compilation cache clear
- [ ] Correct CUDA architecture set (8.9 for RTX 4090)

## Hardware Information

### Current Test Environment
- **GPU**: NVIDIA GeForce RTX 4090
- **Compute Capability**: 8.9 (SM89 architecture)
- **Memory**: Sufficient for test tensors up to 512x512 FP16
- **CUDA Version**: 12.1 compatible
- **Driver**: Compatible with CUDA 12.1

### Multi-GPU Considerations (Phase 3)
- Current setup is single GPU
- Multi-GPU testing will require additional setup
- NVLS features require SM90+ hardware (H100+)
- RTX 4090 can test basic multi-GPU without NVLS

---
*Environment guide created: 2025-09-17*
*Verified with Phase 1 implementation*