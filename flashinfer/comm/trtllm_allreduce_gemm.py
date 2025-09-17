"""
TensorRT-LLM CUTLASS AllReduce GEMM operations for FlashInfer.

This module provides fused GEMM + AllReduce operations using NVLS (NVLink Sharp)
hardware acceleration for multi-GPU communication.
"""

import functools
import math
from types import SimpleNamespace
from typing import Optional, Tuple

import torch

from ..jit import JitSpec
from ..jit import env as jit_env
from ..jit import gen_jit_spec
from ..utils import register_custom_op, register_fake_op


def gen_trtllm_allreduce_gemm_module() -> JitSpec:
    """Generate JIT specification for AllReduce GEMM module."""
    # Phase 1: Minimal JIT spec without CUTLASS dependencies
    return gen_jit_spec(
        "trtllm_allreduce_gemm",
        [
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_allreduce_gemm_wrapper.cu",
        ],
        extra_include_paths=[
            # Minimal includes for Phase 1 stub
        ],
        extra_cuda_cflags=[
            "-DFLASHINFER_ALLREDUCE_GEMM_ENABLED",
            # TODO Phase 2: Add CUTLASS flags back
            # "-DCUTE_ARCH_MULTIMEM_SM90_ENABLED",
            # "-DCUTLASS_ENABLE_TENSOR_CORE_MMA",
            # "-DCUTLASS_NAMESPACE=cutlass",
        ],
        # Phase 1: No device linking needed
        needs_device_linking=False
    )


@functools.cache
def get_trtllm_allreduce_gemm_module():
    """Get cached AllReduce GEMM module with custom operators."""
    module = gen_trtllm_allreduce_gemm_module().build_and_load()

    @register_custom_op(
        "flashinfer::trtllm_allreduce_gemm",
        mutates_args=["A", "B", "C", "D", "workspace"]
    )
    def trtllm_allreduce_gemm(
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        workspace: torch.Tensor,
        rank: int,
        world_size: int,
        alpha: float = 1.0,
        beta: float = 0.0
    ) -> None:
        """
        Fused GEMM + AllReduce operation using NVLS hardware acceleration.

        Args:
            A: Input tensor A [M, K]
            B: Input tensor B [K, N]
            C: Input tensor C (bias) [M, N]
            D: Output tensor D [M, N]
            workspace: Pre-allocated workspace tensor
            rank: Current rank in communicator
            world_size: Total number of ranks
            alpha, beta: GEMM scaling factors
        """
        return module.trtllm_allreduce_gemm(
            A, B, C, D, workspace, rank, world_size, alpha, beta
        )

    @register_fake_op("flashinfer::trtllm_allreduce_gemm")
    def _fake_trtllm_allreduce_gemm(
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        workspace: torch.Tensor,
        rank: int,
        world_size: int,
        alpha: float = 1.0,
        beta: float = 0.0
    ) -> None:
        """Fake implementation for tracing/symbolic execution."""
        pass

    return SimpleNamespace(
        trtllm_allreduce_gemm=trtllm_allreduce_gemm
    )


def get_allreduce_gemm_workspace(
    problem_shape: Tuple[int, int, int],
    dtype: torch.dtype,
    world_size: int
) -> torch.Tensor:
    """
    Allocate workspace for AllReduce GEMM operation.

    Args:
        problem_shape: (M, N, K) dimensions
        dtype: Element data type
        world_size: Number of participating ranks

    Returns:
        workspace: Tensor containing communication buffers and metadata
    """
    M, N, K = problem_shape
    element_size = dtype.itemsize

    # Calculate buffer requirements
    gemm_output_size = M * N * element_size
    workspace_size = (
        gemm_output_size * 3 +  # Triple buffering for pipelined communication
        world_size * 8 +        # IPC handle pointers
        1024 * 32              # Barrier flags (32 bytes per tile)
    )

    # Allocate workspace tensor
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device='cuda')
    return workspace


def allreduce_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    rank: int = 0,
    world_size: int = 1,
    alpha: float = 1.0,
    beta: float = 0.0
) -> torch.Tensor:
    """
    Fused GEMM + AllReduce operation using NVLS hardware acceleration.

    Performs: D = alpha * (A @ B) + beta * C, then AllReduce(D) across ranks

    Args:
        A: Input tensor A [M, K]
        B: Input tensor B [K, N]
        C: Optional bias tensor [M, N]
        workspace: Pre-allocated workspace for communication buffers
        rank: Current rank in the communicator
        world_size: Total number of ranks
        alpha, beta: GEMM scaling factors

    Returns:
        Output tensor D after GEMM + AllReduce [M, N]
    """
    # Input validation
    if not A.device.type == 'cuda':
        raise ValueError("Input tensors must be on CUDA device")
    if A.dtype != B.dtype:
        raise ValueError("A and B must have same dtype")
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"GEMM dimension mismatch: A{list(A.shape)} vs B{list(B.shape)}")

    M, K = A.shape
    K2, N = B.shape

    # Setup workspace if not provided
    if workspace is None:
        workspace = get_allreduce_gemm_workspace((M, N, K), A.dtype, world_size)

    # Allocate output tensor
    D = torch.empty((M, N), dtype=A.dtype, device=A.device)

    # Setup bias tensor
    if C is None:
        C = torch.zeros_like(D)

    # Get module and execute fused operation
    module = get_trtllm_allreduce_gemm_module()
    module.trtllm_allreduce_gemm(
        A, B, C, D, workspace, rank, world_size, alpha, beta
    )

    return D