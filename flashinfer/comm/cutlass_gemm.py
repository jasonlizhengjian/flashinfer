"""
Phase 1.5 CUTLASS GEMM Implementation

Real CUTLASS GEMM kernel for testing integration before AllReduce implementation.
"""

import functools
from types import SimpleNamespace

import torch

from ..jit import JitSpec
from ..jit import env as jit_env
from ..jit import gen_jit_spec
from ..utils import register_custom_op, register_fake_op


def gen_cutlass_gemm_module() -> JitSpec:
    """Generate JIT specification for CUTLASS GEMM module."""
    return gen_jit_spec(
        "cutlass_gemm",
        [
            jit_env.FLASHINFER_CSRC_DIR / "cutlass_gemm_wrapper.cu",
        ],
        extra_include_paths=[
            # CUTLASS includes
            jit_env.FLASHINFER_CSRC_DIR / "3rdparty" / "cutlass" / "include",
            jit_env.FLASHINFER_CSRC_DIR / "3rdparty" / "cutlass" / "tools" / "util" / "include",
        ],
        extra_cuda_cflags=[
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA",
            "-DCUTLASS_NAMESPACE=cutlass",
        ],
        # CUTLASS templates require device linking
        needs_device_linking=True
    )


@functools.cache
def get_cutlass_gemm_module():
    """Get cached CUTLASS GEMM module."""
    module = gen_cutlass_gemm_module().build_and_load()

    @register_custom_op(
        "flashinfer::cutlass_gemm",
        mutates_args=["A", "B", "C", "D"]
    )
    def cutlass_gemm(
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        alpha: float,
        beta: float,
    ):
        return module.cutlass_gemm(A, B, C, D, alpha, beta)

    @register_fake_op("flashinfer::cutlass_gemm")
    def cutlass_gemm_fake(
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        alpha: float,
        beta: float,
    ):
        pass

    return SimpleNamespace(
        cutlass_gemm=cutlass_gemm,
    )


def cutlass_gemm_test(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor = None,
    alpha: float = 1.0,
    beta: float = 0.0
) -> torch.Tensor:
    """
    Phase 1.5 CUTLASS GEMM test function.

    Performs D = alpha * A @ B + beta * C using CUTLASS.

    Args:
        A: Input matrix A [M, K] (float16)
        B: Input matrix B [K, N] (float16)
        C: Optional bias matrix C [M, N] (float16)
        alpha: Scalar multiplier for A @ B
        beta: Scalar multiplier for C

    Returns:
        D: Output matrix [M, N] (float16)
    """
    # Input validation
    if A.device != B.device:
        raise ValueError("A and B must be on the same device")
    if not A.is_cuda or not B.is_cuda:
        raise ValueError("A and B must be CUDA tensors")
    if A.dtype != torch.float16 or B.dtype != torch.float16:
        raise ValueError("A and B must be float16 tensors")

    # Check dimensions
    M, K = A.shape[-2:]
    K2, N = B.shape[-2:]
    if K != K2:
        raise ValueError(f"Matrix dimensions incompatible: A[..., {M}, {K}] vs B[..., {K2}, {N}]")

    # Handle bias tensor
    if C is None:
        C = torch.empty(0, dtype=A.dtype, device=A.device)  # Empty tensor
        beta = 0.0
    else:
        if C.device != A.device or C.dtype != A.dtype:
            raise ValueError("C must be on same device and dtype as A")
        if C.shape != (M, N):
            raise ValueError(f"C shape {C.shape} must match output shape ({M}, {N})")

    # Create output tensor
    D = torch.empty(M, N, dtype=A.dtype, device=A.device)

    # Get module and call CUTLASS GEMM
    module = get_cutlass_gemm_module()
    module.cutlass_gemm(A, B, C, D, alpha, beta)

    return D