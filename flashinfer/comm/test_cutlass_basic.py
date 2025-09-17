"""
Phase 1.5 CUTLASS Basic Integration Test

Test basic CUTLASS header compilation and integration with FlashInfer JIT system.
"""

import functools
from types import SimpleNamespace

import torch

from ..jit import JitSpec
from ..jit import env as jit_env
from ..jit import gen_jit_spec
from ..utils import register_custom_op, register_fake_op


def gen_test_cutlass_basic_module() -> JitSpec:
    """Generate JIT specification for basic CUTLASS test module."""
    # Phase 1.5: Start with minimal CUTLASS configuration
    return gen_jit_spec(
        "test_cutlass_basic",
        [
            jit_env.FLASHINFER_CSRC_DIR / "test_cutlass_basic_wrapper.cu",
        ],
        extra_include_paths=[
            # Basic CUTLASS includes - test if this works
            jit_env.FLASHINFER_CSRC_DIR / "3rdparty" / "cutlass" / "include",
        ],
        extra_cuda_cflags=[
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA",
            "-DCUTLASS_NAMESPACE=cutlass",
        ],
        # Phase 1.5: Test if device linking is needed for CUTLASS
        needs_device_linking=True
    )


@functools.cache
def get_test_cutlass_basic_module():
    """Get cached basic CUTLASS test module."""
    module = gen_test_cutlass_basic_module().build_and_load()

    @register_custom_op(
        "flashinfer::test_cutlass_basic",
        mutates_args=["A", "B", "D"]
    )
    def test_cutlass_basic(
        A: torch.Tensor,
        B: torch.Tensor,
        D: torch.Tensor,
    ):
        return module.test_cutlass_basic(A, B, D)

    @register_fake_op("flashinfer::test_cutlass_basic")
    def test_cutlass_basic_fake(
        A: torch.Tensor,
        B: torch.Tensor,
        D: torch.Tensor,
    ):
        pass

    return SimpleNamespace(
        test_cutlass_basic=test_cutlass_basic,
    )


def cutlass_basic_test(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Phase 1.5 basic CUTLASS test function.

    Tests that CUTLASS headers can be compiled and basic CUTLASS types work.
    This is NOT a real GEMM, just a compilation and basic functionality test.
    """
    # Input validation
    if A.device != B.device:
        raise ValueError("A and B must be on the same device")
    if not A.is_cuda or not B.is_cuda:
        raise ValueError("A and B must be CUDA tensors")
    if A.dtype != B.dtype:
        raise ValueError("A and B must have the same dtype")

    # Check dimensions
    M, K = A.shape[-2:]
    K2, N = B.shape[-2:]
    if K != K2:
        raise ValueError(f"Matrix dimensions incompatible: A[..., {M}, {K}] vs B[..., {K2}, {N}]")

    # Create output tensor
    D = torch.empty(M, N, dtype=A.dtype, device=A.device)

    # Get module and call test function
    module = get_test_cutlass_basic_module()
    module.test_cutlass_basic(A, B, D)

    return D