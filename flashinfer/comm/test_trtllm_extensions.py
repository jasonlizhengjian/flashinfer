"""
Phase 1.5 TensorRT-LLM CUTLASS Extensions Test

Test TensorRT-LLM CUTLASS extensions compilation and integration.
"""

import functools
from types import SimpleNamespace

import torch

from ..jit import JitSpec
from ..jit import env as jit_env
from ..jit import gen_jit_spec
from ..utils import register_custom_op, register_fake_op


def gen_test_trtllm_extensions_module() -> JitSpec:
    """Generate JIT specification for TensorRT-LLM extensions test module."""
    return gen_jit_spec(
        "test_trtllm_extensions",
        [
            jit_env.FLASHINFER_CSRC_DIR / "test_trtllm_extensions_wrapper.cu",
        ],
        extra_include_paths=[
            # Standard CUTLASS includes
            jit_env.FLASHINFER_CSRC_DIR / "3rdparty" / "cutlass" / "include",
            # TensorRT-LLM CUTLASS extensions
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal" / "tensorrt_llm" / "cutlass_extensions" / "include",
        ],
        extra_cuda_cflags=[
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA",
            "-DCUTLASS_NAMESPACE=cutlass",
            # TensorRT-LLM specific flags
            "-DCUTE_ARCH_MULTIMEM_SM90_ENABLED",
        ],
        needs_device_linking=True
    )


@functools.cache
def get_test_trtllm_extensions_module():
    """Get cached TensorRT-LLM extensions test module."""
    module = gen_test_trtllm_extensions_module().build_and_load()

    @register_custom_op(
        "flashinfer::test_trtllm_extensions",
        mutates_args=["A", "B", "D"]
    )
    def test_trtllm_extensions(
        A: torch.Tensor,
        B: torch.Tensor,
        D: torch.Tensor,
    ):
        return module.test_trtllm_extensions(A, B, D)

    @register_fake_op("flashinfer::test_trtllm_extensions")
    def test_trtllm_extensions_fake(
        A: torch.Tensor,
        B: torch.Tensor,
        D: torch.Tensor,
    ):
        pass

    return SimpleNamespace(
        test_trtllm_extensions=test_trtllm_extensions,
    )


def trtllm_extensions_test(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Phase 1.5 TensorRT-LLM CUTLASS extensions test function.

    Tests that TensorRT-LLM CUTLASS extensions can be compiled and accessed.
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
    module = get_test_trtllm_extensions_module()
    module.test_trtllm_extensions(A, B, D)

    return D