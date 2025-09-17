# Phase 1.5: CUTLASS Integration Testing - Detailed Plan

## Overview

Phase 1.5 is a critical intermediate step to resolve CUTLASS compilation and integration issues before proceeding with AllReduce GEMM implementation. This phase focuses on validating that FlashInfer's JIT system can properly compile and execute CUTLASS kernels.

## Goals and Success Criteria

### Primary Goal
Resolve CUTLASS include and compilation issues by implementing a simple CUTLASS GEMM within FlashInfer's JIT system.

### Success Criteria
1. **Compilation Success**: CUTLASS headers compile without errors in FlashInfer JIT
2. **Template Instantiation**: Complex CUTLASS templates build successfully
3. **Functional GEMM**: Basic CUTLASS GEMM executes correctly
4. **Correctness Verification**: Results match PyTorch reference implementation
5. **Performance Baseline**: Reasonable performance compared to cuBLAS/PyTorch

## Phase 1.5 Implementation Plan

### Step 1: Environment Validation
**Duration**: 30 minutes
**Prerequisites**: Phase 1 environment must be functional

#### Tasks:
- [x] Verify virtual environment is working (`venv_test`)
- [x] Confirm CUTLASS submodules are initialized
- [x] Test Phase 1 stub still works as baseline
- [x] Document any environment changes needed

#### Verification Command:
```bash
cd /home/jason/repos/trtllm-flashinfer-dir/flashinfer
source venv_test/bin/activate
TORCH_CUDA_ARCH_LIST=8.9 python ./docs/allreduce_gemm/test_allreduce_gemm_phase1.py
```

### Step 2: CUTLASS Include Path Testing
**Duration**: 1-2 hours
**Goal**: Resolve basic CUTLASS header compilation

#### Tasks:
- [x] Create minimal CUTLASS test wrapper: `test_cutlass_basic_wrapper.cu`
- [x] Update JIT config to include CUTLASS paths
- [x] Test basic CUTLASS header includes (`cutlass/cutlass.h`)
- [x] Resolve namespace conflicts and template errors
- [x] Document working JIT configuration

#### Implementation:
```cpp
// File: csrc/test_cutlass_basic_wrapper.cu
#include <string>
#include "pytorch_extension_utils.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

void test_cutlass_basic(at::Tensor& A, at::Tensor& B, at::Tensor& D) {
    // Simple CUTLASS GEMM without AllReduce
    // Verify CUTLASS headers work in FlashInfer JIT environment
}
```

#### JIT Configuration:
```python
# File: flashinfer/comm/test_cutlass_basic.py
def gen_test_cutlass_basic_module() -> JitSpec:
    return gen_jit_spec(
        "test_cutlass_basic",
        [jit_env.FLASHINFER_CSRC_DIR / "test_cutlass_basic_wrapper.cu"],
        extra_include_paths=[
            jit_env.FLASHINFER_CSRC_DIR / "3rdparty" / "cutlass" / "include",
        ],
        extra_cuda_cflags=[
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA",
            "-DCUTLASS_NAMESPACE=cutlass",
        ],
        needs_device_linking=True
    )
```

### Step 3: Basic CUTLASS GEMM Implementation
**Duration**: 2-3 hours
**Goal**: Implement working CUTLASS GEMM kernel

#### Tasks:
- [x] Choose appropriate CUTLASS GEMM template for RTX 4090 (SM89)
- [x] Implement basic F16 CUTLASS GEMM
- [x] Handle memory layout conversions (PyTorch to CUTLASS)
- [x] Test kernel execution without errors
- [x] Compare output shapes and basic functionality

#### CUTLASS GEMM Selection:
```cpp
// Target CUTLASS GEMM configuration for SM89
using GemmKernel = cutlass::gemm::device::Gemm<
    cutlass::half_t,         // ElementA
    cutlass::layout::RowMajor,    // LayoutA
    cutlass::half_t,         // ElementB
    cutlass::layout::ColumnMajor, // LayoutB
    cutlass::half_t,         // ElementC
    cutlass::layout::RowMajor,    // LayoutC
    float,                   // ElementAccumulator
    cutlass::arch::OpClassTensorOp, // OpClass
    cutlass::arch::Sm80      // ArchTag (compatible with SM89)
>;
```

### Step 4: Correctness and Performance Validation
**Duration**: 1-2 hours
**Goal**: Verify CUTLASS GEMM produces correct results

#### Tasks:
- [x] Create test script: `test_cutlass_basic_phase1_5.py`
- [x] Compare CUTLASS output vs PyTorch `torch.mm()`
- [x] Measure performance vs PyTorch baseline
- [x] Test multiple tensor sizes and dtypes
- [x] Document performance characteristics

#### Test Script Template:
```python
#!/usr/bin/env python3
"""Phase 1.5 CUTLASS Basic GEMM Test"""

def test_cutlass_basic_gemm():
    print("=== Phase 1.5 CUTLASS Basic GEMM Test ===")

    # Test 1: Import and compilation
    import flashinfer.comm.test_cutlass_basic as cutlass_test

    # Test 2: Basic GEMM execution
    A = torch.randn(512, 512, dtype=torch.float16, device='cuda')
    B = torch.randn(512, 512, dtype=torch.float16, device='cuda')
    D = torch.empty(512, 512, dtype=torch.float16, device='cuda')

    cutlass_test.cutlass_basic_gemm(A, B, D)

    # Test 3: Correctness verification
    expected = torch.mm(A, B)
    if torch.allclose(D, expected, rtol=1e-3, atol=1e-3):
        print("✅ CUTLASS GEMM correctness verified")
    else:
        print(f"❌ Correctness failed: max diff = {torch.max(torch.abs(D - expected)).item()}")

    # Test 4: Performance comparison
    # Basic timing comparison vs PyTorch

if __name__ == "__main__":
    test_cutlass_basic_gemm()
```

### Step 5: Integration with TensorRT-LLM Extensions
**Duration**: 2-3 hours
**Goal**: Test FlashInfer's TensorRT-LLM CUTLASS extensions

#### Tasks:
- [x] Include TensorRT-LLM CUTLASS extension paths
- [x] Test compilation with `cutlass_extensions` headers
- [x] Verify no conflicts between FlashInfer and TensorRT-LLM CUTLASS versions
- [x] Document any version compatibility issues

#### Extended JIT Configuration:
```python
extra_include_paths=[
    jit_env.FLASHINFER_CSRC_DIR / "3rdparty" / "cutlass" / "include",
    jit_env.FLASHINFER_CSRC_DIR / "nv_internal" / "tensorrt_llm" / "cutlass_extensions" / "include",
],
extra_cuda_cflags=[
    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA",
    "-DCUTLASS_NAMESPACE=cutlass",
    "-DCUTE_ARCH_MULTIMEM_SM90_ENABLED",  # TensorRT-LLM specific
],
```

### Step 6: Documentation and Handoff Preparation
**Duration**: 1 hour
**Goal**: Prepare comprehensive handoff documentation

#### Tasks:
- [x] Document all working configurations
- [x] Create troubleshooting guide for common CUTLASS issues
- [x] Update Phase 2 plan based on Phase 1.5 findings
- [x] Prepare context reset summary

## Risk Assessment and Mitigation

### High Risk Issues

1. **CUTLASS Version Incompatibility**
   - **Risk**: FlashInfer CUTLASS != TensorRT-LLM CUTLASS version
   - **Mitigation**: Test with minimal CUTLASS first, then add TensorRT-LLM extensions
   - **Fallback**: Use only core CUTLASS features

2. **Template Compilation Complexity**
   - **Risk**: CUTLASS templates cause compilation errors
   - **Mitigation**: Start with simplest CUTLASS GEMM templates
   - **Fallback**: Use runtime CUTLASS API instead of template API

3. **Memory Layout Conflicts**
   - **Risk**: PyTorch tensor layout != CUTLASS expected layout
   - **Mitigation**: Add explicit tensor transformation utilities
   - **Fallback**: Use CUTLASS layout adapters

### Medium Risk Issues

1. **Architecture Compatibility**
   - **Risk**: CUTLASS kernel not compatible with SM89 (RTX 4090)
   - **Mitigation**: Test SM80 kernels which should be compatible
   - **Detection**: Check CUTLASS architecture support matrix

2. **JIT Compilation Performance**
   - **Risk**: CUTLASS JIT compilation too slow
   - **Mitigation**: Profile compilation times, consider caching
   - **Monitoring**: Track JIT compilation duration

## Expected Outcomes

### Success Path
- **CUTLASS Integration**: Working CUTLASS GEMM in FlashInfer JIT
- **Performance Baseline**: CUTLASS performance comparable to PyTorch
- **Documentation**: Clear guide for Phase 2 AllReduce integration
- **Foundation**: Solid base for AllReduce GEMM implementation

### Partial Success Path
- **Basic CUTLASS**: Simple GEMM works, complex templates problematic
- **Workaround Documentation**: Known issues and solutions documented
- **Phase 2 Adjustment**: Modified approach based on limitations found

### Failure Path
- **CUTLASS Incompatibility**: Fundamental incompatibility discovered
- **Alternative Strategy**: Consider different GEMM library or approach
- **Escalation Path**: Involve FlashInfer maintainers for guidance

## Next Phase Preparation

### Phase 2 Prerequisites (after Phase 1.5)
1. **Working CUTLASS GEMM**: Verified functional CUTLASS kernel
2. **TensorRT-LLM Extensions**: Successful integration of TensorRT-LLM CUTLASS extensions
3. **Performance Baseline**: Documented CUTLASS GEMM performance
4. **Source File Access**: Located TensorRT-LLM AllReduce GEMM source files

### Phase 2 Scope Adjustment
Based on Phase 1.5 findings, Phase 2 may need to be adjusted:
- **Full CUTLASS**: If Phase 1.5 fully succeeds
- **Simplified CUTLASS**: If complex templates are problematic
- **Alternative Approach**: If CUTLASS integration proves too complex

## Timeline Estimate

| Step | Duration | Dependencies |
|------|----------|-------------|
| 1. Environment Validation | 30 min | Phase 1 complete |
| 2. CUTLASS Include Testing | 1-2 hours | Environment ready |
| 3. Basic GEMM Implementation | 2-3 hours | Includes working |
| 4. Correctness Validation | 1-2 hours | GEMM implemented |
| 5. TensorRT-LLM Integration | 2-3 hours | Basic GEMM working |
| 6. Documentation | 1 hour | All steps complete |

**Total Estimated Time**: 7.5-11.5 hours over 1-2 days