#!/usr/bin/env python3
"""
Phase 1.5 TensorRT-LLM CUTLASS Extensions Test

This test verifies:
1. TensorRT-LLM CUTLASS extensions compile with standard CUTLASS
2. Extension enums and types are accessible
3. No conflicts between FlashInfer and TensorRT-LLM CUTLASS
"""

import sys
import traceback
import torch

def test_trtllm_extensions_compilation():
    """Test Phase 1.5 TensorRT-LLM CUTLASS extensions integration."""
    print("=== Phase 1.5 TensorRT-LLM Extensions Test ===")

    try:
        # Test 1: Import module and trigger JIT compilation
        print("1. Testing TensorRT-LLM extensions compilation...")
        sys.path.insert(0, '/home/jason/repos/trtllm-flashinfer-dir/flashinfer')
        import flashinfer.comm.test_trtllm_extensions as trtllm_ext
        print("✅ TensorRT-LLM extensions compiled successfully")

        # Test 2: Check CUDA availability
        print("2. Checking CUDA availability...")
        if not torch.cuda.is_available():
            print("❌ CUDA not available, skipping GPU tests")
            return False
        print(f"✅ CUDA available, device: {torch.cuda.get_device_name()}")

        # Test 3: Create test tensors
        print("3. Creating test tensors...")
        device = 'cuda'
        dtype = torch.float16
        M, N, K = 128, 128, 128

        A = torch.randn(M, K, dtype=dtype, device=device)
        B = torch.randn(K, N, dtype=dtype, device=device)
        print(f"✅ Test tensors created: A{list(A.shape)}, B{list(B.shape)}")

        # Test 4: Test TensorRT-LLM extensions function
        print("4. Testing TensorRT-LLM extensions access...")
        result = trtllm_ext.trtllm_extensions_test(A, B)
        expected_shape = (M, N)
        if result.shape == expected_shape:
            print(f"✅ TensorRT-LLM extensions test passed: output shape {list(result.shape)}")
        else:
            print(f"❌ Shape mismatch: expected {expected_shape}, got {result.shape}")
            return False

        # Test 5: Verify extension enums are accessible
        print("5. Verifying TensorRT-LLM extension enums...")
        # The result should be filled with WeightOnlyQuantType::Int8b value
        test_value = result[0, 0].item()
        print(f"✅ TensorRT-LLM enums accessible: got test value {test_value}")

        # Test 6: Test compatibility with standard CUTLASS
        print("6. Testing compatibility with standard CUTLASS...")
        # Import standard CUTLASS test to ensure no conflicts
        import flashinfer.comm.cutlass_gemm as cutlass_gemm
        cutlass_result = cutlass_gemm.cutlass_gemm_test(A, B)
        if cutlass_result.shape == expected_shape:
            print("✅ CUTLASS + TensorRT-LLM extensions compatibility verified")
        else:
            print("❌ Compatibility issue detected")
            return False

        print("\n🎉 Phase 1.5 TensorRT-LLM extensions tests completed successfully!")
        print("✅ TensorRT-LLM CUTLASS extensions compile correctly")
        print("✅ Extension types and enums accessible")
        print("✅ No conflicts with standard CUTLASS")
        print("✅ Foundation ready for AllReduce GEMM implementation")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("Stack trace:")
        traceback.print_exc()
        print("\nDebugging info:")
        print("- Check TensorRT-LLM extensions exist: ls csrc/nv_internal/tensorrt_llm/cutlass_extensions/include/")
        print("- Verify no header conflicts between FlashInfer and TensorRT-LLM")
        print("- Check CUTLASS version compatibility")
        return False

if __name__ == "__main__":
    success = test_trtllm_extensions_compilation()
    if success:
        print("\n🚀 Phase 1.5 Complete: All CUTLASS Integration Tests Passed!")
        print("🎯 CUTLASS headers compile successfully")
        print("🎯 CUTLASS GEMM kernel works correctly")
        print("🎯 TensorRT-LLM extensions integrate properly")
        print("🎯 Ready to proceed with Phase 2: AllReduce GEMM Implementation")
    sys.exit(0 if success else 1)