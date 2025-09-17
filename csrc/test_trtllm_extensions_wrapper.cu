/*
 * Copyright (c) 2025, FlashInfer Contributors.  All rights reserved.
 *
 * Phase 1.5 TensorRT-LLM CUTLASS Extensions Test
 * Test that TensorRT-LLM CUTLASS extensions can be compiled with FlashInfer
 */
#include <string>
#include "pytorch_extension_utils.h"

// Test basic CUTLASS includes
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

// Test TensorRT-LLM CUTLASS extensions
#include "cutlass_extensions/gemm_configs.h"
#include "cutlass_extensions/epilogue_helpers.h"
#include "cutlass_extensions/weight_only_quant_op.h"

// Phase 1.5: Test function to verify TensorRT-LLM extensions work
void test_trtllm_extensions_impl(at::Tensor& A, at::Tensor& B, at::Tensor& D) {
  // Input validation
  TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
  TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
  TORCH_CHECK(D.is_cuda(), "D must be a CUDA tensor");
  TORCH_CHECK(A.dtype() == B.dtype(), "A and B must have same dtype");

  // Get dimensions
  auto M = A.size(0);
  auto K = A.size(1);
  auto N = B.size(1);
  TORCH_CHECK(B.size(0) == K, "Matrix dimensions must be compatible for GEMM");
  TORCH_CHECK(D.size(0) == M && D.size(1) == N, "Output tensor has wrong shape");

  // Phase 1.5: Test access to TensorRT-LLM extension types and functions
  using namespace tensorrt_llm::cutlass_extensions;

  // Test that we can access TensorRT-LLM CUTLASS extension enums
  auto weight_only_op = cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY;
  auto tile_config = CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64;

  // Test access to CUTLASS extension configurations
  constexpr int kSm = 80;  // Test SM architecture constant

  // Simple placeholder - fill with weight_only_op enum value as test
  D.fill_(static_cast<double>(static_cast<int>(weight_only_op)));

  // This function successfully compiles if TensorRT-LLM extensions work
}

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("test_trtllm_extensions", &test_trtllm_extensions_impl);
}