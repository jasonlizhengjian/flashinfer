/*
 * Copyright (c) 2025, FlashInfer Contributors.  All rights reserved.
 *
 * Phase 1.5 CUTLASS Integration Test
 * Simple test to verify CUTLASS headers compile in FlashInfer JIT environment
 */
#include <string>
#include "pytorch_extension_utils.h"

// Test basic CUTLASS includes
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

// Phase 1.5: Simple test function to verify CUTLASS headers work
void test_cutlass_basic_impl(at::Tensor& A, at::Tensor& B, at::Tensor& D) {
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

  // Phase 1.5: Just verify we can use CUTLASS types and constants
  // This tests that CUTLASS headers compile correctly
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;

  // Test CUTLASS namespace access
  constexpr int kThreads = cutlass::NumThreadsPerWarpGroup;

  // Simple placeholder - fill with thread count as test value
  D.fill_(static_cast<double>(kThreads));

  // This function successfully compiles if CUTLASS headers work
}

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("test_cutlass_basic", &test_cutlass_basic_impl);
}