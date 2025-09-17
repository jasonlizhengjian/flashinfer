/*
 * Copyright (c) 2025, FlashInfer Contributors.  All rights reserved.
 *
 * Phase 1.5 CUTLASS GEMM Implementation
 * Real CUTLASS GEMM kernel for testing before AllReduce integration
 */
#include <string>
#include "pytorch_extension_utils.h"

// CUTLASS includes
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"

// Simplified CUTLASS GEMM configuration for better compatibility
using GemmKernel = cutlass::gemm::device::Gemm<
    cutlass::half_t,                           // ElementA
    cutlass::layout::RowMajor,                 // LayoutA
    cutlass::half_t,                           // ElementB
    cutlass::layout::ColumnMajor,              // LayoutB
    cutlass::half_t,                           // ElementC
    cutlass::layout::RowMajor,                 // LayoutC
    float                                      // ElementAccumulator
>;

void cutlass_gemm_impl(at::Tensor& A, at::Tensor& B, at::Tensor& C, at::Tensor& D,
                       double alpha, double beta) {
  // Input validation
  TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
  TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
  TORCH_CHECK(D.is_cuda(), "D must be a CUDA tensor");
  TORCH_CHECK(A.dtype() == at::ScalarType::Half, "A must be float16");
  TORCH_CHECK(B.dtype() == at::ScalarType::Half, "B must be float16");
  TORCH_CHECK(D.dtype() == at::ScalarType::Half, "D must be float16");

  // Get dimensions
  auto M = A.size(0);
  auto K = A.size(1);
  auto N = B.size(1);
  TORCH_CHECK(B.size(0) == K, "Matrix dimensions must be compatible for GEMM");
  TORCH_CHECK(D.size(0) == M && D.size(1) == N, "Output tensor has wrong shape");

  // Set CUDA device guard
  const c10::cuda::OptionalCUDAGuard device_guard(A.device());

  // Get CUDA stream
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Setup CUTLASS GEMM arguments
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  // Convert PyTorch tensors to CUTLASS tensor refs
  cutlass::TensorRef<cutlass::half_t const, cutlass::layout::RowMajor> ref_A(
      reinterpret_cast<cutlass::half_t const*>(A.data_ptr<at::Half>()),
      cutlass::layout::RowMajor(K)
  );

  cutlass::TensorRef<cutlass::half_t const, cutlass::layout::ColumnMajor> ref_B(
      reinterpret_cast<cutlass::half_t const*>(B.data_ptr<at::Half>()),
      cutlass::layout::ColumnMajor(K)
  );

  cutlass::TensorRef<cutlass::half_t, cutlass::layout::RowMajor> ref_D(
      reinterpret_cast<cutlass::half_t*>(D.data_ptr<at::Half>()),
      cutlass::layout::RowMajor(N)
  );

  // Handle bias tensor C (optional)
  cutlass::TensorRef<cutlass::half_t const, cutlass::layout::RowMajor> ref_C;
  if (C.numel() > 0) {
      TORCH_CHECK(C.dtype() == at::ScalarType::Half, "C must be float16");
      TORCH_CHECK(C.size(0) == M && C.size(1) == N, "C must have same shape as D");
      ref_C = cutlass::TensorRef<cutlass::half_t const, cutlass::layout::RowMajor>(
          reinterpret_cast<cutlass::half_t const*>(C.data_ptr<at::Half>()),
          cutlass::layout::RowMajor(N)
      );
  } else {
      // No bias, set C to point to D with beta=0
      ref_C = cutlass::TensorRef<cutlass::half_t const, cutlass::layout::RowMajor>(
          reinterpret_cast<cutlass::half_t const*>(D.data_ptr<at::Half>()),
          cutlass::layout::RowMajor(N)
      );
      beta = 0.0;
  }

  // Create CUTLASS GEMM operator
  GemmKernel gemm_op;

  // Setup GEMM arguments
  GemmKernel::Arguments args(
      problem_size,                              // problem_size
      ref_A,                                     // ref_A
      ref_B,                                     // ref_B
      ref_C,                                     // ref_C
      ref_D,                                     // ref_D
      {static_cast<float>(alpha), static_cast<float>(beta)}, // epilogue
      1                                          // split_k_slices
  );

  // Check if GEMM can run
  cutlass::Status status = gemm_op.can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GEMM cannot implement this configuration");

  // Initialize GEMM
  status = gemm_op.initialize(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to initialize CUTLASS GEMM");

  // Run GEMM
  status = gemm_op(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GEMM execution failed");

  // Synchronize to ensure completion
  cudaStreamSynchronize(stream);
}

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("cutlass_gemm", &cutlass_gemm_impl);
}