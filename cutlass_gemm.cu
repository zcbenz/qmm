#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "cutlass/arch/mma.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

using namespace cute;

template <typename Gemm>
__global__
__launch_bounds__(Gemm::MaxThreadsPerBlock, Gemm::MinBlocksPerMultiprocessor)
void gemm_device(typename Gemm::Params params) {
  extern __shared__ char smem_buf[];
  Gemm gemm;
  gemm(params, smem_buf);
}

// Setup params for a NT GEMM
void
gemm_nt(int m, int n, int k,
        float alpha,
        half_t const* A,
        half_t const* B,
        float beta,
        half_t      * C,
        cudaStream_t stream = 0)
{
  auto problem_shape = make_shape(m,n,k);
  auto dA = make_stride(_1{}, m, _0{});
  auto dB = make_stride(_1{}, n, _0{});
  auto dC = make_stride(n, _1{}, _0{});

  static constexpr int ThreadCount = 128;
  using DispatchPolicy = cutlass::gemm::MainloopSm80CpAsync<5>;
  using TileShape = Shape<Int<128>, Int<128>, Int<32>>;
  using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>, Layout<Shape<_2,_2,_1>>, Tile<_32,_32,_16>>;

  using GmemCopyAtomA = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, half_t>;
  using GmemCopyThreadLayoutA = Layout<Shape<_16,_8>, Stride<_1,_16>>;
  using GmemCopyValueLayoutA = Layout<Shape<_8,_1>>;
  using GmemTiledCopyA = decltype(make_tiled_copy(GmemCopyAtomA{}, GmemCopyThreadLayoutA{}, GmemCopyValueLayoutA{}));

  using GmemCopyAtomB = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, half_t>;
  using GmemCopyThreadLayoutB = Layout<Shape<_16,_8>, Stride<_1,_16>>;
  using GmemCopyValueLayoutB = Layout<Shape<_8,_1>>;
  using GmemTiledCopyB = decltype(make_tiled_copy(GmemCopyAtomB{}, GmemCopyThreadLayoutB{}, GmemCopyValueLayoutB{}));

  using SmemLayoutAtomA = Layout<Shape<_64,_8>, Stride<_1,_64>>;
  using SwizzledSmemLayoutAtomA = decltype(composition(Swizzle<3,3,3>{}, SmemLayoutAtomA{}));
  using SmemCopyAtomA = Copy_Atom<SM75_U16x8_LDSM_T, half_t>;

  using SmemLayoutAtomB = Layout<Shape<_64,_8>, Stride<_1,_64>>;
  using SwizzledSmemLayoutAtomB = decltype(composition(Swizzle<3,3,3>{}, SmemLayoutAtomB{}));
  using SmemCopyAtomB = Copy_Atom<SM75_U16x8_LDSM_T, half_t>;

  using AccSmemLayout = Layout<Shape<_128,_128>, Stride<_128,_1>>;
  using SwizzledAccSmemLayout = decltype(composition(Swizzle<3,3,3>{}, AccSmemLayout{}));
  using CopyAtomR2S = Copy_Atom<UniversalCopy<uint32_t>, half_t>;
  using CopyAtomR2G = Copy_Atom<UniversalCopy<uint128_t>, half_t>;
  using ThreadLayoutS2R = Layout<Shape <_8,_16>, Stride< _16,_1>>;
  using ValueLayoutS2R = Layout<Shape<_1,_8>>;
  using TiledCopyS2R = decltype(make_tiled_copy(CopyAtomR2G{}, ThreadLayoutS2R{}, ValueLayoutS2R{}));

  using ThreadEpilogueOp = cutlass::epilogue::thread::LinearCombination<half_t, 1, half_t, half_t>;

  // Collecitve Mainloop
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    DispatchPolicy, TileShape,
    half_t, decltype(dA),
    half_t, decltype(dB),
    TiledMma,
    GmemTiledCopyA, SwizzledSmemLayoutAtomA, SmemCopyAtomA, cute::identity,
    GmemTiledCopyB, SwizzledSmemLayoutAtomB, SmemCopyAtomB, cute::identity>;

  // Collective Epilogue
  using CollectiveEpilogue = cutlass::epilogue::collective::Epilogue<
    decltype(dC), decltype(dC), ThreadEpilogueOp, SwizzledAccSmemLayout, CopyAtomR2S, TiledCopyS2R, CopyAtomR2G>;

  using Gemm = cutlass::gemm::kernel::GemmUniversal<decltype(problem_shape), CollectiveMainloop, CollectiveEpilogue>;

  // check whether allocation surpasses A100 SMEM capacity
  if (Gemm::SharedStorageSize > 164*1024)
    printf("[WARNING]: %d bytes allocated in SMEM, but limit is %d bytes (A100).", Gemm::SharedStorageSize, 164*1024);

  // associate params
  typename CollectiveMainloop::Params mainloop_params;
  typename CollectiveEpilogue::Params epilogue_params;
  typename ThreadEpilogueOp::Params epilogue_op_params;
  typename Gemm::Params gemm_params;

  mainloop_params.ptr_A = A;
  mainloop_params.dA = dA;
  mainloop_params.ptr_B = B;
  mainloop_params.dB = dB;

  epilogue_op_params.alpha = alpha;
  epilogue_op_params.beta = beta;

  epilogue_params.ptr_C = C;
  epilogue_params.dC = dC;
  epilogue_params.ptr_D = C;
  epilogue_params.dD = dC;
  epilogue_params.thread = epilogue_op_params;

  gemm_params.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  gemm_params.problem_shape = problem_shape;
  gemm_params.mainloop = mainloop_params;
  gemm_params.epilogue = epilogue_params;

  // Launch kernel
  dim3 dimBlock(ThreadCount);
  dim3 dimGrid(size(ceil_div(m, Int<128>{})),
               size(ceil_div(n, Int<128>{})));
  cudaFuncSetAttribute(gemm_device<Gemm>, cudaFuncAttributeMaxDynamicSharedMemorySize, Gemm::SharedStorageSize);
  gemm_device<Gemm>
    <<<dimGrid, dimBlock, Gemm::SharedStorageSize, stream>>>(gemm_params);
}

int main(int argc, char** argv)
{
  int m = 8;
  int n = 6144;
  int k = 4096;

  using TA = cute::half_t;
  using TB = cute::half_t;
  using TC = cute::half_t;
  using TI = float;

  float alpha = 1.0f;
  float beta  = 0.0f;

  thrust::host_vector<TA> h_A(m*k);
  thrust::host_vector<TB> h_B(n*k);
  thrust::host_vector<TC> h_C(m*n);

  for (int j = 0; j < m*k; ++j) h_A[j]  = static_cast<TA>(2*(rand() / double(RAND_MAX)) - 1);
  for (int j = 0; j < n*k; ++j) h_B[j]  = static_cast<TB>(2*(rand() / double(RAND_MAX)) - 1);
  for (int j = 0; j < m*n; ++j) h_C[j]  = static_cast<TC>(-1);

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;

  // Run once for reference
  gemm_nt(m, n, k,
       alpha,
       d_A.data().get(),
       d_B.data().get(),
       beta,
       d_C.data().get());
  CUTE_CHECK_LAST();
  thrust::host_vector<TC> cute_result = d_C;

  // Verify output
  cutlass::reference::device::Gemm<TA,
                                   cutlass::layout::ColumnMajor,
                                   TB,
                                   cutlass::layout::RowMajor,
                                   TC,
                                   cutlass::layout::RowMajor,
                                   TI,
                                   TC
                                  > gemm_ref;

  thrust::device_vector<TC> d_C_ref(m*n);

  cutlass::TensorRef ref_A(d_A.data().get(), cutlass::layout::ColumnMajor::packed({m, k}));
  cutlass::TensorRef ref_B(d_B.data().get(), cutlass::layout::RowMajor::packed({k, n}));
  cutlass::TensorRef ref_C(d_C_ref.data().get(), cutlass::layout::RowMajor::packed({m, n}));

  gemm_ref(
    {m, n, k},
    alpha,
    ref_A,
    ref_B,
    beta,
    ref_C,
    ref_C
  );
  CUTE_CHECK_LAST();
  thrust::host_vector<TC> h_C_ref = d_C_ref;

  double threshold = 5e-2;
  double atol = 1e-5;
  double rtol = 1e-1;
  int counter = 0;

  for (int i = 0; i < m * n; ++i) {
    double diff = std::abs(cute_result[i] - h_C_ref[i]);
    double rel_diff = diff / (std::abs(h_C_ref[i]) + 1e-5);

    if (diff > atol && rel_diff > rtol) {
      counter++;
      #if 0
      std::cout << "Mismatch at index " << i << ": "
                << "Expected " << h_C_ref[i] << ", but got " << cute_result[i]
                << " (diff = " << diff << ", rel_diff = " << rel_diff << ")\n";
      #endif
    }
  }

  double err_rate = double(counter) / (m*n);

  if (err_rate < threshold) {
    std::cout << "Verification Passed.\n";
    #if 1
    std::cout << "Error Rate = " << err_rate << "\n";
    #endif
  } else {
    std::cout << "Verification Failed. Error Rate = " << err_rate << "\n";
    return EXIT_FAILURE;
  }

  // Timing iterations
  #if 1
  double flops = (2.0*m*n*k) * 1e-9;
  const int warmup_iterations = 1000;
  const int timing_iterations = 1000;
  GPU_Clock timer;

  for (int i = 0; i < warmup_iterations; ++i) {
    gemm_nt(m, n, k,
        alpha,
        d_A.data().get(),
        d_B.data().get(),
        beta,
        d_C.data().get());
  }
  CUTE_CHECK_LAST();
  
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    gemm_nt(m, n, k,
        alpha,
        d_A.data().get(),
        d_B.data().get(),
        beta,
        d_C.data().get());
  }
  double cute_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUTE_GEMM:\t[%6.3f]TFlop/s\t(%6.3f)ms\n", flops / cute_time / 1000, cute_time * 1000);
  #endif

  return EXIT_SUCCESS;
}
