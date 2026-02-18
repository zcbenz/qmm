#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>
#include <cublas_v2.h>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

namespace cute {

template <typename A, typename B>
struct F32FMA {
  using C = float;
  using D = float;
  using DRegisters = D[1];
  using ARegisters = A[1];
  using BRegisters = B[1];
  using CRegisters = C[1];
  CUTE_HOST_DEVICE static void fma(D& d, const A& a, const B& b, const C& c) {
    d = float(a) * float(b) + c;
  }
};

template <typename A, typename B>
struct MMA_Traits<F32FMA<A,B>> {
  using ValTypeD = float;
  using ValTypeA = A;
  using ValTypeB = B;
  using ValTypeC = float;
  using Shape_MNK = Shape<_1,_1,_1>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape<_1,_1>>;
  using BLayout = Layout<Shape<_1,_1>>;
  using CLayout = Layout<Shape<_1,_1>>;
};

}  // namespace cute

template <typename ProblemShape, typename CtaTiler,
          typename TA, typename AStride, typename ASmemLayout, typename TiledCopyA,
          typename TB, typename BStride, typename BSmemLayout, typename TiledCopyB,
          typename TC, typename CStride, typename TiledMma>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNKL, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
            TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
            TC      * C, CStride dC, TiledMma mma)
{
  using namespace cute;

  CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));
  CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));
  CUTE_STATIC_ASSERT_V(congruent(select<0,2,3>(shape_MNKL), dA));
  CUTE_STATIC_ASSERT_V(congruent(select<1,2,3>(shape_MNKL), dB));
  CUTE_STATIC_ASSERT_V(congruent(select<0,1,3>(shape_MNKL), dC));

  int thread_idx = int(threadIdx.x);
  auto [m_coord, n_coord, l_coord] = static_cast<uint3>(blockIdx);

  // Represent the full tensors.
  Tensor mA_mkl = make_tensor(make_gmem_ptr(A), select<0, 2, 3>(shape_MNKL), dA); // (M,K,L)
  Tensor mB_nkl = make_tensor(make_gmem_ptr(B), select<1, 2, 3>(shape_MNKL), dB); // (N,K,L)
  Tensor mC_mnl = make_tensor(make_gmem_ptr(C), select<0, 1, 3>(shape_MNKL), dC); // (M,N,L)

  // Get batch slice.
  Tensor mA = mA_mkl(_, _, l_coord); // (M,K)
  Tensor mB = mB_nkl(_, _, l_coord); // (N,K)
  Tensor mC = mC_mnl(_, _, l_coord); // (M,N)

  // Get the appropriate blocks for this thread block.
  auto cta_coord = make_coord(m_coord, n_coord, _); // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{}); // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{}); // (BLK_M,BLK_N)

  auto m_max_coord = size<0>(shape_MNKL) - size<0>(gA) * m_coord; // M - BLK_M * m_coord
  auto n_max_coord = size<1>(shape_MNKL) - size<0>(gB) * n_coord; // N - BLK_N * n_coord

  // Shared memory buffers.
  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (BLK_M,BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // (BLK_N,BLK_K)

  // Partition the copying of A and B tiles across the threads.
  ThrCopy thr_copy_a = copy_a.get_slice(thread_idx);
  Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA); // (CPY,CPY_M,CPY_K)
  Tensor tArA = make_fragment_like(tAsA); // (CPY,CPY_M,CPY_K)

  ThrCopy thr_copy_b = copy_b.get_slice(thread_idx);
  Tensor tBgB = thr_copy_b.partition_S(gB); // (CPY,CPY_N,CPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(sB); // (CPY,CPY_N,CPY_K)
  Tensor tBrB = make_fragment_like(tBsB); // (CPY,CPY_N,CPY_K)

  ThrMMA thr_mma = mma.get_slice(thread_idx);
  Tensor tCsA = thr_mma.partition_A(sA); // (MMA,MMA_M,MMA_K)
  Tensor tCsB = thr_mma.partition_B(sB); // (MMA,MMA_N,MMA_K)
  Tensor tCgC = thr_mma.partition_C(gC); // (MMA,MMA_M,MMA_N)

  // Accumulators.
  Tensor tCrC = thr_mma.make_fragment_C(tCgC); // (MMA,MMA_M,MMA_N)
  clear(tCrC);

  // Predicates for m/n bounds.
  Tensor tApA = make_tensor<bool>(make_shape(size<1>(tAsA), size<2>(tAsA)), Stride<_1,_0>{}); // (CPY_M,CPY_K)
  Tensor tBpB = make_tensor<bool>(make_shape(size<1>(tBsB), size<2>(tBsB)), Stride<_1,_0>{}); // (CPY_N,CPY_K)
  Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA))); // (BLK_M,BLK_K)
  Tensor cB = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB))); // (BLK_N,BLK_K)
  Tensor cC = make_identity_tensor(make_shape(size<0>(gC), size<1>(gC))); // (BLK_M,BLK_N)
  Tensor tAcA = thr_copy_a.partition_S(cA); // (CPY,CPY_M,CPY_K)
  Tensor tBcB = thr_copy_b.partition_S(cB); // (CPY,CPY_N,CPY_K)
  Tensor tCcC = thr_mma.partition_C(cC);    // (MMA,MMA_M,MMA_N)
  CUTE_UNROLL
  for (int m = 0; m < size<0>(tApA); ++m) {
    tApA(m,0) = get<0>(tAcA(0,m,0)) < m_max_coord;
  }
  CUTE_UNROLL
  for (int n = 0; n < size<0>(tBpB); ++n) {
    tBpB(n,0) = get<0>(tBcB(0,n,0)) < n_max_coord;
  }

  // Copy gmem to rmem for k_tile=0.
  copy_if(copy_a, tApA, tAgA(_,_,_,0), tArA);
  copy_if(copy_b, tBpB, tBgB(_,_,_,0), tBrB);

  auto K_TILE_MAX = size<3>(tAgA);

  // Main loop.
  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
    __syncthreads();

    copy(tArA, tAsA);
    copy(tBrB, tBsB);
    __syncthreads();

    // Copy gmem to rmem for k_tile+1 with tA|tB thread-partitioned tensors.
    int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
    copy_if(copy_a, tApA, tAgA(_,_,_,k_tile_next), tArA);
    copy_if(copy_b, tBpB, tBgB(_,_,_,k_tile_next), tBrB);

    // Compute gemm on mma-partitioned smem
    gemm(mma, tCsA, tCsB, tCrC);
  }

  CUTE_UNROLL
  for (int i = 0; i < size(tCrC); ++i) {
    if (elem_less(tCcC(i), make_coord(m_max_coord, n_max_coord))) {
      tCgC(i) = tCrC(i);
    }
  }
}


// Setup params for a NT GEMM
template <typename TA, typename TB, typename TC>
void
gemm_nt(int m, int n, int k, int l,
        TA const* A, int ldA,
        TB const* B, int ldB,
        TC      * C, int ldC,
        cudaStream_t stream = 0)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto prob_shape = make_shape(m, n, k, l); // (M, N, K, L)

  // Define NT strides (mixed)
  auto dA = make_stride(Int<1>{}, ldA, m * k); // (dM, dK, dL)
  auto dB = make_stride(Int<1>{}, ldB, n * k); // (dN, dK, dL)
  auto dC = make_stride(Int<1>{}, ldC, m * n); // (dM, dN, dL)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<8>{};
  auto bK = Int<64>{};
  auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M, BLK_N, BLK_K)

  auto swizzle_atom = composition(Swizzle<3,3,3>{},
                                  Layout<Shape <_8, _64>,
                                         Stride<_64, _1>>{});

  auto sA = tile_to_shape(swizzle_atom, make_shape(bM,bK));
  auto sB = tile_to_shape(swizzle_atom, make_shape(bN,bK));

  // Define the thread layouts (static)

  TiledCopy copy_a = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TA>{},
                                     Layout<Shape<_4,_16>>{},
                                     Layout<Shape<_8,_1>>{});
  TiledCopy copy_b = make_tiled_copy(Copy_Atom<UniversalCopy<uint32_t>, TB>{},
                                     Layout<Shape<_4,_16>>{},
                                     Layout<Shape<_2,_1>>{});

#if 0
  TiledMMA mma = make_tiled_mma(F32FMA<TA,TB>{},
                                Layout<Shape<_16,_8,_1>>{});
#else
  TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
                                Layout<Shape<_2,_1,_1>>{},
                                Tile<_32, _8, _16>{});
#endif

  dim3 dimBlock(size(mma));
  dim3 dimGrid(size(ceil_div(m, bM)),
               size(ceil_div(n, bN)),
               l);
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, copy_a,
       B, dB, sB, copy_b,
       C, dC, mma);
}

// Setup params for a TN GEMM
template <typename TA, typename TB, typename TC>
void
gemm_tn(int m, int n, int k, int l,
        TA const* A, int ldA,
        TB const* B, int ldB,
        TC      * C, int ldC,
        cudaStream_t stream = 0)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto prob_shape = make_shape(m, n, k, l); // (M, N, K, L)

  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{}, m * k); // (dM, dK, dL)
  auto dB = make_stride(ldB, Int<1>{}, n * k); // (dN, dK, dL)
  auto dC = make_stride(ldC, Int<1>{}, m * n); // (dM, dN, dL)

  // Define CTA tile sizes (static)
  auto bM = Int<16>{};
  auto bN = Int<128>{};
  auto bK = Int<64>{};
  auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M, BLK_N, BLK_K)

  // Define the smem layouts (static)
  auto swizzle_atom = composition(Swizzle<3,3,3>{},
                                  Layout<Shape <_8, _64>,
                                         Stride<_64, _1>>{});

  auto sA = tile_to_shape(swizzle_atom, make_shape(bM,bK));
  auto sB = tile_to_shape(swizzle_atom, make_shape(bN,bK));

  TiledCopy copy_a = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TA>{},
                                     Layout<Shape<_16,_8>,Stride<_8,_1>>{},
                                     Layout<Shape< _1,_8>>{});
  TiledCopy copy_b = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TB>{},
                                     Layout<Shape<_16,_8>,Stride<_8,_1>>{},
                                     Layout<Shape< _1,_8>>{});

#if 0
  TiledMMA mma = make_tiled_mma(F32FMA<TA,TB>{},
                                Layout<Shape<_16,_8,_1>>{});
#else
  TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
                                Layout<Shape<_1,_4,_1>>{},
                                Tile<_16, _32, _16>{});
#endif

  dim3 dimBlock(size(mma));
  dim3 dimGrid(size(ceil_div(m, bM)),
               size(ceil_div(n, bN)),
               l);
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, copy_a,
       B, dB, sB, copy_b,
       C, dC, mma);
}

template <typename TA, typename TB, typename TC>
void
gemm(char transA, char transB, int m, int n, int k, int l,
     TA const* A, int ldA,
     TB const* B, int ldB,
     TC      * C, int ldC,
     cudaStream_t stream = 0)
{
  if (transA == 'N' && transB == 'T') {
    return gemm_nt(m, n, k, l, A, ldA, B, ldB, C, ldC, stream);
  } else
  if (transA == 'T' && transB == 'N') {
    return gemm_tn(m, n, k, l, A, ldA, B, ldB, C, ldC, stream);
  }
  assert(false && "Not implemented");
}

template <typename TA, typename TB, typename TC>
void cublas_gemm(char transA, char transB,
                 int m, int n, int k, int l,
                 const TA* A,
                 const TB* B,
                 TC* C) {
  static cublasHandle_t h = nullptr;
  if (!h) {
    cublasCreate(&h);
  }
  float alpha_f = 1, beta_f = 0;
  __half alpha_h = 1, beta_h = 0;
  void* p_alpha;
  void* p_beta;
  cudaDataType_t dtype;
  cublasComputeType_t compute_type;
  if constexpr (std::is_same_v<TA, float>) {
    p_alpha = &alpha_f;
    p_beta = &beta_f;
    dtype = CUDA_R_32F;
    compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
  } else {
    p_alpha = &alpha_h;
    p_beta = &beta_h;
    dtype = CUDA_R_16F;
    compute_type = CUBLAS_COMPUTE_16F;
  }
  if (transA == 'N' && transB == 'T') {
    cublasGemmStridedBatchedEx(h,
      CUBLAS_OP_N, CUBLAS_OP_T,
      m, n, k,
      p_alpha,
      A, dtype, m, m*k,
      B, dtype, n, n*k,
      p_beta,
      C, dtype, n, m*n,
      l,
      compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  } else {
    cublasGemmStridedBatchedEx(h,
      CUBLAS_OP_T, CUBLAS_OP_N,
      n, m, k,
      p_alpha,
      B, dtype, k, n*k,
      A, dtype, k, m*k,
      p_beta,
      C, dtype, n, m*n,
      l,
      compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  }
}

int main(int argc, char** argv)
{
  int m = 5120;
  if (argc >= 2)
    sscanf(argv[1], "%d", &m);

  int n = 5120;
  if (argc >= 3)
    sscanf(argv[2], "%d", &n);

  int k = 4096;
  if (argc >= 4)
    sscanf(argv[3], "%d", &k);

  int l = 1;
  if (argc >= 5)
    sscanf(argv[4], "%d", &l);

  char transA = 'T';
  if (argc >= 6)
    sscanf(argv[5], "%c", &transA);

  char transB = 'N';
  if (argc >= 7)
    sscanf(argv[6], "%c", &transB);

  using TA = cute::half_t;
  using TB = cute::half_t;
  using TC = cute::half_t;

  std::cout << "M = " << m << std::endl;
  std::cout << "N = " << n << std::endl;
  std::cout << "K = " << k << std::endl;
  std::cout << "L = " << l << std::endl;
  std::cout << "C = A^" << transA << " B^" << transB << std::endl;

  CUTE_CHECK_ERROR(cudaSetDevice(0));
  cudaDeviceProp device_prop;
  CUTE_CHECK_ERROR(cudaGetDeviceProperties(&device_prop, 0));
  bool is_sm80 = device_prop.major >= 8;

  thrust::host_vector<TA> h_A(m*k*l);
  thrust::host_vector<TB> h_B(n*k*l);
  thrust::host_vector<TC> h_C(m*n*l);

  for (int j = 0; j < h_A.size(); ++j) h_A[j] = static_cast<TA>(2*(rand() / double(RAND_MAX)) - 1);
  for (int j = 0; j < h_B.size(); ++j) h_B[j] = static_cast<TB>(2*(rand() / double(RAND_MAX)) - 1);
  for (int j = 0; j < h_C.size(); ++j) h_C[j] = static_cast<TC>(-1);

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;

  double gflops = (2.0*m*n*k) * 1e-9;

  int ldA = 0, ldB = 0, ldC = n;

  if (transA == 'N') {
    ldA = m;
  } else if (transA == 'T') {
    ldA = k;
  } else {
    assert(false);
  }

  if (transB == 'N') {
    ldB = k;
  } else if (transB == 'T') {
    ldB = n;
  } else {
    assert(false);
  }

  // Run once
  gemm(transA, transB,
       m, n, k, l,
       d_A.data().get(), ldA,
       d_B.data().get(), ldB,
       d_C.data().get(), ldC);
  CUTE_CHECK_LAST();
  thrust::host_vector<TC> cute_result = d_C;

  // Verify
  cublas_gemm(transA, transB,
              m, n, k, l,
              d_A.data().get(),
              d_B.data().get(),
              d_C.data().get());
  thrust::host_vector<TC> cutlass_result = d_C;
  for (size_t i = 0; i < cute_result.size(); ++i) {
    float delta = fabs(float(cute_result[i]) - float(cutlass_result[i]));
    if (delta > 1) {
      printf("!!Wrong result found at %d: %f : %f\n", int(i), float(cute_result[i]), float(cutlass_result[i]));
      exit(1);
    }
  }

#if 1
  // Timing iterations
  const int timing_iterations = 100;
  double gflops = (2.0*m*n*k) * 1e-9;
  GPU_Clock timer;
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    cute_gemm::qmm(
        m, n, k, l, cute::Int<group_size>{},
        d_A.data().get(),
        d_B.data().get(),
        d_S.data().get(),
        d_Z.data().get(),
        d_C.data().get(),
        is_sm80,
        launch_kernel);
  }
  double cute_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUTE:    [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time*1000);

  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    cublas_gemm(
        'T', 'N',
        m, n, k, l,
        d_A.data().get(),
        d_B_ref.data().get(),
        d_C.data().get());
  }
  double cublas_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUBLAS:  [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cublas_time, cublas_time*1000);

  printf("Speedup: %.2fx\n", cublas_time / cute_time);
#endif

  return 0;
}
