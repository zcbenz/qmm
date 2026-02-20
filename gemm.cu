#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>
#include <cublas_v2.h>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

namespace cute_gemm {

using namespace cute;

template <class ElementA,
          class ElementB,
          class ElementC,
          class SmemLayoutA,
          class SmemLayoutB,
          class SmemLayoutC>
union SharedStorage {
  struct {
    ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
    ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;
  } mainloop;
  struct {
    ArrayEngine<ElementC, cosize_v<SmemLayoutC>> C;
  } epilogue;
};

template <typename ProblemShape, typename CtaTiler,
          typename TA, typename AStride, typename ASmemLayout, typename TiledCopyA, typename S2RAtomA,
          typename TB, typename BStride, typename BSmemLayout, typename TiledCopyB, typename S2RAtomB,
          typename TC, typename CStride, typename CSmemLayout, typename TiledCopyC, typename R2SAtomC,
          typename TiledMma>
__global__ void gemm_impl(
    ProblemShape shape_MNKL, CtaTiler cta_tiler,
    TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a, S2RAtomA s2r_atom_a,
    TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b, S2RAtomB s2r_atom_b,
    TC      * C, CStride dC, CSmemLayout sC_layout, TiledCopyC copy_c, R2SAtomC r2s_atom_c,
    TiledMma mma) {
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
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, TC, ASmemLayout, BSmemLayout, CSmemLayout>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.mainloop.A.begin()), sA_layout); // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.mainloop.B.begin()), sB_layout); // (BLK_N,BLK_K,PIPE)
  Tensor sC = make_tensor(make_smem_ptr(smem.epilogue.C.begin()), sC_layout); // (BLK_M,BLK_N)

  // Partition the copying of A/B/C tiles across the threads.
  ThrCopy thr_copy_a = copy_a.get_slice(thread_idx);
  Tensor tAgA = thr_copy_a.partition_S(gA); // (ACPY,ACPY_M,ACPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA); // (ACPY,ACPY_M,ACPY_K,PIPE)

  ThrCopy thr_copy_b = copy_b.get_slice(thread_idx);
  Tensor tBgB = thr_copy_b.partition_S(gB); // (BCPY,BCPY_N,BCPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(sB); // (BCPY,BCPY_N,BCPY_K,PIPE)

  ThrCopy s2g_thr_copy_c = copy_c.get_slice(thread_idx);
  Tensor s2g_tCsC = s2g_thr_copy_c.partition_S(sC); // (CCPY,CCPY_M,CCPY_N)
  Tensor s2g_tCgC = s2g_thr_copy_c.partition_D(gC); // (CCPY,CCPY_M,CCPY_N)

  // MMA.
  ThrMMA thr_mma = mma.get_slice(thread_idx);
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0)); // (MMA,MMA_M,MMA_K)
  Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0)); // (MMA,MMA_N,MMA_K)
  Tensor tCrC_accu = thr_mma.partition_fragment_C(gC);   // (MMA,MMA_M,MMA_N)
  Tensor tCrC = make_tensor_like<TC>(tCrC_accu);         // (MMA,MMA_M,MMA_N)

  // Copy Atom retiling.
  TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
  ThrCopy s2r_thr_copy_a = s2r_copy_a.get_slice(thread_idx);
  Tensor s2r_tCsA = s2r_thr_copy_a.partition_S(sA); // (ACPY,MMA_M,MMA_K,PIPE)
  Tensor s2r_tCrA = s2r_thr_copy_a.retile_D(tCrA);  // (ACPY,MMA_M,MMA_K)

  TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
  ThrCopy s2r_thr_copy_b = s2r_copy_b.get_slice(thread_idx);
  Tensor s2r_tCsB = s2r_thr_copy_b.partition_S(sB); // (BCPY,MMA_N,MMA_K,PIPE)
  Tensor s2r_tCrB = s2r_thr_copy_b.retile_D(tCrB);  // (BCPY,MMA_N,MMA_K)

  TiledCopy r2s_copy_c = make_tiled_copy_C(r2s_atom_c, mma);
  ThrCopy r2s_thr_copy_c = r2s_copy_c.get_slice(thread_idx);
  Tensor r2s_tCrC = r2s_thr_copy_c.retile_S(tCrC);  // (CCPY,MMA_M,MMA_N)
  Tensor r2s_tCsC = r2s_thr_copy_c.partition_D(sC); // (CCPY,MMA_M,MMA_N)

  // Predicates for m/n bounds.
  Tensor tApA = make_tensor<bool>(make_shape(size<1>(tAsA), size<2>(tAsA)), Stride<_1,_0>{}); // (CPY_M,CPY_K)
  Tensor tBpB = make_tensor<bool>(make_shape(size<1>(tBsB), size<2>(tBsB)), Stride<_1,_0>{}); // (CPY_N,CPY_K)
  Tensor tCpC = make_tensor<bool>(make_shape(size<1>(s2g_tCsC), size<2>(s2g_tCsC)));          // (CPY_M,CPY_N)
  Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA))); // (BLK_M,BLK_K)
  Tensor cB = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB))); // (BLK_N,BLK_K)
  Tensor cC = make_identity_tensor(make_shape(size<0>(sC), size<1>(sC))); // (BLK_M,BLK_N)
  Tensor tAcA = thr_copy_a.partition_S(cA);     // (CPY,CPY_M,CPY_K)
  Tensor tBcB = thr_copy_b.partition_S(cB);     // (CPY,CPY_N,CPY_K)
  Tensor tCcC = s2g_thr_copy_c.partition_D(cC); // (CPY,CPY_M,CPY_N)
  CUTE_UNROLL
  for (int m = 0; m < size<0>(tApA); ++m) {
    tApA(m,0) = get<0>(tAcA(0,m,0)) < m_max_coord;
  }
  CUTE_UNROLL
  for (int n = 0; n < size<0>(tBpB); ++n) {
    tBpB(n,0) = get<0>(tBcB(0,n,0)) < n_max_coord;
  }
  CUTE_UNROLL
  for (int m = 0; m < size<0>(tCpC); ++m) {
    CUTE_UNROLL
    for (int n = 0; n < size<0>(tCpC); ++n) {
      tCpC(m,n) = elem_less(tCcC(0,m,n), make_coord(m_max_coord, n_max_coord));
    }
  }

  // SMEM pipeline size (static).
  auto K_PIPE_MAX = size<3>(tAsA);
  // Tile iterator.
  int k_tile_count = size<3>(tAgA);
  int k_tile_next = 0;

  // Async load GMEM => SMEM for all SMEM pipelines except last.
  CUTE_UNROLL
  for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe) {
    copy_if(copy_a, tApA, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
    copy_if(copy_b, tBpB, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
    cp_async_fence();

    --k_tile_count;
    if (k_tile_count > 0) {
      ++k_tile_next;
    }
  }

  // Clear accumulators.
  clear(tCrC_accu);

  // RMEM pipeline size (static).
  auto K_BLOCK_MAX = size<2>(tCrA);
  // SMEM iterator.
  int smem_pipe_read  = 0;
  int smem_pipe_write = K_PIPE_MAX - 1;

  // Prefetch SMEM => RMEM for first block in RMEM pipeline.
  Tensor s2r_tCsA_p = s2r_tCsA(_,_,_,smem_pipe_read);
  Tensor s2r_tCsB_p = s2r_tCsB(_,_,_,smem_pipe_read);
  if (K_BLOCK_MAX > 1) {
    cp_async_wait<K_PIPE_MAX - 2>(); // wait first tile
    __syncthreads();
    copy(s2r_atom_a, s2r_tCsA_p(_,_,0), s2r_tCrA(_,_,0));
    copy(s2r_atom_b, s2r_tCsB_p(_,_,0), s2r_tCrB(_,_,0));
  }

  // Loop over k-tiles in SMEM pipeline.
  while (k_tile_count > -(K_PIPE_MAX - 1)) {
    // Loop over blocks in RMEM pipeline.
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
      // Wait for next tile.
      if (k_block == K_BLOCK_MAX - 1) {
        s2r_tCsA_p = s2r_tCsA(_,_,_,smem_pipe_read);
        s2r_tCsB_p = s2r_tCsB(_,_,_,smem_pipe_read);
        cp_async_wait<K_PIPE_MAX - 2>();
        __syncthreads();
      }

      // Prefetch SMEM => RMEM for next block.
      int k_block_next = (k_block + 1) % K_BLOCK_MAX;
      copy(s2r_atom_a, s2r_tCsA_p(_,_,k_block_next), s2r_tCrA(_,_,k_block_next));
      copy(s2r_atom_b, s2r_tCsB_p(_,_,k_block_next), s2r_tCrB(_,_,k_block_next));

      // Async load GMEM => SMEM for next tile.
      if (k_block == 0) {
        copy_if(copy_a, tApA, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
        copy_if(copy_b, tBpB, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
        cp_async_fence();

        --k_tile_count;
        if (k_tile_count > 0) {
          ++k_tile_next;
        }

        smem_pipe_write = smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == K_PIPE_MAX - 1) ? 0 : smem_pipe_read + 1;
      }

      // GEMM for current block.
      gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC_accu);
    }
  }

  // Epilogue.
  CUTE_UNROLL
  for (int i = 0; i < size(tCrC_accu); i++) {
    tCrC(i) = TC(tCrC_accu(i));
  }
  copy(r2s_copy_c, r2s_tCrC, r2s_tCsC);
  __syncthreads();
  copy_if(copy_c, tCpC, s2g_tCsC, s2g_tCgC);
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
}

// Setup params for a TN GEMM
template <typename TA, typename TB, typename TC>
void
gemm_tn(int m, int n, int k, int l,
        TA const* A, int ldA,
        TB const* B, int ldB,
        TC      * C, int ldC,
        cudaStream_t stream = 0) {
  // Define shapes (dynamic).
  auto prob_shape = make_shape(m, n, k, l); // (M, N, K, L)

  // Define TN strides (mixed).
  auto dA = make_stride(ldA, Int<1>{}, m * k); // (dM, dK, dL)
  auto dB = make_stride(ldB, Int<1>{}, n * k); // (dN, dK, dL)
  auto dC = make_stride(ldC, Int<1>{}, m * n); // (dM, dN, dL)

  // Define CTA tile sizes (static).
  auto bM = Int<16>{};
  auto bN = Int<128>{};
  auto bK = Int<64>{};
  auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M, BLK_N, BLK_K)

  // Define the A/B smem layouts (static).
  auto swizzle_ab = composition(Swizzle<3,3,3>{},
                                Layout<Shape <_8,Shape <_8, _8>>,
                                       Stride<_8,Stride<_1,_64>>>{});
  auto bP = Int<5>{}; // pipeline
  auto sA_layout = tile_to_shape(swizzle_ab, make_shape(bM,bK,bP));
  auto sB_layout = tile_to_shape(swizzle_ab, make_shape(bN,bK,bP));

  // The permutation shape of mma.
  auto pM = Int<16>{};
  auto pN = Int<32>{};
  auto pK = Int<16>{};

  TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
                                Layout<Shape<_1,_2,_1>>{},
                                make_tile(pM,pN,pK));
  auto kThreads = size(mma);

  // Define the C smem layouts (static).
  auto swizzle_c = composition(Swizzle<2,3,3>{},
                               make_layout(make_shape(pM,pN), LayoutRight{}));
  auto sC_layout = tile_to_shape(swizzle_c, make_shape(bM,bN));

  // Atoms.
  TiledCopy copy_a = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TA>{},
                                     Layout<Shape<Int<kThreads/8>,_8>,
                                            Stride<_8,_1>>{},
                                     Layout<Shape<_1,Int<128/sizeof_bits<TA>::value>>>{});
  TiledCopy copy_b = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TB>{},
                                     Layout<Shape<Int<kThreads/8>,_8>,
                                            Stride<_8,_1>>{},
                                     Layout<Shape<_1,Int<128/sizeof_bits<TB>::value>>>{});
  TiledCopy copy_c = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TC>{},
                                     Layout<Shape<Int<kThreads/16>,_16>,
                                            Stride<_16,_1>>{},
                                     Layout<Shape<_1,Int<128/sizeof_bits<TC>::value>>>{});

  Copy_Atom<SM75_U32x4_LDSM_N, TA> s2r_atom_a;
  Copy_Atom<SM75_U32x4_LDSM_N, TB> s2r_atom_b;
  Copy_Atom<UniversalCopy<uint32_t>, TC> r2s_atom_c;

  auto* kernel = &gemm_impl<
      decltype(prob_shape), decltype(cta_tiler),
      TA, decltype(dA), decltype(sA_layout), decltype(copy_a), decltype(s2r_atom_a),
      TB, decltype(dB), decltype(sB_layout), decltype(copy_b), decltype(s2r_atom_b),
      TC, decltype(dC), decltype(sC_layout), decltype(copy_c), decltype(r2s_atom_c),
      decltype(mma)>;

  // Set L1 to be SMEM only
  int smem_size = int(sizeof(SharedStorage<TA, TB, TC, decltype(sA_layout), decltype(sB_layout), decltype(sC_layout)>));
  cudaFuncSetAttribute(kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  cudaFuncSetAttribute(kernel,
                       cudaFuncAttributePreferredSharedMemoryCarveout, 100);

  dim3 num_blocks(size(ceil_div(m, bM)), size(ceil_div(n, bN)), l);
  dim3 block_dims(kThreads);
  kernel<<<num_blocks, block_dims, smem_size, stream>>>(
      prob_shape, cta_tiler,
      A, dA, sA_layout, copy_a, s2r_atom_a,
      B, dB, sB_layout, copy_b, s2r_atom_b,
      C, dC, sC_layout, copy_c, r2s_atom_c,
      mma);
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

} // namespace cute_gemm

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

int main(int argc, char** argv) {
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
  // bool is_sm80 = device_prop.major >= 8;

  thrust::host_vector<TA> h_A(m*k*l);
  thrust::host_vector<TB> h_B(n*k*l);
  thrust::host_vector<TC> h_C(m*n*l);

  for (int j = 0; j < h_A.size(); ++j) h_A[j] = static_cast<TA>(2*(rand() / double(RAND_MAX)) - 1) / 5;
  for (int j = 0; j < h_B.size(); ++j) h_B[j] = static_cast<TB>(2*(rand() / double(RAND_MAX)) - 1) / 5;
  for (int j = 0; j < h_C.size(); ++j) h_C[j] = static_cast<TC>(-1);

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;

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
  cute_gemm::gemm(
      transA, transB,
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
  const double tflops = (2.0 * m * n * k) * 1e-12;
  GPU_Clock timer;
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    cute_gemm::gemm(
        transA, transB,
        m, n, k, l,
        d_A.data().get(), ldA,
        d_B.data().get(), ldB,
        d_C.data().get(), ldC);
  }
  double cute_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUTE:    [%5.1f]TFlop/s  (%6.4f)ms\n", tflops / cute_time, cute_time*1000);

  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    cublas_gemm(transA, transB,
                m, n, k, l,
                d_A.data().get(),
                d_B.data().get(),
                d_C.data().get());
  }
  double cublas_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUBLAS:  [%5.1f]TFlop/s  (%6.4f)ms\n", tflops / cublas_time, cublas_time*1000);

  printf("Speedup: [%5.2f]x\n", cublas_time / cute_time);
#endif

  return 0;
}
