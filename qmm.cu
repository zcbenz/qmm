// nvcc qmm.cu -I cutlass/include -I cutlass/tools/util/include --expt-relaxed-constexpr -std=c++20 -lcublas -gencode arch=compute_121a,code=sm_121a -o qmm
// ./qmm M N K L

#include <cute/tensor.hpp>

#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "cutlass/util/GPU_Clock.hpp"

namespace cute_gemm {

using namespace cute;

template <typename Element,
          typename Quant,
          typename SmemLayoutA,
          typename SmemLayoutB,
          typename SmemLayoutC,
          typename SmemLayoutS>
union SharedStorage {
  struct {
    ArrayEngine<Element, cosize_v<SmemLayoutA>> A;
    ArrayEngine<  Quant, cosize_v<SmemLayoutB>> B;
    ArrayEngine<Element, cosize_v<SmemLayoutS>> S;
    ArrayEngine<Element, cosize_v<SmemLayoutS>> Z;
  } mainloop;
  struct {
    ArrayEngine<Element, cosize_v<SmemLayoutC>> C;
  } epilogue;
};

template <typename ProblemShape, typename CtaTiler,
          typename GroupSize, typename Element, typename Quant,
          typename StrideA, typename SmemLayoutA, typename TiledCopyA, typename S2RAtomA,
          typename StrideB, typename SmemLayoutB, typename TiledCopyB, typename S2RAtomB,
          typename StrideC, typename SmemLayoutC, typename TiledCopyC, typename R2SAtomC,
          typename LayoutS, typename SmemLayoutS, typename TiledCopyS, typename S2RAtomS,
          typename TiledMma>
__global__ void qmm_impl(
    ProblemShape shape_MNKL, CtaTiler cta_tiler, GroupSize group_size,
    const Element* A, StrideA dA, SmemLayoutA sA_layout, TiledCopyA copy_a, S2RAtomA s2r_atom_a,
    const   Quant* B, StrideB dB, SmemLayoutB sB_layout, TiledCopyB copy_b, S2RAtomB s2r_atom_b,
          Element* C, StrideC dC, SmemLayoutC sC_layout, TiledCopyC copy_c, R2SAtomC r2s_atom_c,
    const Element* S, const Element* Z, LayoutS S_layout, SmemLayoutS sS_layout, TiledCopyS copy_s, S2RAtomS s2r_atom_s,
    TiledMma mma) {
  CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));
  CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));
  CUTE_STATIC_ASSERT_V(size(copy_c) == size(mma));
  CUTE_STATIC_ASSERT_V(congruent(select<0,2,3>(shape_MNKL), dA));
  CUTE_STATIC_ASSERT_V(congruent(select<1,2,3>(shape_MNKL), dB));
  CUTE_STATIC_ASSERT_V(congruent(select<0,1,3>(shape_MNKL), dC));

  int thread_idx = int(threadIdx.x);
  auto [m_coord, n_coord, l_coord] = static_cast<uint3>(blockIdx);

  // Represent the full tensors.
  Tensor mA_mkl = make_tensor(make_gmem_ptr(A), select<0,2,3>(shape_MNKL), dA); // (M,K,L)
  Tensor mB_nkl = make_tensor(make_gmem_ptr(B), select<1,2,3>(shape_MNKL), dB); // (N,K,L)
  Tensor mC_mnl = make_tensor(make_gmem_ptr(C), select<0,1,3>(shape_MNKL), dC); // (M,N,L)

  Tensor mS_nkl = make_tensor(make_gmem_ptr(S), S_layout); // (N,(group_size,K/group_size),L)
  Tensor mZ_nkl = make_tensor(make_gmem_ptr(Z), S_layout); // (N,(group_size,K/group_size),L)

  // Get batch slice.
  Tensor mA = mA_mkl(_,_,l_coord); // (M,K)
  Tensor mB = mB_nkl(_,_,l_coord); // (N,K)
  Tensor mC = mC_mnl(_,_,l_coord); // (M,N)

  Tensor mS = mS_nkl(_,_,l_coord); // (N,(group_size,K/group_size))
  Tensor mZ = mZ_nkl(_,_,l_coord); // (N,(group_size,K/group_size))

  // Get the appropriate blocks for this thread block.
  auto cta_coord = make_coord(m_coord, n_coord, _); // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{}); // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{}); // (BLK_M,BLK_N)

  Tensor gS = local_tile(mS, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)
  Tensor gZ = local_tile(mZ, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)

  auto m_max_coord = size<0>(shape_MNKL) - size<0>(gA) * m_coord; // M - BLK_M * m_coord

  // Shared memory buffers.
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<Element, Quant,
                                      SmemLayoutA,
                                      SmemLayoutB,
                                      SmemLayoutC,
                                      SmemLayoutS>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.mainloop.A.begin()), sA_layout); // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.mainloop.B.begin()), sB_layout); // (BLK_N,BLK_K,PIPE)
  Tensor sS = make_tensor(make_smem_ptr(smem.mainloop.S.begin()), sS_layout); // (BLK_N,BLK_K,PIPE)
  Tensor sZ = make_tensor(make_smem_ptr(smem.mainloop.Z.begin()), sS_layout); // (BLK_N,BLK_K,PIPE)
  Tensor sC = make_tensor(make_smem_ptr(smem.epilogue.C.begin()), sC_layout); // (BLK_M,BLK_N)

  // Partition the copying of A/B/C tiles across the threads.
  ThrCopy thr_copy_a = copy_a.get_slice(thread_idx);
  Tensor tAgA = thr_copy_a.partition_S(gA); // (ACPY,ACPY_M,ACPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA); // (ACPY,ACPY_M,ACPY_K,PIPE)

  ThrCopy thr_copy_b = copy_b.get_slice(thread_idx);
  Tensor tBgB = thr_copy_b.partition_S(gB); // (BCPY,BCPY_N,BCPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(sB); // (BCPY,BCPY_N,BCPY_K,PIPE)
  Tensor tBgS = thr_copy_b.partition_S(gS); // (BCPY,BCPY_N,BCPY_K,k)
  Tensor tBgZ = thr_copy_b.partition_S(gZ); // (BCPY,BCPY_N,BCPY_K,k)

  ThrCopy s2g_thr_copy_c = copy_c.get_slice(thread_idx);
  Tensor s2g_tCsC = s2g_thr_copy_c.partition_S(sC); // (CCPY,CCPY_M,CCPY_N)
  Tensor s2g_tCgC = s2g_thr_copy_c.partition_D(gC); // (CCPY,CCPY_M,CCPY_N)

  ThrCopy thr_copy_s = copy_s.get_slice(thread_idx);
  Tensor tSgS = thr_copy_s.partition_S(gS); // (SCPY,SCPY_M,SCPY_K,k)
  Tensor tSsS = thr_copy_s.partition_D(sS); // (SCPY,SCPY_M,SCPY_K,PIPE)
  Tensor tSgZ = thr_copy_s.partition_S(gZ); // (SCPY,SCPY_M,SCPY_K,k)
  Tensor tSsZ = thr_copy_s.partition_D(sZ); // (SCPY,SCPY_M,SCPY_K,PIPE)

  // MMA.
  ThrMMA thr_mma = mma.get_slice(thread_idx);
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));   // (MMA,MMA_M,MMA_K)
  Tensor tCsB = thr_mma.partition_B(sB(_,_,0));            // (MMA,MMA_N,MMA_K)
  Tensor tCrB = make_fragment_like<Quant>(tCsB);           // (MMA,MMA_N,MMA_K)
  Tensor tCrB_dequant = make_fragment_like<Element>(tCrB); // (MMA,MMA_N,MMA_K)
  Tensor tCrC_accu = thr_mma.partition_fragment_C(gC);     // (MMA,MMA_M,MMA_N)
  Tensor tCrC = make_tensor_like<Element>(tCrC_accu);      // (MMA,MMA_M,MMA_N)

  Tensor tCgS = thr_mma.partition_B(gS); // (MMA,MMA_N,MMA_K,k)
  Tensor tCgZ = thr_mma.partition_B(gZ); // (MMA,MMA_N,MMA_K,k)

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
  Tensor tApA = make_tensor<bool>(make_shape(size<1>(tAsA),     size<2>(tAsA)    ), Stride<_1,_0>{}); // (CPY_M,CPY_K)
  Tensor tCpC = make_tensor<bool>(make_shape(size<1>(s2g_tCsC), size<2>(s2g_tCsC)), Stride<_1,_0>{}); // (CPY_M,CPY_N)
  Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA))); // (BLK_M,BLK_K)
  Tensor cC = make_identity_tensor(make_shape(size<0>(sC), size<1>(sC))); // (BLK_M,BLK_N)
  Tensor tAcA = thr_copy_a.partition_S(cA);     // (CPY,CPY_M,CPY_K)
  Tensor tCcC = s2g_thr_copy_c.partition_D(cC); // (CPY,CPY_M,CPY_N)
  CUTE_UNROLL
  for (int m = 0; m < size<0>(tApA); ++m) {
    tApA(m,0) = get<0>(tAcA(0,m,0)) < m_max_coord;
  }
  CUTE_UNROLL
  for (int m = 0; m < size<0>(tCpC); ++m) {
    tCpC(m,0) = get<0>(tCcC(0,m,0)) < m_max_coord;
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
    copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
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
  Tensor s2r_tCsA_p = s2r_tCsA(_,_,_,smem_pipe_read); // (ACPY,MMA_M,MMA_K)
  Tensor s2r_tCsB_p = s2r_tCsB(_,_,_,smem_pipe_read); // (BCPY,MMA_N,MMA_K)
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
        copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
        cp_async_fence();

        --k_tile_count;
        if (k_tile_count > 0) {
          ++k_tile_next;
        }

        smem_pipe_write = smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == K_PIPE_MAX - 1) ? 0 : smem_pipe_read + 1;
      }

      // Dequantize B.
      Tensor quant = tCrB(_,_,k_block);
      Tensor weight = tCrB_dequant(_,_,k_block);
      Tensor scale = tCgS(_,_,k_block,k_tile_next);
      Tensor zero_point = tCgZ(_,_,k_block,k_tile_next);
      for (int i = 0; i < size(weight); i++) {
        weight(i) = quant(i) * scale(i) + zero_point(i);
      }

      // GEMM for current block.
      gemm(mma, tCrA(_,_,k_block), tCrB_dequant(_,_,k_block), tCrC_accu);
    }
  }

  // Epilogue.
  CUTE_UNROLL
  for (int i = 0; i < size(tCrC_accu); i++) {
    tCrC(i) = Element(tCrC_accu(i));
  }
  copy(r2s_copy_c, r2s_tCrC, r2s_tCsC);
  __syncthreads();
  copy_if(copy_c, tCpC, s2g_tCsC, s2g_tCgC);
}

template <typename GroupSize, typename Element, typename Quant, typename F>
void qmm(
    int m, int n, int k, int l,
    GroupSize group_size,
    const Element* A, const Quant* B, Element* C,
    const Element* S, const Element* Z,
    F&& launch_kernel) {
  // Define shapes (dynamic).
  auto prob_shape = make_shape(m, n, k, l); // (M,N,K,L)

  // Define TN strides (mixed).
  auto dA = make_stride(k, Int<1>{}, m * k); // (dM,dK,dL)
  auto dB = make_stride(k, Int<1>{}, n * k); // (dN,dK,dL)
  auto dC = make_stride(n, Int<1>{}, m * n); // (dM,dN,dL)

  // Define CTA tile sizes (static).
  auto bM = Int<16>{};
  auto bN = Int<128>{};
  auto bK = Int<64>{};
  auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M,BLK_N,BLK_K)

  // The permutation shape of mma.
  auto pM = Int<16>{};
  auto pN = Int<32>{};
  auto pK = Int<16>{};

  TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
                                Layout<Shape<_1,_2,_1>>{},
                                make_tile(pM, pN, pK));
  auto kThreads = size(mma);

  // Define the A/B smem layouts (static).
  auto swizzle_ab = composition(Swizzle<3,3,3>{},
                                Layout<Shape <_8,Shape <_8, _8>>,
                                       Stride<_8,Stride<_1,_64>>>{});
  auto bP = Int<5>{}; // pipeline
  auto sA_layout = tile_to_shape(swizzle_ab, make_shape(bM, bK, bP));
  auto sB_layout = tile_to_shape(swizzle_ab, make_shape(bN, bK, bP));

  // Define the C smem layouts (static).
  auto swizzle_c = composition(Swizzle<2,3,3>{},
                               make_layout(make_shape(pM, pN), LayoutRight{}));
  auto sC_layout = tile_to_shape(swizzle_c, make_shape(bM, bN));

  // Define the S/Z smem layouts (static).
  // TODO: Do we need swizzle?
  auto bZ = Int<ceil_div(bK, group_size)>{};
#if 0
  auto sS_layout = make_layout(make_shape(bN, bZ, bP),
                               make_stride(bZ, Int<1>{}, bN * bZ));
#else
  auto sS_layout = sA_layout;
#endif

  // Define layout of scales (mixed).
  auto S_layout = make_layout(
      make_shape(n, make_shape(group_size, k / group_size), l),
      make_stride(k / group_size, Stride<_0, _1>{}, n * k / group_size));

  // Atoms.
  TiledCopy copy_a = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, Element>{},
                                     Layout<Shape<Int<kThreads/8>,_8>,
                                            Stride<_8,_1>>{},
                                     Layout<Shape<_1,Int<128/sizeof_bits_v<Element>>>>{});
  TiledCopy copy_b = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint32_t>, Quant>{},
                                     Layout<Shape<Int<kThreads/8>,_8>,
                                            Stride<_8,_1>>{},
                                     Layout<Shape<_1,Int<32/sizeof_bits_v<Quant>>>>{});
  TiledCopy copy_c = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                                     Layout<Shape<Int<kThreads/16>,_16>,
                                            Stride<_16,_1>>{},
                                     Layout<Shape<_1,Int<128/sizeof_bits_v<Element>>>>{});
#if 0
  TiledCopy copy_s = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint32_t>, Element>{},
                                     Layout<Shape<Int<kThreads/2>,_2>,
                                            Stride<_1,_1>>{},
                                     Layout<Shape<_1,Int<32/sizeof_bits_v<Element>>>>{});
#else
  TiledCopy copy_s = copy_a;
#endif

  Copy_Atom<SM75_U32x4_LDSM_N, Element> s2r_atom_a;
  Copy_Atom<UniversalCopy<Quant>, Quant> s2r_atom_b;
  Copy_Atom<UniversalCopy<uint32_t>, Element> r2s_atom_c;
  Copy_Atom<UniversalCopy<Element>, Element> s2r_atom_s;

  auto* kernel = &qmm_impl<
      decltype(prob_shape), decltype(cta_tiler),
      GroupSize, Element, Quant,
      decltype(dA), decltype(sA_layout), decltype(copy_a), decltype(s2r_atom_a),
      decltype(dB), decltype(sB_layout), decltype(copy_b), decltype(s2r_atom_b),
      decltype(dC), decltype(sC_layout), decltype(copy_c), decltype(r2s_atom_c),
      decltype(S_layout), decltype(sS_layout), decltype(copy_s), decltype(s2r_atom_s),
      decltype(mma)>;

  // Set L1 to be SMEM only.
  int smem_size = int(sizeof(SharedStorage<Element, Quant,
                                           decltype(sA_layout),
                                           decltype(sB_layout),
                                           decltype(sC_layout),
                                           decltype(sS_layout)>));
  cudaFuncSetAttribute(kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  cudaFuncSetAttribute(kernel,
                       cudaFuncAttributePreferredSharedMemoryCarveout, 100);

  // Launch kernel.
  dim3 num_blocks(size(ceil_div(m, bM)), size(ceil_div(n, bN)), l);
  dim3 block_dims(kThreads);
  void* args[] = {
      &prob_shape, &cta_tiler, &group_size,
      &A, &dA, &sA_layout, &copy_a, &s2r_atom_a,
      &B, &dB, &sB_layout, &copy_b, &s2r_atom_b,
      &C, &dC, &sC_layout, &copy_c, &r2s_atom_c,
      &S, &Z, &S_layout, &sS_layout, &copy_s, &s2r_atom_s,
      &mma};
  launch_kernel(reinterpret_cast<void*>(kernel), num_blocks, block_dims, smem_size, args);
}

}  // namespace cute_gemm

template <typename Element>
void cublas_gemm(char transA, char transB,
                 int m, int n, int k, int l,
                 const Element* A, const Element* B, Element* C) {
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
  if constexpr (std::is_same_v<Element, float>) {
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
      A, dtype, m, m * k,
      B, dtype, n, n * k,
      p_beta,
      C, dtype, m, m * n,
      l,
      compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  } else {
    cublasGemmStridedBatchedEx(h,
      CUBLAS_OP_T, CUBLAS_OP_N,
      n, m, k,
      p_alpha,
      B, dtype, k, m * k,
      A, dtype, k, n * k,
      p_beta,
      C, dtype, n, m * n,
      l,
      compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  }
}

void launch_kernel(void* func, dim3 num_blocks, dim3 block_dims, size_t smem_bytes, void** args) {
  cudaLaunchKernel(func, num_blocks, block_dims, args, smem_bytes, /* stream */ nullptr);
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

  std::cout << "M = " << m << std::endl;
  std::cout << "N = " << n << std::endl;
  std::cout << "K = " << k << std::endl;
  std::cout << "L = " << l << std::endl;

  CUTE_CHECK_ERROR(cudaSetDevice(0));
  cudaDeviceProp device_prop;
  CUTE_CHECK_ERROR(cudaGetDeviceProperties(&device_prop, 0));
  bool is_sm80 = device_prop.major >= 8;

  using Element = cute::half_t;
  using Quant = cute::int8_t;

  constexpr int group_size = 64;

  thrust::host_vector<Element> h_A(m*k*l);
  thrust::host_vector<Quant> h_B(n*k*l); // quantized B
  thrust::host_vector<Element> h_S(n*k*l/group_size); // scales
  thrust::host_vector<Element> h_Z(n*k*l/group_size); // zero points
  thrust::host_vector<Element> h_B_ref(n*k*l); // dequantized B
  thrust::host_vector<Element> h_C(m*n*l);

  for (int j = 0; j < h_A.size(); ++j) h_A[j] = (2*(rand() / double(RAND_MAX)) - 1) / 10;
#if 1
  for (int j = 0; j < h_B.size(); ++j) h_B[j] = rand() % 16;
  for (int j = 0; j < h_S.size(); ++j) h_S[j] = (0.01f * (rand() / double(RAND_MAX)) + 0.001f);
  for (int j = 0; j < h_Z.size(); ++j) h_Z[j] = (0.1f * (rand() / double(RAND_MAX)) + 0.01f);
#else
  for (int j = 0; j < h_B.size(); ++j) h_B[j] = Quant::from_float((2*(rand() / double(RAND_MAX)) - 1) / 10);
  for (int j = 0; j < h_S.size(); ++j) h_S[j] = 1;
  for (int j = 0; j < h_Z.size(); ++j) h_Z[j] = 0;
#endif
  for (int j = 0; j < h_C.size(); ++j) h_C[j] = -1;

  // Dequantize B: B_ref = B * S + Z
  for (int j = 0; j < h_B_ref.size(); ++j) {
    h_B_ref[j] = static_cast<Element>(h_B[j]) * h_S[j/group_size] + h_Z[j/group_size];
  }

  thrust::device_vector<Element> d_A = h_A;
  thrust::device_vector<Quant> d_B = h_B;
  thrust::device_vector<Element> d_S = h_S;
  thrust::device_vector<Element> d_Z = h_Z;
  thrust::device_vector<Element> d_B_ref = h_B_ref;
  thrust::device_vector<Element> d_C = h_C;

  // Run once
  cute_gemm::qmm(
      m, n, k, l,
      cute::Int<group_size>{},
      d_A.data().get(),
      d_B.data().get(),
      d_C.data().get(),
      d_S.data().get(),
      d_Z.data().get(),
      launch_kernel);
  CUTE_CHECK_LAST();
  thrust::host_vector<Element> cute_result = d_C;

  // Verify
  cublas_gemm(
      'T', 'N',
      m, n, k, l,
      d_A.data().get(),
      d_B_ref.data().get(),
      d_C.data().get());
  thrust::host_vector<Element> cutlass_result = d_C;
  for (size_t i = 0; i < cute_result.size(); ++i) {
    float delta = fabs(float(cute_result[i]) - float(cutlass_result[i]));
    if (delta > 3e-1) {
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
        m, n, k, l,
        cute::Int<group_size>{},
        d_A.data().get(),
        d_B.data().get(),
        d_C.data().get(),
        d_S.data().get(),
        d_Z.data().get(),
        launch_kernel);
  }
  double cute_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("QMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time*1000);

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
