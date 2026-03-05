#include <cublas_v2.h>
#include <cute/layout.hpp>
#include <cutlass/util/GPU_Clock.hpp>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/reference/device/tensor_compare.h>
#include <cutlass/util/mixed_dtype_utils.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define USE_GEMM 0

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
    ArrayEngine<Quant,   cosize_v<SmemLayoutB>> B;
    ArrayEngine<Element, cosize_v<SmemLayoutS>> S;
    ArrayEngine<Element, cosize_v<SmemLayoutS>> Z;
  } mainloop;
  struct {
    ArrayEngine<Element, cosize_v<SmemLayoutC>> C;
  } epilogue;
};

template <typename ProblemShape, typename CtaTiler,
          typename Element, typename Quant,
          typename StrideA, typename SmemLayoutA, typename TiledCopyA, typename S2RAtomA,
          typename StrideB, typename SmemLayoutB, typename TiledCopyB, typename S2RAtomB,
          typename StrideC, typename SmemLayoutC, typename TiledCopyC, typename R2SAtomC,
          typename LayoutS, typename SmemLayoutS, typename TiledCopyS,
          typename TiledMma>
__global__ void qmm_sm80_kernel(
    ProblemShape shape_MNKL, CtaTiler cta_tiler,
    const Element* A, StrideA dA, SmemLayoutA sA_layout, TiledCopyA g2s_copy_a, S2RAtomA s2r_atom_a,
    const Quant*   B, StrideB dB, SmemLayoutB sB_layout, TiledCopyB g2s_copy_b, S2RAtomB s2r_atom_b,
          Element* C, StrideC dC, SmemLayoutC sC_layout, TiledCopyC s2g_copy_c, R2SAtomC r2s_atom_c,
    const Element* S, const Element* Z, LayoutS gS_layout, SmemLayoutS sS_layout, TiledCopyS g2s_copy_s,
    TiledMma mma) {
  CUTE_STATIC_ASSERT_V(size(g2s_copy_a) == size(mma));
  CUTE_STATIC_ASSERT_V(size(g2s_copy_b) == size(mma));
  CUTE_STATIC_ASSERT_V(size(s2g_copy_c) == size(mma));
  CUTE_STATIC_ASSERT_V(congruent(select<0,2,3>(shape_MNKL), dA));
  CUTE_STATIC_ASSERT_V(congruent(select<1,2,3>(shape_MNKL), dB));
  CUTE_STATIC_ASSERT_V(congruent(select<0,1,3>(shape_MNKL), dC));

  int thread_idx = int(threadIdx.x);
  auto [m_coord, n_coord, l_coord] = static_cast<uint3>(blockIdx);

  // Represent the full tensors.
  Tensor mA_mkl = make_tensor(make_gmem_ptr(A), select<0,2,3>(shape_MNKL), dA); // (M,K,L)
  Tensor mB_nkl = make_tensor(make_gmem_ptr(B), select<1,2,3>(shape_MNKL), dB); // (N,K,L)
  Tensor mC_mnl = make_tensor(make_gmem_ptr(C), select<0,1,3>(shape_MNKL), dC); // (M,N,L)

  Tensor mS_nkl = make_tensor(make_gmem_ptr(S), gS_layout); // (N,(S,k*K/group_size/S),L)
  Tensor mZ_nkl = make_tensor(make_gmem_ptr(Z), gS_layout); // (N,(S,k*K/group_size/S),L)

  // Get batch slice.
  Tensor mA = mA_mkl(_,_,l_coord); // (M,K)
  Tensor mB = mB_nkl(_,_,l_coord); // (N,K)
  Tensor mC = mC_mnl(_,_,l_coord); // (M,N)

  Tensor mS = mS_nkl(_,_,l_coord); // (N,(S,k*K/group_size/S))
  Tensor mZ = mZ_nkl(_,_,l_coord); // (N,(S,k*K/group_size/S))

  // Get the appropriate blocks for this thread block.
  auto cta_coord = make_coord(m_coord, n_coord, _); // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{}); // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{}); // (BLK_M,BLK_N)

  auto bS = get<1, 0>(shape(gS_layout));
  auto s_tiler = make_shape(get<1>(cta_tiler), bS); // (N,S)
  Tensor gS = local_tile(mS, s_tiler, make_coord(n_coord, _)); // (BLK_N,BLK_S,k*K/group_size/S)
  Tensor gZ = local_tile(mZ, s_tiler, make_coord(n_coord, _)); // (BLK_N,BLK_S,k*K/group_size/S)

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
  Tensor sC = make_tensor(make_smem_ptr(smem.epilogue.C.begin()), sC_layout); // (BLK_M,BLK_N,PIPE)

  Tensor sS = make_tensor(make_smem_ptr(smem.mainloop.S.begin()), sS_layout); // (BLK_N,(group_size,K/group_size),S,PIPE)
  Tensor sZ = make_tensor(make_smem_ptr(smem.mainloop.Z.begin()), sS_layout); // (BLK_N,(group_size,K/group_size),S,PIPE)
  Tensor sS_c = make_tensor(make_smem_ptr(smem.mainloop.S.begin()), filter(sS_layout)); // (BLK_N,BLK_S,PIPE)
  Tensor sZ_c = make_tensor(make_smem_ptr(smem.mainloop.Z.begin()), filter(sS_layout)); // (BLK_N,BLK_S,PIPE)

  // Partition the copying of tiles across the threads.
  ThrCopy g2s_thr_copy_a = g2s_copy_a.get_slice(thread_idx);
  Tensor tAgA = g2s_thr_copy_a.partition_S(gA); // (ACPY,ACPY_M,ACPY_K,k)
  Tensor tAsA = g2s_thr_copy_a.partition_D(sA); // (ACPY,ACPY_M,ACPY_K,PIPE)

  ThrCopy g2s_thr_copy_b = g2s_copy_b.get_slice(thread_idx);
  Tensor tBgB = g2s_thr_copy_b.partition_S(gB); // (BCPY,BCPY_N,BCPY_K,k)
  Tensor tBsB = g2s_thr_copy_b.partition_D(sB); // (BCPY,BCPY_N,BCPY_K,PIPE)

  ThrCopy s2g_thr_copy_c = s2g_copy_c.get_slice(thread_idx);
  Tensor s2g_tCsC = s2g_thr_copy_c.partition_S(sC); // (CCPY,CCPY_M,CCPY_N)
  Tensor s2g_tCgC = s2g_thr_copy_c.partition_D(gC); // (CCPY,CCPY_M,CCPY_N)

  ThrCopy g2s_thr_copy_s = g2s_copy_s.get_slice(thread_idx);
  Tensor tSgS = g2s_thr_copy_s.partition_S(gS);   // (BCPY,BCPY_N,BCPY_S,k*K/group_size/S)
  Tensor tSsS = g2s_thr_copy_s.partition_D(sS_c); // (BCPY,BCPY_N,BCPY_S,PIPE)
  Tensor tSgZ = g2s_thr_copy_s.partition_S(gZ);   // (BCPY,BCPY_N,BCPY_S,k*K/group_size/S)
  Tensor tSsZ = g2s_thr_copy_s.partition_D(sZ_c); // (BCPY,BCPY_N,BCPY_S,PIPE)

  // MMA.
  ThrMMA thr_mma = mma.get_slice(thread_idx);
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0)); // (MMA,MMA_M,MMA_K)
  Tensor tCsB = thr_mma.partition_B(sB(_,_,0));          // (MMA,MMA_N,MMA_K)
  Tensor tCrB = make_fragment_like<Quant>(tCsB);         // (MMA,MMA_N,MMA_K)
#if !USE_GEMM
  Tensor tCrB_dq = make_fragment_like<Element>(tCsB);    // (MMA,MMA_N,MMA_K)
#endif
  Tensor tCgC = thr_mma.partition_C(gC);                 // (MMA,MMA_M,MMA_N)
  Tensor tCrC_accu = make_fragment_like<float>(tCgC);    // (MMA,MMA_M,MMA_N)
  Tensor tCrC = make_fragment_like<Element>(tCgC);       // (MMA,MMA_M,MMA_N)

#if !USE_GEMM
  Tensor tCsS = thr_mma.partition_B(sS); // (MMA,MMA_N,MMA_K,S,PIPE)
  Tensor tCsS_p = tCsS(_,_,_,0,0);       // (MMA,MMA_N,MMA_K)
  Tensor tCsZ = thr_mma.partition_B(sZ); // (MMA,MMA_N,MMA_K,S,PIPE)
  Tensor tCsZ_p = tCsZ(_,_,_,0,0);       // (MMA,MMA_N,MMA_K)
#endif

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

#if !USE_GEMM
  if (false) {
    print("s_tiler: "); print(s_tiler); print("\n");
    print("mZ: "); print(mZ); print("\n");
    print("gZ: "); print(gZ); print("\n");
    print("sZ: "); print(sZ); print("\n");
    print("sZ_c: "); print(sZ_c); print("\n");
    print("tSgZ: "); print(tSgZ); print("\n");
    print("tSsZ: "); print(tSsZ); print("\n");
    print("tCsZ: "); print(tCsZ); print("\n");
  }
#endif

  // Predicates for m bound.
  auto m_max_coord = size<0>(shape_MNKL) - size<0>(gA) * m_coord; // M - BLK_M * m_coord
  Tensor tCpC = make_tensor<bool>(make_shape(size<1>(s2g_tCsC), size<2>(s2g_tCsC)), Stride<_1,_0>{}); // (CPY_M,CPY_N)
  Tensor cC = make_identity_tensor(make_shape(size<0>(sC), size<1>(sC))); // (BLK_M,BLK_N)
  Tensor tCcC = s2g_thr_copy_c.partition_D(cC); // (CPY,CPY_M,CPY_N)
  CUTE_UNROLL
  for (int m = 0; m < size<0>(tCpC); ++m) {
    tCpC(m,0) = get<0>(tCcC(0,m,0)) < m_max_coord;
  }

  auto K_PIPE_MAX = size<3>(tAsA);
  int ab_pipe_read = 0;
  int ab_pipe_write = 0;
  int sz_pipe_read = 0;
  int sz_pipe_write = 0;

  // GMEM => SMEM.
  auto fetch_gmem = [&](int tile) {
    copy(g2s_copy_a, tAgA(_,_,_,tile), tAsA(_,_,_,ab_pipe_write));
    copy(g2s_copy_b, tBgB(_,_,_,tile), tBsB(_,_,_,ab_pipe_write));
#if !USE_GEMM
    if (tile % bS == 0) {
      int tile_s = tile / bS;
      copy(g2s_copy_s, tSgS(_,_,_,tile_s), tSsS(_,_,_,sz_pipe_write));
      copy(g2s_copy_s, tSgZ(_,_,_,tile_s), tSsZ(_,_,_,sz_pipe_write));
    }
#endif
    cp_async_fence();
    ab_pipe_write = (ab_pipe_write + 1) % K_PIPE_MAX;
    if (tile % bS == 0) {
      sz_pipe_write = (sz_pipe_write + 1) % 2;
    }
  };
  // SMEM => RMEM.
  auto fetch_smem = [&](auto block) {
    copy(s2r_atom_a, s2r_tCsA(_,_,block,ab_pipe_read), s2r_tCrA(_,_,block));
    copy(s2r_atom_b, s2r_tCsB(_,_,block,ab_pipe_read), s2r_tCrB(_,_,block));
#if !USE_GEMM
    Tensor scale = tCsS_p(_,_,block);
    Tensor zero_point = tCsZ_p(_,_,block);
    Tensor quant = tCrB(_,_,block);
    Tensor weight = tCrB_dq(_,_,block);
    CUTE_UNROLL
    for (int i = 0; i < size(quant); ++i) {
      weight(i) = quant(i) * scale(i) + zero_point(i);
    }
#endif
  };

  auto K_TILE_MAX = size<3>(tAgA);
  auto K_BLOCK_MAX = size<2>(tCrA);

  // Prefetch beginning tiles.
  fetch_gmem(0);

  // Clear accumulators.
  clear(tCrC_accu);

  // Prefetch first block.
  if constexpr (K_BLOCK_MAX > 1) {
    cp_async_wait<K_PIPE_MAX - 2>();
    __syncthreads();
    fetch_smem(Int<0>{});
  }

  // Loop over CTA tiles.
  for (int tile = 0; tile < K_TILE_MAX; ++tile) {
    // Unroll MMA blocks.
    CUTE_UNROLL
    for (int block = 0; block < K_BLOCK_MAX; ++block) {
      // Wait for last tile.
      int tile_next = (tile + 1 < K_TILE_MAX) ? tile + 1 : tile;
      if (block == K_BLOCK_MAX - 1) {
        ab_pipe_read = (ab_pipe_read + 1) % K_PIPE_MAX;
        if (tile_next % bS == 0) {
          sz_pipe_read = (sz_pipe_read + 1) % 2;
        }
#if !USE_GEMM
        int tile_s = tile_next % bS;
        tCsS_p = tCsS(_,_,_,tile_s,sz_pipe_read);
        tCsZ_p = tCsZ(_,_,_,tile_s,sz_pipe_read);
#endif
        cp_async_wait<K_PIPE_MAX - 2>();
        __syncthreads();
      }
      // Prefetch next block.
      fetch_smem((block + Int<1>{}) % K_BLOCK_MAX);
      // Prefetch next tile.
      if (block == 0) {
        fetch_gmem(tile_next);
      }
      // MMA.
#if USE_GEMM
      gemm(mma, tCrA(_,_,block), tCrB(_,_,block), tCrC_accu);
#else
      gemm(mma, tCrA(_,_,block), tCrB_dq(_,_,block), tCrC_accu);
#endif
    }
  }

  // Epilogue.
  CUTE_UNROLL
  for (int i = 0; i < size(tCrC_accu); i++) {
    tCrC(i) = Element(tCrC_accu(i));
  }
  copy(r2s_copy_c, r2s_tCrC, r2s_tCsC);
  __syncthreads();
  copy_if(s2g_copy_c, tCpC, s2g_tCsC, s2g_tCgC);
}

template <typename Element, int bits, template <typename T> typename Atom,
          typename NumThreads, typename BlockSize>
inline auto make_tiled_copy(NumThreads num_threads, BlockSize block_size) {
  constexpr auto elems_per_copy = Int<bits / sizeof_bits_v<Element>>{};
  constexpr auto thrs_per_block = Int<block_size / elems_per_copy>{};
  constexpr auto rest_threads = Int<num_threads / thrs_per_block>{};
  return make_tiled_copy(
      Copy_Atom<Atom<uint_bit_t<bits>>, Element>{},
      make_layout(make_shape(rest_threads, thrs_per_block), LayoutRight{}),
      make_layout(make_shape(Int<1>{}, elems_per_copy)));
}

template <typename Element, typename Quant, typename GroupSize, typename F>
void qmm_sm80(
    const Element* A,
    const Quant*   B,
    const Element* S,
    const Element* Z,
    Element* C,
    int m, int n, int k, int l,
    GroupSize group_size,
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

  // Define MMA.
  auto pM = Int<16>{};
  auto pN = Int<32>{};
  auto pK = Int<16>{};
  TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
                                Layout<Shape<_1,_2,_1>>{},
                                make_tile(pM, pN, pK));
  auto num_threads = size(mma);

  // Define the A/B smem layouts (static).
  auto swizzle_ab = composition(Swizzle<3,3,3>{},
                                Layout<Shape <_8,Shape <_8, _8>>,
                                       Stride<_8,Stride<_1,_64>>>{});
  auto bP = Int<2>{}; // pipeline
  auto sA_layout = tile_to_shape(swizzle_ab, make_shape(bM, bK, bP));
  auto sB_layout = tile_to_shape(swizzle_ab, make_shape(bN, bK, bP));

  // Define the C smem layouts (static).
  auto sC_layout = make_layout(make_shape(bM, bN), LayoutRight{});

  // Define the S/Z smem layouts (static).
  auto bS = Int<8>{};
  auto sS_layout = make_layout(
      make_shape(bN, make_shape(group_size, bK / group_size), bS, Int<2>{}),
      make_stride(bN * bK / group_size, Stride<_0,_1>{}, bK / group_size, bN * bK / group_size * bS));

  // Define the S/Z gmem layouts (mixed).
  auto gS_layout = make_layout(
      make_shape(n, make_shape(bS, k / group_size / bS), l),
      make_stride(k / group_size, make_stride(Int<1>{}, bS), n * k / group_size));

  // Atoms.
  constexpr int quant_load_bits = 128 / (sizeof_bits_v<Element> / sizeof_bits_v<Quant>);
  TiledCopy g2s_copy_a = make_tiled_copy<Element, 128, SM80_CP_ASYNC_CACHEALWAYS>(num_threads, bK);
#if USE_GEMM
  TiledCopy g2s_copy_b = make_tiled_copy<Element, 128, SM80_CP_ASYNC_CACHEALWAYS>(num_threads, bK);
#else
  TiledCopy g2s_copy_b = make_tiled_copy<Quant, quant_load_bits, SM80_CP_ASYNC_CACHEALWAYS>(num_threads, bK);
#endif
  TiledCopy s2g_copy_c = make_tiled_copy<Element, 128, UniversalCopy>(num_threads, bN);
  TiledCopy g2s_copy_s = make_tiled_copy<Element, 128, SM80_CP_ASYNC_CACHEALWAYS>(num_threads, bS);

  Copy_Atom<SM75_U32x4_LDSM_N, Element> s2r_atom_a;
#if USE_GEMM
  Copy_Atom<SM75_U32x4_LDSM_N, Element> s2r_atom_b;
#else
  Copy_Atom<UniversalCopy<uint_bit_t<2 * sizeof_bits_v<Quant>>>, Quant> s2r_atom_b;
#endif
  Copy_Atom<UniversalCopy<uint_bit_t<2 * sizeof_bits_v<Element>>>, Element> r2s_atom_c;

  auto* kernel = &qmm_sm80_kernel<
      decltype(prob_shape), decltype(cta_tiler),
      Element, Quant,
      decltype(dA), decltype(sA_layout), decltype(g2s_copy_a), decltype(s2r_atom_a),
      decltype(dB), decltype(sB_layout), decltype(g2s_copy_b), decltype(s2r_atom_b),
      decltype(dC), decltype(sC_layout), decltype(s2g_copy_c), decltype(r2s_atom_c),
      decltype(gS_layout), decltype(sS_layout), decltype(g2s_copy_s),
      decltype(mma)>;

  // Set L1 to be SMEM only.
  size_t smem_bytes = sizeof(SharedStorage<Element, Quant,
                                           decltype(sA_layout),
                                           decltype(sB_layout),
                                           decltype(sC_layout),
                                           decltype(sS_layout)>);
  cudaFuncSetAttribute(kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  cudaFuncSetAttribute(kernel,
                       cudaFuncAttributePreferredSharedMemoryCarveout, 100);

  dim3 num_blocks(size(ceil_div(m, bM)), size(ceil_div(n, bN)), l);
  dim3 block_dims(num_threads);
  void* args[] = {
      &prob_shape, &cta_tiler,
      &A, &dA, &sA_layout, &g2s_copy_a, &s2r_atom_a,
      &B, &dB, &sB_layout, &g2s_copy_b, &s2r_atom_b,
      &C, &dC, &sC_layout, &s2g_copy_c, &r2s_atom_c,
      &S, &Z, &gS_layout, &sS_layout, &g2s_copy_s,
      &mma};
  launch_kernel(
      reinterpret_cast<void*>(kernel), num_blocks, block_dims, smem_bytes, args);
}

} // namespace cute_gemm

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
  cudaLaunchConfig_t config = {};
  config.gridDim = num_blocks;
  config.blockDim = block_dims;
  config.dynamicSmemBytes = smem_bytes;
  config.stream = nullptr;
  cudaLaunchKernelExC(&config, func, args);
}

int main(int argc, char** argv) {
  int m = 16;
  if (argc >= 2)
    sscanf(argv[1], "%d", &m);

  int n = 4096;
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

  using Element = cutlass::half_t;
  using Quant = int8_t;

  constexpr int group_size = 64;
  constexpr bool has_bias = cute::sizeof_bits_v<Quant> > 8 || !cutlass::has_negative_zero_v<Quant>;

  thrust::device_vector<Element> d_A(m*k*l);
  thrust::device_vector<Quant>   d_B(n*k*l);    // quantized B
  thrust::device_vector<Element> d_B_dq(n*k*l); // dequantized B
  thrust::device_vector<Element> d_S(n*k*l/group_size); // scales
  thrust::device_vector<Element> d_Z(n*k*l/group_size); // zero points
  thrust::device_vector<Element> d_D(m*n*l);
  thrust::device_vector<Element> d_D_ref(m*n*l);

  int seed = 42;
  cutlass::reference::device::BlockFillRandomUniform(
      d_A.data().get(), d_A.size(), seed, Element(0.1f), Element(-0.1f));
  cutlass::reference::device::BlockFillRandomUniform(
      d_B.data().get(), d_B.size(), seed, Quant(0), Quant(6));
  cutlass::reference::device::BlockFillRandomUniform(
      d_S.data().get(), d_S.size(), seed, Element(0.1f), Element(-0.1f));
  if constexpr (has_bias) {
    cutlass::reference::device::BlockFillRandomUniform(
        d_Z.data().get(), d_Z.size(), seed, Element(0.1f), Element(-0.1f));
  } else {
    cutlass::reference::device::BlockFillSequential(
        d_Z.data().get(), d_Z.size(), Element(0.f), Element(0.f));
  }
  cutlass::reference::device::BlockFillSequential(
      d_D.data().get(), d_D.size(), Element(-1.f), Element(0.f));
  cutlass::reference::device::BlockFillSequential(
      d_D_ref.data().get(), d_D_ref.size(), Element(-1.f), Element(0.f));
#if 0
  cutlass::reference::device::BlockFillSequential(
      d_S.data().get(), d_S.size(), Element(0.05f), Element(0.f));
  cutlass::reference::device::BlockFillSequential(
      d_Z.data().get(), d_Z.size(), Element(0.f), Element(0.f));
#endif

  using namespace cute;
  cudaStream_t stream = nullptr;
  cutlass::dequantize(
      d_B_dq.data().get(),
      d_B.data().get(),
      make_layout(make_shape(n, k, l), make_stride(k, Int<1>{}, n * k)),
      d_S.data().get(),
      d_Z.data().get(),
      make_layout(make_shape(n, k / group_size, l), make_stride(k / group_size, Int<1>{}, n * k / group_size)),
      group_size,
      stream);

  // Run once
  cute_gemm::qmm_sm80(
      d_A.data().get(),
#if USE_GEMM
      d_B_dq.data().get(),
#else
      d_B.data().get(),
#endif
      d_S.data().get(),
      d_Z.data().get(),
      d_D.data().get(),
      m, n, k, l,
      Int<group_size>{},
      launch_kernel);
  CUTE_CHECK_LAST();

  // Verify
  cublas_gemm(
      'T', 'N',
      m, n, k, l,
      d_A.data().get(),
      d_B_dq.data().get(),
      d_D_ref.data().get());
  Element epsilon{1e-2f};
  Element non_zero_floor{1e-4f};
  bool passed = cutlass::reference::device::BlockCompareRelativelyEqual(d_D_ref.data().get(), d_D.data().get(), d_D.size(), epsilon, non_zero_floor);
  if (!passed) {
    thrust::host_vector<Element> cute_result = d_D;
    thrust::host_vector<Element> cublas_result = d_D_ref;
    for (size_t i = 0; i < cute_result.size(); ++i) {
      float delta = fabs(float(cute_result[i]) - float(cublas_result[i]));
      if (delta > 3e-1) {
        printf("!!Wrong result found at %d: %f : %f\n", int(i), float(cute_result[i]), float(cublas_result[i]));
        exit(1);
      }
    }
  }

#if 0
  printf("OK!\n");
#else
  const double tflops = (2.0 * m * n * k * l) * 1e-12;

  const double gemm_bytes =
      bits_to_bytes<double>(double(sizeof_bits_v<Element>) * m * k * l) +
      bits_to_bytes<double>(double(sizeof_bits_v<Element>) * k * n * l) +
      bits_to_bytes<double>(double(sizeof_bits_v<Element>) * m * n * l);
  const double gemm_gibs = gemm_bytes * 1e-9;

#if USE_GEMM
  const double qmm_gibs = gemm_gibs;
#else
  using cutlass::bits_to_bytes;
  const double qmm_bytes =
      bits_to_bytes<double>(double(sizeof_bits_v<Element>) * m * k * l) +
      bits_to_bytes<double>(double(sizeof_bits_v<Quant>  ) * k * n * l) +
      bits_to_bytes<double>(double(sizeof_bits_v<Element>) * k * n * l / group_size) +
      bits_to_bytes<double>(double(sizeof_bits_v<Element>) * k * n * l / group_size) +
      bits_to_bytes<double>(double(sizeof_bits_v<Element>) * m * n * l);
  const double qmm_gibs = qmm_bytes * 1e-9;
#endif

  // Timing iterations
  const int timing_iterations = 100;
  GPU_Clock timer;
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    cute_gemm::qmm_sm80(
        d_A.data().get(),
#if USE_GEMM
        d_B_dq.data().get(),
#else
        d_B.data().get(),
#endif        
        d_S.data().get(),
        d_Z.data().get(),
        d_D.data().get(),
        m, n, k, l,
        Int<group_size>{},
        launch_kernel);
  }
  double cute_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("QMM:     [%6.2f]TFlop/s  [%6.1f]GiB/s  (%6.4f)ms\n",
         tflops / cute_time,
         qmm_gibs / cute_time,
         cute_time * 1000);

  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    cublas_gemm(
        'T', 'N',
        m, n, k, l,
        d_A.data().get(),
        d_B_dq.data().get(),
        d_D.data().get());
  }
  double cublas_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUBLAS:  [%6.2f]TFlop/s  [%6.1f]GiB/s  (%6.4f)ms\n",
         tflops / cublas_time,
         gemm_gibs / cublas_time,
         cublas_time * 1000);

  printf("Speedup: [%5.2f]x\n", cublas_time / cute_time);
#endif

  return 0;
}
