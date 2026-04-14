#include <cublas_v2.h>
#include <cute/layout.hpp>
#include <cutlass/numeric_conversion.h>
#include <cutlass/util/GPU_Clock.hpp>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/reference/device/tensor_compare.h>
#include <cutlass/util/mixed_dtype_utils.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "cute_dequant.cuh"

namespace cutlass_gemm {

using namespace cute;

template <typename T, int bits, template <typename U> typename Atom>
inline constexpr auto make_tiled_copy(auto num_threads) {
  return make_tiled_copy(
      Copy_Atom<Atom<uint_bit_t<bits>>, T>{},
      make_layout(make_shape(Int<num_threads / 8>{}, Int<8>{}), LayoutRight{}),
      make_layout(make_shape(Int<1>{}, Int<bits / sizeof_bits_v<T>>{})));
}

template <typename CtaTiler,
          typename TensorA,
          typename TensorB,
          typename TensorS,
          typename TensorZ,
          typename TensorC,
          typename TiledMma>
CUTE_DEVICE void qmm_sm80_mainloop(
    CtaTiler cta_tiler,
    TensorA gA,
    TensorB gB,
    TensorS gS,
    TensorZ gZ,
    TensorC gC,
    TiledMma mma,
    int m_max_coord,
    int thread_idx) {
  // Get the types of operands.
  using Element = decltype(gA)::value_type;
  using Quant = decltype(gB)::value_type;
  using Scale = decltype(gS)::value_type;

  // Define the A/B smem layouts (static).
  auto swizzle_ab = composition(Swizzle<3,3,3>{},
                                Layout<Shape <_8,Shape <_8, _8>>,
                                       Stride<_8,Stride<_1,_64>>>{});
  auto [bM, bN, bK] = cta_tiler;
  auto bP = Int<3>{}; // pipeline
  auto sA_layout = tile_to_shape(swizzle_ab, make_shape(bM, bK, bP));
  auto sB_layout = tile_to_shape(swizzle_ab, make_shape(bN, bK, bP));

  // Define the C smem layouts (static).
  // TODO: Find a better swizzle.
  auto sC_layout = tile_to_shape(swizzle_ab, make_shape(bM, bN));

  // Define copy atoms.
  constexpr int element_bits = sizeof_bits_v<Element>;
  constexpr int quant_bits = sizeof_bits_v<Quant>;
  constexpr int qload = 128 / (element_bits / quant_bits);
  auto num_threads = size(mma);
  TiledCopy g2s_copy_a = make_tiled_copy<Element, 128, SM80_CP_ASYNC_CACHEALWAYS>(num_threads);
  TiledCopy g2s_copy_b = make_tiled_copy<Quant, qload, SM80_CP_ASYNC_CACHEALWAYS>(num_threads);
  TiledCopy s2g_copy_c = make_tiled_copy<Element, 128, UniversalCopy>(num_threads);

  Copy_Atom<SM75_U32x4_LDSM_N, Element> s2r_atom_a;
  Copy_Atom<UniversalCopy<uint_bit_t<2 * quant_bits>>, Quant> s2r_atom_b;
  Copy_Atom<UniversalCopy<uint_bit_t<2 * element_bits>>, Element> r2s_atom_c;
  Copy_Atom<UniversalCopy<Scale>, Scale> g2r_atom_s;

  // Shared memory buffers.
  __shared__ union {
    struct {
      ArrayEngine<Element, cosize_v<decltype(sA_layout)>> A;
      ArrayEngine<Quant,   cosize_v<decltype(sB_layout)>> B;
    } mainloop;
    struct {
      ArrayEngine<Element, cosize_v<decltype(sC_layout)>> C;
    } epilogue;
  } smem;
  Tensor sA = make_tensor(make_smem_ptr(smem.mainloop.A.begin()), sA_layout); // (BLK_M,BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smem.mainloop.B.begin()), sB_layout); // (BLK_N,BLK_K)
  Tensor sC = make_tensor(make_smem_ptr(smem.epilogue.C.begin()), sC_layout); // (BLK_M,BLK_N)

  // Partition the copying of A/B/C tiles across the threads.
  ThrCopy g2s_thr_copy_a = g2s_copy_a.get_slice(thread_idx);
  Tensor tAgA = g2s_thr_copy_a.partition_S(gA); // (ACPY,ACPY_M,ACPY_K,k)
  Tensor tAsA = g2s_thr_copy_a.partition_D(sA); // (ACPY,ACPY_M,ACPY_K,PIPE)

  ThrCopy g2s_thr_copy_b = g2s_copy_b.get_slice(thread_idx);
  Tensor tBgB = g2s_thr_copy_b.partition_S(gB);  // (BCPY,BCPY_N,BCPY_K,k)
  Tensor tBsB = g2s_thr_copy_b.partition_D(sB);  // (BCPY,BCPY_N,BCPY_K,PIPE)

  ThrCopy s2g_thr_copy_c = s2g_copy_c.get_slice(thread_idx);
  Tensor s2g_tCsC = s2g_thr_copy_c.partition_S(sC); // (CCPY,CCPY_M,CCPY_N)
  Tensor s2g_tCgC = s2g_thr_copy_c.partition_D(gC); // (CCPY,CCPY_M,CCPY_N)

  // MMA.
  ThrMMA thr_mma = mma.get_slice(thread_idx);
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0)); // (MMA,MMA_M,MMA_K)
  Tensor tCsB = thr_mma.partition_B(sB(_,_,0));          // (MMA,MMA_N,MMA_K)
  Tensor tCrB = make_fragment_like<Quant>(tCsB);         // (MMA,MMA_N,MMA_K)
  Tensor tCrB_dq = make_fragment_like<Element>(tCsB);    // (MMA,MMA_N,MMA_K)
  Tensor tCgC = thr_mma.partition_C(gC);                 // (MMA,MMA_M,MMA_N)
  Tensor tCrC_accu = make_fragment_like<float>(tCgC);    // (MMA,MMA_M,MMA_N)
  Tensor tCrC = make_fragment_like<Element>(tCgC);       // (MMA,MMA_M,MMA_N)

  Tensor tCgS = thr_mma.partition_B(gS);         // (MMA,MMA_N,MMA_K,k)
  Tensor tCrS = make_tensor_like(tCgS(_,_,_,0)); // (MMA,MMA_N,MMA_K)
  Tensor tCgZ = thr_mma.partition_B(gZ);         // (MMA,MMA_N,MMA_K,k)
  Tensor tCrZ = make_tensor_like(tCgZ(_,_,_,0)); // (MMA,MMA_N,MMA_K)

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

  TiledCopy g2r_copy_s = make_tiled_copy_B(g2r_atom_s, mma);
  ThrCopy g2r_thr_copy_s = g2r_copy_s.get_slice(thread_idx);
  Tensor g2r_tCgS = g2r_thr_copy_s.partition_S(gS); // (BCPY,MMA_N,MMA_K,k)
  Tensor g2r_tCrS = g2r_thr_copy_s.retile_D(tCrS);  // (BCPY,MMA_N,MMA_K)
  Tensor g2r_tCgZ = g2r_thr_copy_s.partition_S(gZ); // (BCPY,MMA_N,MMA_K,k)
  Tensor g2r_tCrZ = g2r_thr_copy_s.retile_D(tCrZ);  // (BCPY,MMA_N,MMA_K)

  // Predicates for m bound.
  Tensor tApA = make_tensor<bool>(make_shape(size<1>(tAsA), size<2>(tAsA)), Stride<_1,_0>{});         // (CPY_M,CPY_K)
  Tensor tCpC = make_tensor<bool>(make_shape(size<1>(s2g_tCsC), size<2>(s2g_tCsC)), Stride<_1,_0>{}); // (CPY_M,CPY_N)
  Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA))); // (BLK_M,BLK_K)
  Tensor cC = make_identity_tensor(make_shape(size<0>(sC), size<1>(sC))); // (BLK_M,BLK_N)
  Tensor tAcA = g2s_thr_copy_a.partition_D(cA); // (CPY,CPY_M,CPY_K)
  Tensor tCcC = s2g_thr_copy_c.partition_D(cC); // (CPY,CPY_M,CPY_N)
  CUTE_UNROLL
  for (int m = 0; m < size<0>(tApA); ++m) {
    tApA(m,0) = get<0>(tAcA(0,m,0)) < m_max_coord;
  }
  CUTE_UNROLL
  for (int m = 0; m < size<0>(tCpC); ++m) {
    tCpC(m,0) = get<0>(tCcC(0,m,0)) < m_max_coord;
  }

  auto K_PIPE_MAX = size<3>(tAsA);
  int smem_pipe_read = 0;
  int smem_pipe_write = 0;

  // Copy A/B: GMEM => SMEM.
  auto fetch_gmem = [&](int tile) {
    copy_if(g2s_copy_a, tApA, tAgA(_,_,_,tile), tAsA(_,_,_,smem_pipe_write));
    copy(g2s_copy_b, tBgB(_,_,_,tile), tBsB(_,_,_,smem_pipe_write));
    cp_async_fence();
    smem_pipe_write = (smem_pipe_write + 1) % K_PIPE_MAX;
  };
  // Copy S/Z: GMEM => RMEM.
  auto fetch_scales = [&](int tile) {
    copy(g2r_copy_s, g2r_tCgS(_,_,_,tile), g2r_tCrS);
    if constexpr (quant_has_bias_v<Quant>) {
      copy(g2r_copy_s, g2r_tCgZ(_,_,_,tile), g2r_tCrZ);
    }
  };
  // Copy A/B: SMEM => RMEM.
  auto fetch_smem = [&](auto block) {
    copy(s2r_atom_a, s2r_tCsA(_,_,block,smem_pipe_read), s2r_tCrA(_,_,block));
    copy(s2r_atom_b, s2r_tCsB(_,_,block,smem_pipe_read), s2r_tCrB(_,_,block));
    CUTE_UNROLL
    for (int n = 0; n < size<1>(tCrB); ++n) {
      cute_vectorized_dequant(
          tCrB(_,n,block),
          tCrS(_,n,block),
          tCrZ(_,n,block),
          tCrB_dq(_,n,block));
    }
  };

  auto K_TILE_MAX = size<3>(tAgA);
  auto K_BLOCK_MAX = size<2>(tCrA);

  // Prefetch beginning tiles.
  int tile_pipe = 0;
  CUTE_UNROLL
  for (; tile_pipe < K_PIPE_MAX - 1; ++tile_pipe) {
    fetch_gmem(tile_pipe);
  }

  // Clear accumulators.
  clear(tCrC_accu);

  // Prefetch first block.
  if constexpr (K_BLOCK_MAX > 1) {
    cp_async_wait<K_PIPE_MAX - 2>();
    __syncthreads();
    fetch_scales(0);
    fetch_smem(Int<0>{});
  }

  // Loop over CTA tiles.
  for (int tile = 0; tile < K_TILE_MAX; ++tile) {
    // Unroll MMA blocks.
    CUTE_UNROLL
    for (int block = 0; block < K_BLOCK_MAX; ++block) {
      // Wait for last tile.
      if (block == K_BLOCK_MAX - 1) {
        smem_pipe_read = (smem_pipe_read + 1) % K_PIPE_MAX;
        cp_async_wait<K_PIPE_MAX - 2>();
        __syncthreads();
        fetch_scales((tile + 1 < K_TILE_MAX) ? tile + 1 : tile);
      }
      // Prefetch next block.
      fetch_smem((block + 1) % K_BLOCK_MAX);
      // Prefetch next tile.
      if (block == 0) {
        fetch_gmem(tile_pipe);
        tile_pipe = (tile_pipe + 1 < K_TILE_MAX) ? tile_pipe + 1 : tile_pipe;
      }
      // MMA.
      gemm(mma, tCrA(_,_,block), tCrB_dq(_,_,block), tCrC_accu);
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

inline constexpr auto make_scales_layout(auto n, auto k, auto l, auto group_size) {
  return make_layout(
      make_shape(n, make_shape(group_size, k / group_size), l),
      make_stride(k / group_size, Stride<_0,_1>{}, n * k / group_size));
}

template <int TileM>
inline constexpr auto make_cta_tiler(auto group_size) {
  auto bM = Int<TileM>{};
  auto bN = Int<128>{};
  auto bK = Int<max(64, group_size)>{};
  return make_shape(bM, bN, bK);
}

template <int TileM, typename Element>
inline constexpr auto make_tiled_mma() {
  using Atom = std::conditional_t<
      std::is_same_v<Element, half_t>,
      SM80_16x8x16_F32F16F16F32_TN,
      std::conditional_t<
          std::is_same_v<Element, bfloat16_t>,
          SM80_16x8x16_F32BF16BF16F32_TN,
          UniversalFMA<float>>>;
  if constexpr (TileM >= 32) {
    return make_tiled_mma(Atom{}, Layout<Shape<_2,_2,_1>>{}, Tile<_32,_32,_16>{});
  } else {
    return make_tiled_mma(Atom{}, Layout<Shape<_1,_4,_1>>{}, Tile<_16,_32,_16>{});
  }
}

template <typename Element, typename Quant, typename Scale,
          typename ProblemShape,
          typename CtaTiler,
          typename StrideA,
          typename StrideB,
          typename LayoutS,
          typename StrideC,
          typename TiledMma>
__global__
__launch_bounds__(decltype(size(TiledMma{}))::value)
void qmm_sm80_kernel(
    ProblemShape shape_MNKL, CtaTiler cta_tiler,
    const Element* A, StrideA dA,
    const Quant* B, StrideB dB,
    const Scale* S, const Element* Z, LayoutS S_layout,
    const uint32_t* lhs_indices, const uint32_t* rhs_indices,
    Element* C, StrideC dC,
    TiledMma mma) {
  CUTE_STATIC_ASSERT_V(congruent(select<0,2,3>(shape_MNKL), dA));
  CUTE_STATIC_ASSERT_V(congruent(select<1,2,3>(shape_MNKL), dB));
  CUTE_STATIC_ASSERT_V(congruent(select<0,1,3>(shape_MNKL), dC));

  int thread_idx = int(threadIdx.x);
  auto [m_coord, n_coord, l_coord] = static_cast<uint3>(blockIdx);

  // For gather, use index lookup for input batch slicing.
  uint32_t a_batch = lhs_indices ? lhs_indices[l_coord] : l_coord;
  uint32_t b_batch = rhs_indices ? rhs_indices[l_coord] : l_coord;

  // Represent the full tensors.
  Tensor mA_mkl = make_tensor(make_gmem_ptr(A),        select<0,2,3>(shape_MNKL), dA); // (M,K,L)
  Tensor mB_nkl = make_tensor(make_gmem_ptr<Quant>(B), select<1,2,3>(shape_MNKL), dB); // (N,K,L)
  Tensor mC_mnl = make_tensor(make_gmem_ptr(C),        select<0,1,3>(shape_MNKL), dC); // (M,N,L)

  Tensor mS_nkl = make_tensor(make_gmem_ptr(S), S_layout); // (N,(group_size,K/group_size),L)
  Tensor mZ_nkl = make_tensor(make_gmem_ptr(Z), S_layout); // (N,(group_size,K/group_size),L)

  // Get batch slice.
  Tensor mA = mA_mkl(_,_,a_batch); // (M,K)
  Tensor mB = mB_nkl(_,_,b_batch); // (N,K)
  Tensor mC = mC_mnl(_,_,l_coord); // (M,N)

  Tensor mS = mS_nkl(_,_,b_batch); // (N,(group_size,K/group_size))
  Tensor mZ = mZ_nkl(_,_,b_batch); // (N,(group_size,K/group_size))

  // Get the appropriate blocks for this thread block.
  auto cta_coord = make_coord(m_coord, n_coord, _); // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{}); // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{}); // (BLK_M,BLK_N)

  Tensor gS = local_tile(mS, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)
  Tensor gZ = local_tile(mZ, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)

  // Compute tile residues for predication.
  auto m_max_coord = size<0>(shape_MNKL) - size<0>(gA) * m_coord; // M - BLK_M * m_coord

  qmm_sm80_mainloop(
      cta_tiler,
      gA,
      gB,
      gS,
      gZ,
      gC,
      mma,
      m_max_coord,
      thread_idx);
}

template <int TileM,
          typename Element, typename Quant, typename Scale>
void qmm_sm80(
    const Element* A,
    const Quant*   B,
    const Scale* S,
    const Element* Z,
    const uint32_t* lhs_indices,
    const uint32_t* rhs_indices,
    Element* C,
    int m, int n, int k, int l,
    bool broadcast_b,
    auto group_size,
    auto&& launch_kernel) {
  // Define shapes (dynamic).
  auto shape_MNKL = make_shape(m, n, k, l); // (M,N,K,L)

  // Define layouts (mixed).
  auto dA = make_stride(k, Int<1>{}, m * k); // (dM,dK,dL)
  auto dB = make_stride(k, Int<1>{}, n * k); // (dN,dK,dL)
  auto dC = make_stride(n, Int<1>{}, m * n); // (dM,dN,dL)
  auto S_layout = make_scales_layout(n, k, l, group_size);

  // Handle broadcasting.
  if (broadcast_b) {
    get<2>(dB) = 0;
    get<2>(stride(S_layout)) = 0;
  }

  // Define CTA tile sizes (static).
  auto cta_tiler = make_cta_tiler<TileM>(group_size);

  // Define MMA.
  TiledMMA mma = make_tiled_mma<TileM, Element>();
  auto num_threads = size(mma);

  auto* kernel = &qmm_sm80_kernel<
      Element, Quant, Scale,
      decltype(shape_MNKL),
      decltype(cta_tiler),
      decltype(dA),
      decltype(dB),
      decltype(S_layout),
      decltype(dC),
      decltype(mma)>;

  dim3 num_blocks{uint32_t(ceil_div(m, size<0>(cta_tiler))),
                  uint32_t(ceil_div(n, size<1>(cta_tiler))),
                  uint32_t(l)};
  dim3 block_dims{num_threads};
  void* args[] = {
      &shape_MNKL, &cta_tiler,
      &A, &dA,
      &B, &dB,
      &S, &Z, &S_layout,
      &lhs_indices, &rhs_indices,
      &C, &dC,
      &mma};
  launch_kernel(reinterpret_cast<void*>(kernel), num_blocks, block_dims, args);
}

} // namespace cutlass_gemm

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

void launch_kernel(void* func, dim3 num_blocks, dim3 block_dims, void** args) {
  cudaLaunchConfig_t config = {};
  config.gridDim = num_blocks;
  config.blockDim = block_dims;
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
  using Quant = uint8_t;
  using Scale = Element;

  constexpr int group_size = 64;
  constexpr int tile_m = 16;
  constexpr bool has_bias = !cutlass::has_negative_zero_v<Quant>;

  thrust::device_vector<Element> d_A(m*k*l);
  thrust::device_vector<Quant>   d_B(n*k*l);    // quantized B
  thrust::device_vector<Element> d_B_dq(n*k*l); // dequantized B
  thrust::device_vector<Scale> d_S(n*k*l/group_size); // scales
  thrust::device_vector<Element> d_Z(n*k*l/group_size); // zero points
  thrust::device_vector<Element> d_D(m*n*l);
  thrust::device_vector<Element> d_D_ref(m*n*l);

  int seed = 42;
  cutlass::reference::device::BlockFillRandomUniform(
      d_A.data().get(), d_A.size(), seed, Element(0.1f), Element(-0.1f));
  cutlass::reference::device::BlockFillRandomUniform(
      d_B.data().get(), d_B.size(), seed, Quant(0), Quant(16));
  cutlass::reference::device::BlockFillRandomUniform(
      d_S.data().get(), d_S.size(), seed, Scale(0.1f), Scale(-0.1f));
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
  cutlass_gemm::qmm_sm80<tile_m>(
      d_A.data().get(),
      d_B.data().get(),
      d_S.data().get(),
      d_Z.data().get(),
      nullptr,
      nullptr,
      d_D.data().get(),
      m, n, k, l,
      false,
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

#if 1
  const double tflops = (2.0 * m * n * k * l) * 1e-12;

  const double qmm_bytes =
      bits_to_bytes<double>(double(sizeof_bits_v<Element>) * m * k * l) +
      bits_to_bytes<double>(double(sizeof_bits_v<Quant>  ) * k * n * l) +
      bits_to_bytes<double>(double(sizeof_bits_v<Element>) * k * n * l / group_size) +
      bits_to_bytes<double>(double(sizeof_bits_v<Element>) * k * n * l / group_size) +
      bits_to_bytes<double>(double(sizeof_bits_v<Element>) * m * n * l);
  const double qmm_gibs = qmm_bytes * 1e-9;

  const double gemm_bytes =
      bits_to_bytes<double>(double(sizeof_bits_v<Element>) * m * k * l) +
      bits_to_bytes<double>(double(sizeof_bits_v<Element>) * k * n * l) +
      bits_to_bytes<double>(double(sizeof_bits_v<Element>) * m * n * l);
  const double gemm_gibs = gemm_bytes * 1e-9;

  // Timing iterations
  const int timing_iterations = 100;
  GPU_Clock timer;
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    cutlass_gemm::qmm_sm80<tile_m>(
        d_A.data().get(),
        d_B.data().get(),
        d_S.data().get(),
        d_Z.data().get(),
        nullptr,
        nullptr,
        d_D.data().get(),
        m, n, k, l,
        false,
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
