#include <cublas_v2.h>
#include <cute/layout.hpp>
#include <cutlass/numeric_conversion.h>
#include <cutlass/util/GPU_Clock.hpp>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/reference/device/tensor_compare.h>
#include <cutlass/util/mixed_dtype_utils.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace cute_gemm {

using namespace cute;

template <typename Element, typename SmemLayoutA, typename SmemLayoutB>
struct SharedStorage {
  ArrayEngine<Element, cosize_v<SmemLayoutA>> A;
  ArrayEngine<Element, cosize_v<SmemLayoutB>> B;
};

template <typename Q, typename S, typename Z, typename T>
__device__ __forceinline__ void
dequant(const Q& w, const S& s, const Z& z, T out) {
  // Scale must be one element.
  CUTE_STATIC_ASSERT_V(cosize(s.layout()) == Int<1>{});
  CUTE_STATIC_ASSERT_V(cosize(z.layout()) == Int<1>{});
  // Quant must be contiguous.
  auto layout = coalesce(w.layout());
  CUTE_STATIC_ASSERT_V(stride(layout) == Int<1>{});
#if 0
  using Element = typename T::value_type;
  using Quant = typename Q::value_type;
  transform(w, out, [] (Quant q) { return Element(q); } );
  transform(out, s, out, multiplies{});
  transform(out, z, out, plus{});
#else
  // Use cutlass for conversions.
  constexpr int N = size(layout);
  using Element = typename T::value_type;
  using Quant = typename Q::value_type;
  auto& w_vec = *(reinterpret_cast<const cutlass::Array<Quant, N>*>(raw_pointer_cast(w.data())));
  Element scale = s[0];
  Element zero_point = z[0];
  cutlass::NumericArrayConverter<Element, Quant, N> converter;
  auto w_dq = converter(w_vec) * scale + zero_point;
  copy(make_tensor(make_rmem_ptr<Element>(&w_dq), out.layout()), out);
#endif
}

template <typename ProblemShape, typename CtaTiler,
          typename Element, typename Quant,
          typename StrideA, typename SmemLayoutA, typename TiledCopyA,
          typename StrideB, typename SmemLayoutB, typename TiledCopyB,
          typename StrideC,
          typename LayoutS, typename TiledMma>
__global__ void qmm_naive_kernel(
    ProblemShape shape_MNKL, CtaTiler cta_tiler,
    const Element* A, StrideA dA, SmemLayoutA sA_layout, TiledCopyA g2s_copy_a,
    const Quant*   B, StrideB dB, SmemLayoutB sB_layout, TiledCopyB g2s_copy_b,
          Element* C, StrideC dC,
    const Element* S, const Element* Z, LayoutS S_layout, TiledMma mma) {
  CUTE_STATIC_ASSERT_V(size(g2s_copy_a) == size(mma));
  CUTE_STATIC_ASSERT_V(size(g2s_copy_b) == size(mma));
  CUTE_STATIC_ASSERT_V(congruent(select<0,2,3>(shape_MNKL), dA));
  CUTE_STATIC_ASSERT_V(congruent(select<1,2,3>(shape_MNKL), dB));
  CUTE_STATIC_ASSERT_V(congruent(select<0,1,3>(shape_MNKL), dC));

  int thread_idx = int(threadIdx.x);
  auto [m_coord, n_coord, l_coord] = static_cast<uint3>(blockIdx);

  // Represent the full tensors.
  Tensor mA_mkl = make_tensor(make_gmem_ptr(A),        select<0,2,3>(shape_MNKL), dA); // (M,K,L)
  Tensor mB_nkl = make_tensor(make_gmem_ptr<Quant>(B), select<1,2,3>(shape_MNKL), dB); // (N,K,L)
  Tensor mC_mnl = make_tensor(make_gmem_ptr(C),        select<0,1,3>(shape_MNKL), dC); // (M,N,L)

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
  auto n_max_coord = size<1>(shape_MNKL) - size<0>(gB) * n_coord; // N - BLK_N * n_coord

  // Shared memory buffers.
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<Element, SmemLayoutA, SmemLayoutB>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout); // (BLK_M,BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout); // (BLK_N,BLK_K)

  // Partition the copying of A/B/C tiles across the threads.
  ThrCopy g2s_thr_copy_a = g2s_copy_a.get_slice(thread_idx);
  Tensor tAgA = g2s_thr_copy_a.partition_S(gA); // (ACPY,ACPY_M,ACPY_K,k)
  Tensor tAsA = g2s_thr_copy_a.partition_D(sA); // (ACPY,ACPY_M,ACPY_K)
  Tensor tArA = make_fragment_like(tAsA);       // (ACPY,ACPY_M,ACPY_K)

  ThrCopy g2s_thr_copy_b = g2s_copy_b.get_slice(thread_idx);
  Tensor tBgB = g2s_thr_copy_b.partition_S(gB);  // (BCPY,BCPY_N,BCPY_K,k)
  Tensor tBsB = g2s_thr_copy_b.partition_D(sB);  // (BCPY,BCPY_N,BCPY_K)
  Tensor tBrB = make_fragment_like<Quant>(tBsB); // (BCPY,BCPY_M,BCPY_K)
  Tensor tBrB_dq = make_fragment_like(tBsB);     // (BCPY,BCPY_M,BCPY_K)
  Tensor tBgS = g2s_thr_copy_b.partition_S(gS);  // (BCPY,BCPY_N,BCPY_K,k)
  Tensor tBgZ = g2s_thr_copy_b.partition_S(gZ);  // (BCPY,BCPY_N,BCPY_K,k)

  // MMA.
  ThrMMA thr_mma = mma.get_slice(thread_idx);
  Tensor tCsA = thr_mma.partition_A(sA);       // (MMA,MMA_M,MMA_K)
  Tensor tCsB = thr_mma.partition_B(sB);       // (MMA,MMA_N,MMA_K)
  Tensor tCgC = thr_mma.partition_C(gC);       // (MMA,MMA_M,MMA_N)
  Tensor tCrC = thr_mma.make_fragment_C(tCgC); // (MMA,MMA_M,MMA_N)

  // Predicates for m/n bounds.
  Tensor tApA = make_tensor<bool>(make_shape(size<1>(tAsA), size<2>(tAsA)), Stride<_1,_0>{}); // (CPY_M,CPY_K)
  Tensor tBpB = make_tensor<bool>(make_shape(size<1>(tBsB), size<2>(tBsB)), Stride<_1,_0>{}); // (CPY_N,CPY_K)
  Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA))); // (BLK_M,BLK_K)
  Tensor cB = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB))); // (BLK_N,BLK_K)
  Tensor cC = make_identity_tensor(make_shape(size<0>(gC), size<1>(gC))); // (BLK_M,BLK_N)
  Tensor tAcA = g2s_thr_copy_a.partition_S(cA); // (CPY,CPY_M,CPY_K)
  Tensor tBcB = g2s_thr_copy_b.partition_S(cB); // (CPY,CPY_N,CPY_K)
  Tensor tCcC = thr_mma.partition_C(cC);        // (MMA,MMA_M,MMA_N)
  CUTE_UNROLL
  for (int m = 0; m < size<0>(tApA); ++m) {
    tApA(m,0) = get<0>(tAcA(0,m,0)) < m_max_coord;
  }
  CUTE_UNROLL
  for (int n = 0; n < size<0>(tBpB); ++n) {
    tBpB(n,0) = get<0>(tBcB(0,n,0)) < n_max_coord;
  }

  // GMEM => RMEM.
  auto fetch_gmem = [&](int tile) {
    copy_if(g2s_copy_a, tApA, tAgA(_,_,_,tile), tArA);
    copy_if(g2s_copy_b, tBpB, tBgB(_,_,_,tile), tBrB);
  };
  // RMEM => SMEM.
  auto store_smem = [&](int tile) {
    __syncthreads();
    copy(tArA, tAsA);
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tBrB); ++k) {
      CUTE_UNROLL
      for (int n = 0; n < size<1>(tBrB); ++n) {
        dequant(tBrB(_,n,k), tBgS(_,n,k,tile), tBgZ(_,n,k,tile), tBrB_dq(_,n,k));
      }
    }
    copy(tBrB_dq, tBsB);
    __syncthreads();
  };

  // Prefetch first tile.
  fetch_gmem(0);

  // Clear accumulators.
  clear(tCrC);

  // Loop over CTA tiles.
  auto K_TILE_MAX  = size<3>(tAgA);
  for (int tile = 0; tile < K_TILE_MAX; ++tile) {
    store_smem(tile);
    fetch_gmem((tile + 1 < K_TILE_MAX) ? tile + 1 : tile);
    gemm(mma, tCsA, tCsB, tCrC);
  }

  // Epilogue.
  CUTE_UNROLL
  for (int i = 0; i < size(tCrC); ++i) {
    if (get<0>(tCcC(i)) < m_max_coord) {
      tCgC(i) = tCrC(i);
    }
  }
}

template <typename Element>
inline constexpr auto make_mma_atom() {
  if constexpr (std::is_same_v<Element, float>) {
    return UniversalFMA<float>{};
  }
  if constexpr (std::is_same_v<Element, half_t>) {
    return SM80_16x8x16_F32F16F16F32_TN{};
  }
  if constexpr (std::is_same_v<Element, bfloat16_t>) {
    return SM80_16x8x16_F32BF16BF16F32_TN{};
  }
}

template <int TileM, typename Element>
inline constexpr auto make_tiled_mma() {
  constexpr auto atom = make_mma_atom<Element>();
  if constexpr (std::is_same_v<Element, float>) {
    return make_tiled_mma(atom, Layout<Shape<_16,_8,_1>>{});
  } else {
    if constexpr (TileM >= 32) {
      return make_tiled_mma(atom, Layout<Shape<_2,_2,_1>>{}, Tile<_32,_32,_16>{});
    } else {
      return make_tiled_mma(atom, Layout<Shape<_1,_4,_1>>{}, Tile<_16,_32,_16>{});
    }
  }
}

template <typename T, int bits, typename NumThreads>
inline auto make_tiled_copy(NumThreads num_threads) {
  return make_tiled_copy(
      Copy_Atom<UniversalCopy<uint_bit_t<bits>>, T>{},
      make_layout(make_shape(Int<num_threads / 8>{}, Int<8>{}), LayoutRight{}),
      make_layout(make_shape(Int<1>{}, Int<bits / sizeof_bits_v<T>>{})));
}

template <int TileM = 16, typename Element, typename Quant, typename GroupSize, typename F>
void qmm_naive(
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
  auto bK = Int<max(64, group_size)>{};
  auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M,BLK_N,BLK_K)

  // Define MMA.
  TiledMMA mma = make_tiled_mma<TileM, Element>();
  auto num_threads = size(mma);

  // Define the A/B smem layouts (static).
  auto swizzle_ab = composition(Swizzle<3,3,3>{},
                                Layout<Shape <_8,Shape <_8, _8>>,
                                       Stride<_8,Stride<_1,_64>>>{});
  auto sA_layout = tile_to_shape(swizzle_ab, make_shape(bM, bK));
  auto sB_layout = tile_to_shape(swizzle_ab, make_shape(bN, bK));

  // Define layout of scales/biases (mixed).
  auto S_layout = make_layout(
      make_shape(n, make_shape(group_size, k / group_size), l),
      make_stride(k / group_size, Stride<_0, _1>{}, n * k / group_size));

  // Atoms.
  TiledCopy g2s_copy_a = make_tiled_copy<Element, 128>(num_threads);
  TiledCopy g2s_copy_b = make_tiled_copy<Quant, 32>(num_threads);

  auto* kernel = &qmm_naive_kernel<
      decltype(prob_shape), decltype(cta_tiler),
      Element, Quant,
      decltype(dA), decltype(sA_layout), decltype(g2s_copy_a),
      decltype(dB), decltype(sB_layout), decltype(g2s_copy_b),
      decltype(dC),
      decltype(S_layout), decltype(mma)>;

  // Set L1 to be SMEM only.
  size_t smem_bytes = sizeof(SharedStorage<Element, decltype(sA_layout), decltype(sB_layout)>);
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

  dim3 num_blocks(size(ceil_div(m, bM)), size(ceil_div(n, bN)), l);
  dim3 block_dims(num_threads);
  void* args[] = {
      &prob_shape, &cta_tiler,
      &A, &dA, &sA_layout, &g2s_copy_a,
      &B, &dB, &sB_layout, &g2s_copy_b,
      &C, &dC,
      &S, &Z, &S_layout, &mma};
  launch_kernel(reinterpret_cast<void*>(kernel), num_blocks, block_dims, smem_bytes, args);
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
  using Quant = cutlass::uint4b_t;

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
  cute_gemm::qmm_naive(
      d_A.data().get(),
      d_B.data().get(),
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

#if 1
  using cutlass::bits_to_bytes;
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
  const double tflops = (2.0 * m * n * k * l) * 1e-12;

  // Timing iterations
  const int timing_iterations = 100;
  GPU_Clock timer;
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    cute_gemm::qmm_naive(
        d_A.data().get(),
        d_B.data().get(),
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
