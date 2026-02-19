// nvcc qmm.cu -I cutlass/include -I cutlass/tools/util/include --expt-relaxed-constexpr -std=c++20 -lcublas -gencode arch=compute_121a,code=sm_121a -o qmm
// ./qmm M N K L

#include <cute/tensor.hpp>

#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "cutlass/util/GPU_Clock.hpp"

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

namespace cute_gemm {

using namespace cute;

template <typename ProblemShape, typename CtaTiler,
          typename Element, typename Quant,
          typename AStride, typename ASmemLayout, typename TiledCopyA,
          typename BStride, typename BSmemLayout, typename TiledCopyB,
          typename SLayout, typename CStride, typename TiledMma>
__global__ void qmm_impl(
    ProblemShape shape_MNKL, CtaTiler cta_tiler,
    const Element* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
    const Quant* B,   BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
    const Element* S, const Element* Z, SLayout S_layout,
    Element* C, CStride dC, TiledMma mma) {
  CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));
  CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));
  CUTE_STATIC_ASSERT_V(congruent(select<0,2,3>(shape_MNKL), dA));
  CUTE_STATIC_ASSERT_V(congruent(select<1,2,3>(shape_MNKL), dB));
  CUTE_STATIC_ASSERT_V(congruent(select<0,1,3>(shape_MNKL), dC));

  int thread_idx = int(threadIdx.x);
  auto [m_coord, n_coord, l_coord] = static_cast<uint3>(blockIdx);

  // Represent the full tensors.
  Tensor mA_mkl = make_tensor(make_gmem_ptr(A), select<0,2,3>(shape_MNKL), dA); // (M,K,L)
  Tensor mB_nkl = make_tensor(make_gmem_ptr(B), select<1,2,3>(shape_MNKL), dB); // (N,K,L)
  Tensor mS_nkl = make_tensor(make_gmem_ptr(S), S_layout);                      // (N,(group_size,K/group_size),L)
  Tensor mZ_nkl = make_tensor(make_gmem_ptr(Z), S_layout);                      // (N,(group_size,K/group_size),L)
  Tensor mC_mnl = make_tensor(make_gmem_ptr(C), select<0,1,3>(shape_MNKL), dC); // (M,N,L)

  // Get batch slice.
  Tensor mA = mA_mkl(_,_,l_coord); // (M,K)
  Tensor mB = mB_nkl(_,_,l_coord); // (N,K)
  Tensor mS = mS_nkl(_,_,l_coord); // (N,(group_size,K/group_size))
  Tensor mZ = mZ_nkl(_,_,l_coord); // (N,(group_size,K/group_size))
  Tensor mC = mC_mnl(_,_,l_coord); // (M,N)

  // Get the appropriate blocks for this thread block.
  auto cta_coord = make_coord(m_coord, n_coord, _); // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{}); // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)
  Tensor gS = local_tile(mS, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)
  Tensor gZ = local_tile(mZ, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{}); // (BLK_M,BLK_N)

  auto m_max_coord = size<0>(shape_MNKL) - size<0>(gA) * m_coord; // M - BLK_M * m_coord

  // Shared memory buffers.
  __shared__ Element smemA[cosize_v<ASmemLayout>];
  __shared__ Element smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (BLK_M,BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // (BLK_N,BLK_K)

  // Partition the copying of A and B tiles across the threads.
  ThrCopy thr_copy_a = copy_a.get_slice(thread_idx);
  Tensor tAgA = thr_copy_a.partition_S(gA); // (ACPY,ACPY_M,ACPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA); // (ACPY,ACPY_M,ACPY_K)
  Tensor tArA = make_fragment_like(tAsA);   // (ACPY,ACPY_M,ACPY_K)

  ThrCopy thr_copy_b = copy_b.get_slice(thread_idx);
  Tensor tBgB = thr_copy_b.partition_S(gB);       // (BCPY,BCPY_N,BCPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(sB);       // (BCPY,BCPY_N,BCPY_K)
  Tensor tBrB = make_fragment_like(tBsB);         // (BCPY,BCPY_N,BCPY_K)
  Tensor tBrBq = make_fragment_like<Quant>(tBsB); // (BCPY,BCPY_N,BCPY_K)
  Tensor tBgS = thr_copy_b.partition_S(gS);       // (BCPY,BCPY_N,BCPY_K,k)
  Tensor tBgZ = thr_copy_b.partition_S(gZ);       // (BCPY,BCPY_N,BCPY_K,k)

  // MMA.
  ThrMMA thr_mma = mma.get_slice(thread_idx);
  Tensor tCsA = thr_mma.partition_A(sA); // (MMA,MMA_M,MMA_K)
  Tensor tCsB = thr_mma.partition_B(sB); // (MMA,MMA_N,MMA_K)
  Tensor tCgC = thr_mma.partition_C(gC); // (MMA,MMA_M,MMA_N)

  // Accumulators.
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);

  // Predicates for m bounds.
  Tensor tApA = make_tensor<bool>(make_shape(size<1>(tAsA), size<2>(tAsA)),
                                  Stride<_1,_0>{});                       // (ACPY_M,ACPY_K)
  Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA))); // (BLK_M,BLK_K)
  Tensor cC = make_identity_tensor(make_shape(size<0>(gC), size<1>(gC))); // (BLK_M,BLK_N)
  Tensor tAcA = thr_copy_a.partition_S(cA);                               // (ACPY,ACPY_M,ACPY_K)
  Tensor tCcC = thr_mma.partition_C(cC);                                  // (MMA,MMA_M,MMA_N)
  CUTE_UNROLL
  for (int m = 0; m < size<0>(tApA); ++m) {
    tApA(m,0) = get<0>(tAcA(0,m,0)) < m_max_coord;
  }

  // Copy gmem to rmem for k_tile=0.
  copy_if(copy_a, tApA, tAgA(_,_,_,0), tArA);
  copy(copy_b, tBgB(_,_,_,0), tBrBq);

  auto K_TILE_MAX = size<3>(tAgA);

  // Main loop.
  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
    __syncthreads();

    // Dequantize B and then copy A/B to smem.
    Tensor scale = tBgS(_,_,_,k_tile);
    Tensor zero_point = tBgZ(_,_,_,k_tile);
    for (int i = 0; i < size(tBrB); ++i) {
      tBrB(i) = tBrBq(i) * scale(i) + zero_point(i);
    }
    copy(tArA, tAsA);
    copy(tBrB, tBsB);
    __syncthreads();

    // Copy gmem to rmem for k_tile+1 with tA|tB thread-partitioned tensors.
    int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
    copy_if(copy_a, tApA, tAgA(_,_,_,k_tile_next), tArA);
    copy(copy_b, tBgB(_,_,_,k_tile_next), tBrBq);

    // Compute gemm on mma-partitioned smem.
    gemm(mma, tCsA, tCsB, tCrC);
  }

  CUTE_UNROLL
  for (int i = 0; i < size(tCrC); ++i) {
    if (get<0>(tCcC(i)) < m_max_coord) {
      tCgC(i) = tCrC(i);
    }
  }
}

template <typename Element, typename GroupSize, typename F>
inline auto dispatch_swizzle(F&& f) {
  if constexpr (sizeof(Element) == 4) {
    if constexpr (GroupSize::value <= 32) {
      f(Swizzle<3,2,3>{});
    } else {
      f(Swizzle<3,3,3>{});
    }
  } else {
    if constexpr (GroupSize::value <= 32) {
      f(Swizzle<2,3,3>{});
    } else {
      f(Swizzle<3,3,3>{});
    }
  }
}

template <typename Element, typename F>
inline auto dispatch_mma(bool is_sm80, F&& f) {
  if (is_sm80) {
    if constexpr (std::is_same_v<Element, float>) {
      f(make_tiled_mma(SM80_16x8x8_F32TF32TF32F32_TN{},
                       Layout<Shape<_1,_4,_1>>{},
                       Tile<_16,_32,_8>{}));
      return;
    } else if constexpr (std::is_same_v<Element, cute::half_t>) {
      f(make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
                       Layout<Shape<_1,_4,_1>>{},
                       Tile<_16,_32,_16>{}));
      return;
    }
  }
  f(make_tiled_mma(F32FMA<Element, Element>{},
                   Layout<Shape<_16,_8,_1>>{}));
}

template <typename GroupSize, typename Element, typename Quant, typename F>
void qmm(
    int m, int n, int k, int l,
    GroupSize group_size,
    const Element* A,
    const Quant* B,
    const Element* S,
    const Element* Z,
    Element* C,
    bool is_sm80,
    F&& launch_kernel) {
  // Define shapes (dynamic).
  auto prob_shape = make_shape(m, n, k, l); // (M,N,K,L)

  // Define TN strides (mixed).
  auto dA = make_stride(k, Int<1>{}, m * k); // (dM,dK,dL)
  auto dB = make_stride(k, Int<1>{}, n * k); // (dN,dK,dL)
  auto dC = make_stride(n, Int<1>{}, m * n); // (dM,dN,dL)

  // Define layout of scales (mixed).
  auto S_layout = make_layout(
      make_shape(n, make_shape(group_size, k / group_size), l),
      make_stride(k / group_size, make_stride(Int<0>{}, Int<1>{}), n * k / group_size));

  // Define CTA tile sizes (static).
  auto bM = Int<16>{};
  auto bN = Int<128>{};
  auto bK = Int<max(64,group_size)>{};
  auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M,BLK_N,BLK_K)

  TiledCopy copy_a = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                                     Layout<Shape<_16,_8>,Stride<_8,_1>>{},
                                     Layout<Shape< _1,_8>>{});
  TiledCopy copy_b = make_tiled_copy(Copy_Atom<UniversalCopy<uint32_t>, Quant>{},
                                     Layout<Shape<_16,_8>,Stride<_8,_1>>{},
                                     Layout<Shape<_1,Int<32/sizeof_bits<Quant>::value>>>{});

  // Define the smem layouts (static).
  dispatch_swizzle<Element, GroupSize>([&](auto swizzle) {
    auto swizzle_atom = composition(swizzle,
                                    Layout<Shape<_8,GroupSize>,
                                           Stride<GroupSize,_1>>{});
    auto sA_layout = tile_to_shape(swizzle_atom, make_shape(bM, bK));
    auto sB_layout = tile_to_shape(swizzle_atom, make_shape(bN, bK));

    // Create tiled MMA.
    dispatch_mma<Element>(is_sm80, [&](auto mma) {
      // Launch kernel.
      auto* kernel = &qmm_impl<
          decltype(prob_shape), decltype(cta_tiler),
          Element, Quant,
          decltype(dA), decltype(sA_layout), decltype(copy_a),
          decltype(dB), decltype(sB_layout), decltype(copy_b),
          decltype(S_layout), decltype(dC), decltype(mma)>;
      dim3 num_blocks(size(ceil_div(m, bM)), size(ceil_div(n, bN)), l);
      dim3 block_dims(size(mma));
      void* args[] = {
          &prob_shape, &cta_tiler,
          &A, &dA, &sA_layout, &copy_a,
          &B, &dB, &sB_layout, &copy_b,
          &S, &Z, &S_layout,
          &C, &dC, &mma};
      launch_kernel(reinterpret_cast<void*>(kernel), num_blocks, block_dims, 0, args);
    });
  });
}

}  // namespace cute_gemm

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

  using Element = cute::half_t;
  using Quant = uint8_t;

  std::cout << "M = " << m << std::endl;
  std::cout << "N = " << n << std::endl;
  std::cout << "K = " << k << std::endl;
  std::cout << "L = " << l << std::endl;

  CUTE_CHECK_ERROR(cudaSetDevice(0));
  cudaDeviceProp device_prop;
  CUTE_CHECK_ERROR(cudaGetDeviceProperties(&device_prop, 0));
  bool is_sm80 = device_prop.major >= 8;

  constexpr int group_size = 64;

  thrust::host_vector<Element> h_A(m*k*l);
  thrust::host_vector<Quant> h_B(n*k*l); // quantized B
  thrust::host_vector<Element> h_S(n*k*l/group_size); // scales
  thrust::host_vector<Element> h_Z(n*k*l/group_size); // zero points
  thrust::host_vector<Element> h_B_ref(n*k*l); // dequantized B
  thrust::host_vector<Element> h_C(m*n*l);

  for (int j = 0; j < h_A.size(); ++j) h_A[j] = static_cast<Element>(2*(rand() / double(RAND_MAX)) - 1);
  for (int j = 0; j < h_B.size(); ++j) h_B[j] = static_cast<Quant>(rand() % 16);
  for (int j = 0; j < h_S.size(); ++j) h_S[j] = static_cast<Element>(0.01f * (rand() / double(RAND_MAX)) + 0.001f);
  for (int j = 0; j < h_Z.size(); ++j) h_Z[j] = static_cast<Element>(0.1f * (rand() / double(RAND_MAX)) + 0.01f);
  for (int j = 0; j < h_C.size(); ++j) h_C[j] = static_cast<Element>(-1);

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
      m, n, k, l, cute::Int<group_size>{},
      d_A.data().get(),
      d_B.data().get(),
      d_S.data().get(),
      d_Z.data().get(),
      d_C.data().get(),
      is_sm80,
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
