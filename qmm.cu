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

template <class ElementA,
          class ElementB,
          class SmemLayoutA,
          class SmemLayoutB>
struct SharedStorage {
  ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;
};

template <typename ProblemShape, typename CtaTiler,
          typename Element, typename Quant,
          typename AStride, typename ASmemLayout, typename TiledCopyA, typename S2RAtomA,
          typename BStride, typename BSmemLayout, typename TiledCopyB,
          typename SLayout, typename CStride, typename TiledMma>
__global__ void qmm_impl(
    ProblemShape shape_MNKL, CtaTiler cta_tiler,
    const Element* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a, S2RAtomA s2r_atom_a,
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
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<Element, Quant, ASmemLayout, BSmemLayout>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout); // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout); // (BLK_N,BLK_K,PIPE)

  // Partition the copying of A and B tiles across the threads.
  ThrCopy thr_copy_a = copy_a.get_slice(thread_idx);
  Tensor tAgA = thr_copy_a.partition_S(gA); // (ACPY,ACPY_M,ACPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA); // (ACPY,ACPY_M,ACPY_K,PIPE)

  ThrCopy thr_copy_b = copy_b.get_slice(thread_idx);
  Tensor tBgB = thr_copy_b.partition_S(gB); // (BCPY,BCPY_N,BCPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(sB); // (BCPY,BCPY_N,BCPY_K,PIPE)
  Tensor tBgS = thr_copy_b.partition_S(gS); // (BCPY,BCPY_N,BCPY_K,k)
  Tensor tBgZ = thr_copy_b.partition_S(gZ); // (BCPY,BCPY_N,BCPY_K,k)

  // MMA.
  ThrMMA thr_mma = mma.get_slice(thread_idx);
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));    // (MMA,MMA_M,MMA_K)
  Tensor tCsB = thr_mma.partition_B(sB);                    // (MMA,MMA_N,MMA_K,PIPE)
  Tensor tCrB = make_fragment_like(tCsB(_,_,_,0));          // (MMA,MMA_N,MMA_K)
  Tensor tCrB_dequant = make_fragment_like<Element>(tCrB);  // (MMA,MMA_N,MMA_K)
  Tensor tCgS = thr_mma.partition_B(gB);                    // (MMA,MMA_N,MMA_K)
  Tensor tCgZ = thr_mma.partition_B(gZ);                    // (MMA,MMA_N,MMA_K)
  Tensor tCgC = thr_mma.partition_C(gC);                    // (MMA,MMA_M,MMA_N)
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);              // (MMA,MMA_M,MMA_N)

  if(thread0()) {
    print("tCrA: "); print(tCrA); print("\n");
    print("tCrB: "); print(tCrB); print("\n");
  }

#if 0
  // Copy Atom retiling.
  TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
  ThrCopy s2r_thr_copy_a = s2r_copy_a.get_slice(thread_idx);
  Tensor tCsA           = s2r_thr_copy_a.partition_S(sA); // (ACPY,MMA_M,MMA_K,PIPE)
  Tensor tCrA_copy_view = s2r_thr_copy_a.retile_D(tCrA);  // (ACPY,MMA_M,MMA_K)

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
  clear(tCrC);

  // RMEM pipeline size (static).
  auto K_BLOCK_MAX = size<2>(tCrA);
  // SMEM iterator.
  int smem_pipe_read  = 0;
  int smem_pipe_write = K_PIPE_MAX - 1;

  // Prefetch SMEM => RMEM for first block in RMEM pipeline.
  Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
  Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);
  if (K_BLOCK_MAX > 1) {
    cp_async_wait<K_PIPE_MAX - 2>(); // wait first tile
    __syncthreads();
    copy(s2r_atom_a, tCsA_p(_,_,0), tCrA_copy_view(_,_,0));
    copy(tCsB_p(_,_,0), tCrB(_,_,0));
  }

  // Loop over k-tiles in SMEM pipeline.
  while (k_tile_count > -(K_PIPE_MAX - 1)) {
    // Loop over blocks in RMEM pipeline.
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
      // Wait for next tile.
      if (k_block == K_BLOCK_MAX - 1) {
        tCsA_p = tCsA(_,_,_,smem_pipe_read);
        tCsB_p = tCsB(_,_,_,smem_pipe_read);
        cp_async_wait<K_PIPE_MAX - 2>();
        __syncthreads();
      }

      // Prefetch SMEM => RMEM for next block.
      int k_block_next = (k_block + 1) % K_BLOCK_MAX;
      copy(s2r_atom_a, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
      copy(tCsB_p(_,_,k_block_next), tCrB(_,_,k_block_next));

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
        smem_pipe_read = (smem_pipe_read == K_PIPE_MAX-1) ? 0 : smem_pipe_read + 1;
      }

      // Dequantize B.
      #if 0
      Tensor scale = tCgS(_,_,k_block);
      Tensor zero_point = tCgZ(_,_,k_block);
      for (int i = 0; i < size(tCrB); ++i) {
        tCrB_dequant(i) = tCrB(i) * scale(i) + zero_point(i);
      }
      #endif

      // GEMM for current block.
      gemm(mma, tCrA(_,_,k_block), tCrB_dequant(_,_,k_block), tCrC);
    }
  }

  // Write accumulator to GMEM.
  CUTE_UNROLL
  for (int i = 0; i < size(tCrC); ++i) {
    if (get<0>(tCcC(i)) < m_max_coord) {
      tCgC(i) = tCrC(i);
    }
  }
#endif
}

template <typename Element, typename F>
inline auto dispatch_mma(bool is_sm80, F&& f) {
  if (is_sm80) {
    if constexpr (std::is_same_v<Element, float>) {
      f(make_tiled_mma(SM80_16x8x8_F32TF32TF32F32_TN{},
                       Layout<Shape<_1,_2,_1>>{},
                       Tile<_16,_16,_8>{}));
      return;
    } else if constexpr (std::is_same_v<Element, cute::half_t>) {
      f(make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
                       Layout<Shape<_1,_2,_1>>{},
                       Tile<_16,_16,_16>{}));
      return;
    }
  }
  f(make_tiled_mma(F32FMA<Element, Element>{},
                   Layout<Shape<_8,_8,_1>>{}));
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
      make_stride(k / group_size, Stride<_0, _1>{}, n * k / group_size));

  // Define CTA tile sizes (static).
  auto bM = Int<16>{};
  auto bN = Int<128>{};
  auto bK = Int<64>{};
  auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M,BLK_N,BLK_K)

  // Define the smem layouts (static).
  auto swizzle_atom = composition(Swizzle<3,3,3>{},
                                  Layout<Shape <_8,Shape <_8, _8>>,
                                         Stride<_8,Stride<_1,_64>>>{});
  auto bP = Int<3>{}; // pipeline
  auto sA_layout = tile_to_shape(swizzle_atom, make_shape(bM, bK, bP));
  auto sB_layout = tile_to_shape(swizzle_atom, make_shape(bN, bK, bP));

  // Create tiled MMA.
  auto mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
                            Layout<Shape<_1,_2,_1>>{},
                            Tile<_16,_16,_16>{});

  TiledCopy copy_a = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, Element>{},
                                     Layout<Shape<_8,_8>,Stride<_8,_1>>{},
                                     Layout<Shape<_1,_8>>{});
  TiledCopy copy_b = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint32_t>, Quant>{},
                                     Layout<Shape<_8,_8>,Stride<_8,_1>>{},
                                     Layout<Shape<_1,Int<32/sizeof_bits<Quant>::value>>>{});

  Copy_Atom<SM75_U32x4_LDSM_N, Element> s2r_atom_a;

  auto* kernel = &qmm_impl<
      decltype(prob_shape), decltype(cta_tiler),
      Element, Quant,
      decltype(dA), decltype(sA_layout), decltype(copy_a), decltype(s2r_atom_a),
      decltype(dB), decltype(sB_layout), decltype(copy_b),
      decltype(S_layout), decltype(dC), decltype(mma)>;

  // Set L1 to be SMEM only
  int smem_size = int(sizeof(SharedStorage<Element, Quant, decltype(sA_layout), decltype(sB_layout)>));
  cudaFuncSetAttribute(kernel,
                      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  cudaFuncSetAttribute(kernel,
                      cudaFuncAttributePreferredSharedMemoryCarveout, 100);

  // Launch kernel.
  dim3 num_blocks(size(ceil_div(m, bM)), size(ceil_div(n, bN)), l);
  dim3 block_dims(size(mma));
  void* args[] = {
      &prob_shape, &cta_tiler,
      &A, &dA, &sA_layout, &copy_a, &s2r_atom_a,
      &B, &dB, &sB_layout, &copy_b,
      &S, &Z, &S_layout,
      &C, &dC, &mma};
  launch_kernel(reinterpret_cast<void*>(kernel), num_blocks, block_dims, smem_size, args);
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
