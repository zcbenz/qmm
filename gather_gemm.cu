#include <cublas_v2.h>
#include <cute/layout.hpp>
#include <cutlass/util/GPU_Clock.hpp>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/reference/device/tensor_compare.h>
#include <cutlass/util/mixed_dtype_utils.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "cutlass/test/unit/gemm/device/default_gemm_configuration.hpp"

inline void check_cutlass_error(const char* name, cutlass::Status status) {
  if (status != cutlass::Status::kSuccess) {
    printf("%s failed with code: %s.", name, cutlass::cutlassGetStatusString(status));
  }
}
#define CHECK_CUTLASS_ERROR(cmd) check_cutlass_error(#cmd, (cmd))

namespace cutlass_gemm {

using namespace cute;

// Modified from cutlass/include/cutlass/gemm/kernel/sm70_gemm.hpp to fuse
// gather into GEMM.
template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_>
class GatherGemm {
 public:
  using ProblemShape = ProblemShape_;
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;

  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;

  static constexpr int SharedStorageSize = static_cast<int>(cute::max(
      sizeof(typename CollectiveMainloop::SharedStorage),
      sizeof(typename CollectiveEpilogue::SharedStorage)));
  static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMma{}));
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  struct Arguments {
    ProblemShape problem_shape;
    typename CollectiveMainloop::Arguments mainloop;
    typename CollectiveEpilogue::Arguments epilogue;
  };

  struct Params {
    ProblemShape problem_shape;
    typename CollectiveMainloop::Params mainloop;
    typename CollectiveEpilogue::Params epilogue;
  };

  static Params to_underlying_arguments(const Arguments& args, void* workspace) {
    return {
      args.problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, workspace),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, workspace)};
  }

  static cutlass::Status initialize_workspace(const Arguments&, void*, cudaStream_t, void*) {
    return cutlass::Status::kSuccess;
  }

  static dim3 get_grid_shape(const Params& params) {
    return dim3{
      size(ceil_div(shape<0>(params.problem_shape), shape<0>(TileShape{}))),
      size(ceil_div(shape<1>(params.problem_shape), shape<1>(TileShape{}))),
      size<3>(params.problem_shape)};
  }

  static dim3 get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE void operator()(const Params& params, char* smem_buf) {
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto [M,N,K,L] = problem_shape_MNKL;

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    int thread_idx = int(threadIdx.x);
    auto blk_shape = TileShape{};
    auto [m_coord, n_coord, l_coord] = static_cast<uint3>(blockIdx);
    auto blk_coord_mnkl = make_coord(int(m_coord), int(n_coord), _, int(l_coord));

    // Represent the full tensors.
    Tensor mA_mkl = make_tensor(make_gmem_ptr(params.mainloop.ptr_A), make_shape(M,K,L), params.mainloop.dA);
    Tensor mB_nkl = make_tensor(make_gmem_ptr(params.mainloop.ptr_B), make_shape(N,K,L), params.mainloop.dB);

    // Get batch slice.
    Tensor mA_mk = mA_mkl(_,_,l_coord);
    Tensor mB_nk = mB_nkl(_,_,l_coord);

    // Slice to get the tiles this thread block is responsible for.
    Tensor gA = local_tile(mA_mk, blk_shape, take<0,3>(blk_coord_mnkl), Step<_1, X,_1>{});
    Tensor gB = local_tile(mB_nk, blk_shape, take<0,3>(blk_coord_mnkl), Step< X,_1,_1>{});

    // Compute tile residues for predication.
    auto m_max_coord = M - size<0>(gA) * get<0>(blk_coord_mnkl);
    auto n_max_coord = N - size<0>(gB) * get<1>(blk_coord_mnkl);
    auto k_residue   = K - size<1>(gA) * size<2>(gA);
    auto residue_mnk = make_tuple(m_max_coord, n_max_coord, k_residue);

    // Allocate the tiled_mma and the accumulators for the (M,N) blk_shape.
    TiledMma tiled_mma;
    Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));
    clear(accumulators);

    auto k_tile_iter  = make_coord_iterator(shape<2>(gA));
    int  k_tile_count = size<2>(gA);

    // Perform the collective scoped MMA.
    CollectiveMainloop collective_mma;
    collective_mma(
        accumulators,
        gA,
        gB,
        accumulators,
        k_tile_iter, k_tile_count,
        residue_mnk,
        thread_idx,
        smem_buf);

    // Epilogue and write to out.
    CollectiveEpilogue epilogue(params.epilogue);
    epilogue(
        problem_shape_MNKL,
        blk_shape,
        blk_coord_mnkl,
        accumulators,
        tiled_mma,
        residue_mnk,
        thread_idx,
        smem_buf);
  }
};

template <typename Element, bool KMajor>
struct SimtCopyTraits {};

template <typename Element>
struct SimtCopyTraits<Element, true> {
  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<Element>, Element>{},
                    Layout<Shape<_32,_8>, Stride<_8,_1>>{},
                    Layout<Shape<_1,_1>>{}));
  using SmemLayout = Layout<Shape<_128,_8>, Stride<_1,Int<128 + 4>>>;
  using SmemCopyAtom = Copy_Atom<DefaultCopy, Element>;
};

template <typename Element>
struct SimtCopyTraits<Element, false> {
  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<Element>, Element>{},
                    Layout<Shape<_32,_8>, Stride<_1,_32>>{},
                    Layout<Shape<_1,_1>>{}));
  using SmemLayout = Layout<Shape<_128,_8>, Stride<_1,_128>>;
  using SmemCopyAtom = Copy_Atom<DefaultCopy, Element>;
};

template <typename F>
void dispatch_stride(bool k_major, int m, int k, F&& f) {
  if (k_major) {
    f(make_stride(k, Int<1>{}, m * k), std::true_type{});
  } else {
    f(make_stride(Int<1>{}, m, m * k), std::false_type{});
  }
}

template <typename Element>
void gather_gemm(
    int m, int n, int k, int l,
    bool a_transposed,
    bool b_transposed,
    const Element* A,
    const Element* B,
    Element* C) {
  using TileShape = Shape<_128,_128,_8>;
  using DispatchPolicy = cutlass::gemm::MainloopSm70TwoStage;
  using TiledMma = TiledMMA<
      MMA_Atom<UniversalFMA<float, Element, Element, Element>>,
      Layout<Shape<_16, _16, _1>>>;

  auto problem_shape = make_shape(m, n, k, l);
  auto dC = make_stride(m, Int<1>{}, m * n);

  dispatch_stride(!a_transposed, m, k, [&](auto dA, auto k_major_a) {
    dispatch_stride(b_transposed, n, k, [&](auto dB, auto k_major_b) {
      using CopyTraitsA = SimtCopyTraits<Element, k_major_a.value>;
      using CopyTraitsB = SimtCopyTraits<Element, k_major_b.value>;

      using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
          DispatchPolicy, TileShape,
          Element, decltype(dA),
          Element, decltype(dB),
          TiledMma,
          typename CopyTraitsA::GmemTiledCopy,
          typename CopyTraitsA::SmemLayout,
          typename CopyTraitsA::SmemCopyAtom,
          identity,
          typename CopyTraitsB::GmemTiledCopy,
          typename CopyTraitsB::SmemLayout,
          typename CopyTraitsB::SmemCopyAtom,
          identity>;

      using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
          Element,
          decltype(dC),
          decltype(dC),
          cutlass::epilogue::thread::LinearCombination<Element, 1, float, float>,
          cutlass::gemm::EpilogueDefault>;

      using GemmKernel = GatherGemm<
          decltype(problem_shape),
          CollectiveMainloop,
          CollectiveEpilogue>;
      using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

      Gemm gemm;
      typename Gemm::Arguments args{
          problem_shape,
          {A, dA, B, dB},
          {{1.f, 0.f}, C, dC, C, dC}};

      CHECK_CUTLASS_ERROR(gemm.initialize(args, nullptr));
      CHECK_CUTLASS_ERROR(gemm.run(args));
    });
  });
}

template <typename Element>
void gemm(
    char transA, char transB,
    int m, int n, int k, int l,
    const Element* A, const Element* B, Element* C) {
  return gather_gemm(
      m, n, k, l,
      transA != 'T',
      transB != 'T',
      A,
      B,
      C);
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
  cublasGemmStridedBatchedEx(h,
    transB == 'N' ? CUBLAS_OP_T : CUBLAS_OP_N,
    transA == 'N' ? CUBLAS_OP_T : CUBLAS_OP_N,
    n, m, k,
    p_alpha,
    B, dtype, transB == 'N' ? k : n, n * k,
    A, dtype, transA == 'T' ? k : m, m * k,
    p_beta,
    C, dtype, n, m * n,
    l,
    compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
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

  std::cout << "M = " << m << std::endl;
  std::cout << "N = " << n << std::endl;
  std::cout << "K = " << k << std::endl;
  std::cout << "L = " << l << std::endl;
  std::cout << "C = A^" << transA << " @ B^" << transB << std::endl;

  CUTE_CHECK_ERROR(cudaSetDevice(0));
  cudaDeviceProp device_prop;
  CUTE_CHECK_ERROR(cudaGetDeviceProperties(&device_prop, 0));

  using Element = float;

  thrust::device_vector<Element> d_A(m*k*l);
  thrust::device_vector<Element> d_B(n*k*l);
  thrust::device_vector<Element> d_C(m*n*l);
  thrust::device_vector<Element> d_C_ref(m*n*l);

  int seed = 42;
  cutlass::reference::device::BlockFillRandomUniform(
      d_A.data().get(), d_A.size(), seed, Element(0.1f), Element(-0.1f));
  cutlass::reference::device::BlockFillRandomUniform(
      d_B.data().get(), d_B.size(), seed, Element(0.1f), Element(-0.1f));
  cutlass::reference::device::BlockFillSequential(
      d_C.data().get(), d_C.size(), Element(-1.f), Element(0.f));
  cutlass::reference::device::BlockFillSequential(
      d_C_ref.data().get(), d_C_ref.size(), Element(-1.f), Element(0.f));

  // Run once
  cutlass_gemm::gemm(
      transA, transB,
      m, n, k, l,
      d_A.data().get(),
      d_B.data().get(),
      d_C.data().get());
  CUTE_CHECK_LAST();

  // Verify
  cublas_gemm(transA, transB,
              m, n, k, l,
              d_A.data().get(),
              d_B.data().get(),
              d_C_ref.data().get());
  Element epsilon{1e-2f};
  Element non_zero_floor{1e-4f};
  bool passed = cutlass::reference::device::BlockCompareRelativelyEqual(d_C_ref.data().get(), d_C.data().get(), d_C.size(), epsilon, non_zero_floor);
  if (!passed) {
    thrust::host_vector<Element> cute_result = d_C;
    thrust::host_vector<Element> cublas_result = d_C_ref;
    for (size_t i = 0; i < cute_result.size(); ++i) {
      float delta = fabs(float(cute_result[i]) - float(cublas_result[i]));
      if (delta > 3e-1) {
        printf("!!Wrong result found at %d: %f : %f\n", int(i), float(cute_result[i]), float(cublas_result[i]));
        exit(1);
      }
    }
  }

#if 1
  // Timing iterations
  const int timing_iterations = 100;
  const double tflops = (2.0 * m * n * k) * 1e-12;
  GPU_Clock timer;
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    cutlass_gemm::gemm(
        transA, transB,
        m, n, k, l,
        d_A.data().get(),
        d_B.data().get(),
        d_C.data().get());
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
