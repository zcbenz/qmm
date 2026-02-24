#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cute/tensor.hpp>

#include <cublas_v2.h>
#include <cutlass/util/GPU_Clock.hpp>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/reference/device/tensor_compare.h>
#include <cutlass/util/mixed_dtype_utils.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

inline void check_cutlass_error(const char* name, cutlass::Status status) {
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr,
            "%s failed with code: %s.\n",
            name,
            cutlass::cutlassGetStatusString(status));
    exit(1);
  }
}
#define CHECK_CUTLASS_ERROR(cmd) check_cutlass_error(#cmd, (cmd))

namespace cutlass_gemm {

using namespace cute;

template <
    typename TileShape_ = Shape<_128, _16>,
    typename ClusterShape = Shape<_1, _1, _1>,
    typename Element,
    typename Quant,
    typename GroupSize,
    typename F>
void qmm_sm90(
    const Element* A,
    const Quant* B,
    const Element* S,
    const Element* Z,
    Element* D,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t l,
    GroupSize group_size,
    F&& launch_kernel) {
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  constexpr int kAlignmentA = 128 / sizeof_bits<Element>::value;
  constexpr int kAlignmentB = 128 / sizeof_bits<Quant>::value;
  constexpr int kTileShapeK =
      std::max(64, 128 * 8 / sizeof_bits<Element>::value);
  static_assert(group_size % kTileShapeK == 0);

  using Arch = cutlass::arch::Sm90;
  using Accumulator = float;
  using TileShape = decltype(append(TileShape_{}, Int<kTileShapeK>{}));

  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      Arch,
      cutlass::arch::OpClassTensorOp,
      TileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      Accumulator,
      Accumulator,
      // ElementC:
      void,
      cutlass::layout::ColumnMajor,
      kAlignmentA,
      // ElementD:
      Element,
      cutlass::layout::ColumnMajor,
      kAlignmentA,
      cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;

  // Note that A/B are swapped and transposed to use TMA epilogue.
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      Arch,
      cutlass::arch::OpClassTensorOp,
      // ElementA:
      std::conditional_t<
          // Only int quants have zero points.
          sizeof_bits_v<Quant> <= 8 && cutlass::has_negative_zero_v<Quant>,
          tuple<Quant, Element>,
          tuple<Quant, Element, Element>>,
      cutlass::layout::RowMajor,
      kAlignmentB,
      // ElementB:
      Element,
      cutlass::layout::ColumnMajor,
      kAlignmentA,
      Accumulator,
      TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::
      GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  auto dA = make_stride(k, Int<1>{}, m * k);
  auto dB = make_stride(k, Int<1>{}, n * k);
  auto dS = make_stride(Int<1>{}, n, n * k / group_size);
  auto dD = make_stride(Int<1>{}, n, m * n);

  Gemm gemm;
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {int(n), int(m), int(k), int(l)},
      {B, dB, A, dA, S, dS, group_size, Z},
      {{1.f, 0.f}, D, dD, D, dD}};

  CHECK_CUTLASS_ERROR(gemm.can_implement(args));
  CHECK_CUTLASS_ERROR(gemm.initialize(args, nullptr));

  auto* kernel = &cutlass::device_kernel<GemmKernel>;
  void* kernel_params[] = {const_cast<Gemm::Params*>(&gemm.params())};
  auto cluster = ClusterShape{};
  launch_kernel(
      reinterpret_cast<void*>(kernel),
      gemm.get_grid_shape(gemm.params()),
      GemmKernel::get_block_shape(),
      {get<0>(cluster),get<1>(cluster),get<2>(cluster)},
      GemmKernel::SharedStorageSize,
      kernel_params);
#else
  throw std::runtime_error(
      "[quantized_matmul] Hopper-only kernel is not available.");
#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
}

}

template <typename Element>
void cublas_gemm(char transA, char transB,
                 int m, int n, int k, int l,
                 const Element* A, const Element* B, Element* D) {
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
      D, dtype, m, m * n,
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
      D, dtype, n, m * n,
      l,
      compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  }
}

void launch_kernel(void* func, dim3 num_blocks, dim3 block_dims, dim3 cluster, size_t smem_bytes, void** args) {
  cudaLaunchConfig_t config = {};
  config.gridDim = num_blocks;
  config.blockDim = block_dims;
  config.dynamicSmemBytes = smem_bytes;
  config.stream = nullptr;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = cluster.x;
  attrs[0].val.clusterDim.y = cluster.y;
  attrs[0].val.clusterDim.z = cluster.z;
  config.attrs = attrs;
  config.numAttrs = 1;
  cudaLaunchKernelExC(&config, func, args);
}

int main(int argc, char** argv) {
  int m = 16;
  if (argc >= 2)
    sscanf(argv[1], "%d", &m);

  int n = 16384;
  if (argc >= 3)
    sscanf(argv[2], "%d", &n);

  int k = 16384;
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

  using Element = cute::half_t;
  using Quant = cute::uint4b_t;

  constexpr int group_size = 64;
  constexpr bool fp_quant = cutlass::sizeof_bits_v<Quant> <= 8 && cutlass::has_negative_zero_v<Quant>;

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
  if constexpr (fp_quant) {
    cutlass::reference::device::BlockFillSequential(
        d_Z.data().get(), d_Z.size(), Element(0.f), Element(0.f));
  } else {
    cutlass::reference::device::BlockFillRandomUniform(
        d_Z.data().get(), d_Z.size(), seed, Element(0.1f), Element(-0.1f));
  }
  cutlass::reference::device::BlockFillSequential(
      d_D.data().get(), d_D.size(), Element(-1.f), Element(0.f));
  cutlass::reference::device::BlockFillSequential(
      d_D_ref.data().get(), d_D_ref.size(), Element(-1.f), Element(0.f));

  using namespace cute;
  cudaStream_t stream = nullptr;
  cutlass::dequantize(
      d_B_dq.data().get(),
      d_B.data().get(),
      make_layout(make_shape(n, k, l), make_stride(k, Int<1>{}, n * k)),
      d_S.data().get(),
      d_Z.data().get(),
      make_layout(make_shape(n, k / group_size, l), make_stride(Int<1>{}, n, n * k / group_size)),
      group_size,
      stream);

  // Run once
  cutlass_gemm::qmm_sm90(
      d_A.data().get(),
      d_B.data().get(),
      d_S.data().get(),
      fp_quant ? nullptr : d_Z.data().get(),
      d_D.data().get(),
      m, n, k, l,
      cute::Int<group_size>{},
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
  // Timing iterations
  const int timing_iterations = 100;
  const double tflops = (2.0 * m * n * k) * 1e-12;
  GPU_Clock timer;
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    cutlass_gemm::qmm_sm90(
        d_A.data().get(),
        d_B.data().get(),
        d_S.data().get(),
        fp_quant ? nullptr : d_Z.data().get(),
        d_D.data().get(),
        m, n, k, l,
        cute::Int<group_size>{},
        launch_kernel);
  }
  double cute_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUTE:    [%5.1f]TFlop/s  (%6.4f)ms\n", tflops / cute_time, cute_time*1000);

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
  printf("CUBLAS:  [%5.1f]TFlop/s  (%6.4f)ms\n", tflops / cublas_time, cublas_time*1000);

  printf("Speedup: [%5.2f]x\n", cublas_time / cute_time);
#endif

  return 0;
}
