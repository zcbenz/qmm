#include <cublas_v2.h>
#include <cute/layout.hpp>
#include <cute/numeric/numeric_types.hpp>
#include <cutlass/util/GPU_Clock.hpp>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/reference/device/tensor_compare.h>
#include <cutlass/util/mixed_dtype_utils.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <type_traits>

#include <cutlass/numeric_conversion.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#define WARP_SIZE 32

// Maps CPU types to CUDA types.
template <typename T>
struct CTypeToCudaType {
  using type = T;
};

template <>
struct CTypeToCudaType<cutlass::half_t> {
  using type = __half;
};

template <>
struct CTypeToCudaType<cutlass::bfloat16_t> {
  using type = __nv_bfloat16;
};

template <typename T>
using cuda_type_t = typename CTypeToCudaType<T>::type;

template <typename F>
void dispatch_bool(bool v, F&& f) {
  if (v) {
    f(std::true_type{});
  } else {
    f(std::false_type{});
  }
}

namespace cu {

namespace cg = cooperative_groups;

// Fused vectorized dequantize and multiply-add:
// w_dq = w * scale + bias
// out = fma(x, w_dq, out)
template <int N, typename T, typename Q>
__device__ __forceinline__ void dequant_fma(
    const T* x,
    const Q* w,
    T scale,
    T bias,
    float* out) {
  // Read x/w into registers.
  auto x_vec = *(reinterpret_cast<const cutlass::AlignedArray<T, N>*>(x));
  auto w_vec = *(reinterpret_cast<const cutlass::AlignedArray<Q, N>*>(w));
  // Output is assumed to be registers.
  auto* out_vec = reinterpret_cast<cutlass::Array<float, N>*>(out);

  // Dequantize w.
  cutlass::NumericArrayConverter<T, Q, N> converter_tq;
  cutlass::Array<T, N> w_dq = converter_tq(w_vec);
  w_dq = w_dq * scale + bias;

  // Promote x/w to float.
  static_assert(!cuda::std::is_same_v<T, float>);
  cutlass::NumericArrayConverter<float, T, N> converter_ft;
  cutlass::Array<float, N> x_f = converter_ft(x_vec);
  cutlass::Array<float, N> w_f = converter_ft(w_dq);

  // Multiply and add.
  *out_vec = cutlass::fma(x_f, w_f, *out_vec);
}

// Specialized for float which does not need promotions.
template <int N, typename Q>
__device__ __forceinline__ void dequant_fma(
    const float* x,
    const Q* w,
    float scale,
    float bias,
    float* out) {
  auto x_vec = *(reinterpret_cast<const cutlass::AlignedArray<float, N>*>(x));
  auto w_vec = *(reinterpret_cast<const cutlass::AlignedArray<Q, N>*>(w));
  auto* out_vec = reinterpret_cast<cutlass::Array<float, N>*>(out);

  cutlass::NumericArrayConverter<float, Q, N> converter;
  cutlass::Array<float, N> w_dq = converter(w_vec);
#pragma unroll
  for (int i = 0; i < N; ++i) {
    w_dq[i] = w_dq[i] * scale + bias;
  }

  *out_vec = cutlass::fma(x_vec, w_dq, *out_vec);
}

template <
    int rows_per_block,
    int elems_per_thread,
    int group_size,
    bool has_bias,
    bool has_residue_k,
    typename T,
    typename Q>
__global__ void qmv_kernel(
    const T* x,
    const Q* w,
    const T* scales,
    const T* biases,
    T* out,
    int n,
    int k) {
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  // How many groups (and scales/biases) in a row/block.
  int groups_per_row = k / group_size;
  int groups_per_block = rows_per_block * groups_per_row;

  // The start of row that this block handles.
  int row_head = block.group_index().x * rows_per_block;

  // The row that this warp handles.
  int row = row_head + warp.meta_group_rank(); // [0, n)
  if (row >= n) {
    return;
  }

  // For sub-byte Q, pointer moves by 8bits for each advance, e.g. w += 1 would
  // move past 2 elements for 4-bit Q.
  constexpr int w_step = 8 / cuda::std::min(8, cute::sizeof_bits_v<Q>);

  // Advance pointers of x/out.
  x += block.group_index().y * k;
  out += block.group_index().y * n;

  // Advance w/scales/biases to current row.
  w += static_cast<int64_t>(row) * k / w_step;
  scales += static_cast<int64_t>(row) * groups_per_row;
  if constexpr (has_bias) {
    biases += static_cast<int64_t>(row) * groups_per_row;
  }

  // Accumulations of current row.
  float sums[elems_per_thread] = {};

  auto dequant_fma_tile = [&](int idx) {
    T scale = scales[idx / group_size];
    T bias{0};
    if constexpr (has_bias) {
      bias = biases[idx / group_size];
    }
    dequant_fma<elems_per_thread>(x + idx, w + idx / w_step, scale, bias, sums);
  };

  // Loop over k dimension.
  constexpr int elems_per_warp = WARP_SIZE * elems_per_thread;
  for (int r = 0; r < k / elems_per_warp; ++r) {
    int idx = warp.thread_rank() * elems_per_thread + r * elems_per_warp;
    dequant_fma_tile(idx);
  }

  // Handle remaining elements in k dimension.
  if constexpr (has_residue_k) {
    int rest = k % elems_per_warp;
    int idx = warp.thread_rank() * elems_per_thread + k - rest;
    if (idx < k) {
      dequant_fma_tile(idx);
    }
  }

  // Result for current row.
  float sum{0};
#pragma unroll
  for (int i = 0; i < elems_per_thread; ++i) {
    sum += sums[i];
  }
  sum = cg::reduce(warp, sum, cg::plus<float>{});

  // Write result for current warp, which maps to rows 1-to-1.
  if (warp.thread_rank() == 0) {
    out[row] = static_cast<T>(sum);
  }
}

template <int group_size, bool has_bias, typename T, typename Q, typename F>
void qmv(
    const T* x,
    const Q* w,
    const T* scales,
    const T* biases,
    T* out,
    int m,
    int n,
    int k,
    F&& launch_kernel) {
  constexpr int rows_per_block = 8;
  constexpr int elems_per_thread = 8;

  void* args[] = { &x, &w, &scales, &biases, &out, &n, &k };

  dispatch_bool(k % (WARP_SIZE * elems_per_thread), [&](auto has_residue_k) {
    auto* kernel = &qmv_kernel<
        rows_per_block,
        elems_per_thread,
        group_size,
        has_bias,
        has_residue_k.value,
        cuda_type_t<T>,
        cuda_type_t<Q>>;
    dim3 num_blocks{cuda::ceil_div(n, rows_per_block), m};
    dim3 block_dims{WARP_SIZE, rows_per_block};
    launch_kernel(
        reinterpret_cast<void*>(kernel),
        num_blocks,
        block_dims,
        {},
        0,
        args);
  });
}

} // namespace cu

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
  cudaLaunchKernelExC(&config, func, args);
}

int main(int argc, char** argv) {
  int m = 1;
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
      d_S.data().get(), d_S.size(), Element(1.f), Element(0.f));
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
  cu::qmv<group_size, has_bias>(
      d_A.data().get(),
      d_B.data().get(),
      d_S.data().get(),
      d_Z.data().get(),
      d_D.data().get(),
      m, n, k,
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
    cu::qmv<group_size, has_bias>(
        d_A.data().get(),
        d_B.data().get(),
        d_S.data().get(),
        d_Z.data().get(),
        d_D.data().get(),
        m, n, k,
        launch_kernel);
  }
  double cute_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("QMV:     [%5.2f]TFlop/s  [%6.1f]GiB/s  (%6.4f)ms\n",
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
  printf("CUBLAS:  [%5.2f]TFlop/s  [%6.1f]GiB/s  (%6.4f)ms\n",
         tflops / cublas_time,
         gemm_gibs / cublas_time,
         cublas_time * 1000);

  printf("Speedup: [%5.2f]x\n", cublas_time / cute_time);
#endif

  return 0;
}
