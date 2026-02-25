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

// Accumulate 8 Elements (half precision).
// Dequantizes 8 uint8_t values and accumulates the result with 8 half values
// from A into float sums: sums_f += A_h * dequant(B_quant).
__device__ __forceinline__ void accumulate_8_elements_8bit(
    uint64_t values_quant, // 8 packed uint8_t values
    half scale,
    half bias,
    const half* a,
    float* sums_f) {
  // Dequantization Setup.
  half2 scale_h2 = __half2half2(scale); // broadcast scale
  half2 bias_h2 = __half2half2(bias); // broadcast bias

  // Extract 8 uint8_t values.
  uint8_t q[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    q[i] = (values_quant >> (i * 8)) & 0xFF;
  }

  // Dequantize 8 values into 4 half2 vectors.
  // b_vec = q * scale + bias
  half2 q_01 = __halves2half2(__ushort2half_rn(q[0]), __ushort2half_rn(q[1]));
  half2 q_23 = __halves2half2(__ushort2half_rn(q[2]), __ushort2half_rn(q[3]));
  half2 q_45 = __halves2half2(__ushort2half_rn(q[4]), __ushort2half_rn(q[5]));
  half2 q_67 = __halves2half2(__ushort2half_rn(q[6]), __ushort2half_rn(q[7]));

  half2 mul_01 = __hmul2(q_01, scale_h2);
  half2 mul_23 = __hmul2(q_23, scale_h2);
  half2 mul_45 = __hmul2(q_45, scale_h2);
  half2 mul_67 = __hmul2(q_67, scale_h2);

  half2 b_vec0 = __hadd2(mul_01, bias_h2); // {b0, b1}
  half2 b_vec1 = __hadd2(mul_23, bias_h2); // {b2, b3}
  half2 b_vec2 = __hadd2(mul_45, bias_h2); // {b4, b5}
  half2 b_vec3 = __hadd2(mul_67, bias_h2); // {b6, b7}

  // Load Input A (8 half values as 4 half2 vectors).
  const half2* a_half2 = reinterpret_cast<const half2*>(a);
  half2 a_vec0 = a_half2[0];  // {a0, a1}
  half2 a_vec1 = a_half2[1];  // {a2, a3}
  half2 a_vec2 = a_half2[2];  // {a4, a5}
  half2 a_vec3 = a_half2[3];  // {a6, a7}

  // Convert half2 inputs to float2 for fmaf operations on sums_f.
  float2 a_vec0_f = __half22float2(a_vec0);
  float2 a_vec1_f = __half22float2(a_vec1);
  float2 a_vec2_f = __half22float2(a_vec2);
  float2 a_vec3_f = __half22float2(a_vec3);

  float2 b_vec0_f = __half22float2(b_vec0);
  float2 b_vec1_f = __half22float2(b_vec1);
  float2 b_vec2_f = __half22float2(b_vec2);
  float2 b_vec3_f = __half22float2(b_vec3);

  sums_f[0] = fmaf(a_vec0_f.x, b_vec0_f.x, sums_f[0]);
  sums_f[1] = fmaf(a_vec0_f.y, b_vec0_f.y, sums_f[1]);
  sums_f[2] = fmaf(a_vec1_f.x, b_vec1_f.x, sums_f[2]);
  sums_f[3] = fmaf(a_vec1_f.y, b_vec1_f.y, sums_f[3]);
  sums_f[4] = fmaf(a_vec2_f.x, b_vec2_f.x, sums_f[4]);
  sums_f[5] = fmaf(a_vec2_f.y, b_vec2_f.y, sums_f[5]);
  sums_f[6] = fmaf(a_vec3_f.x, b_vec3_f.x, sums_f[6]);
  sums_f[7] = fmaf(a_vec3_f.y, b_vec3_f.y, sums_f[7]);
}

// Accumulate 8 Elements (bfloat16 precision).
__device__ __forceinline__ void accumulate_8_elements_8bit(
    uint64_t values_quant, // 8 packed uint8_t values
    nv_bfloat16 scale,
    nv_bfloat16 bias,
    const nv_bfloat16* a,
    float* sums_f) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  float scale_f = __bfloat162float(scale);
  float bias_f = __bfloat162float(bias);

  float a_f[8];
  float b_dequant_f[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    a_f[i] = __bfloat162float(a[i]);
    uint8_t q_val = (values_quant >> (i * 8)) & 0xFF;
    b_dequant_f[i] = float(q_val) * scale_f + bias_f;
  }

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    sums_f[i] = fmaf(a_f[i], b_dequant_f[i], sums_f[i]);
  }
#endif
}

// Accumulate 8 Elements (float precision).
__device__ __forceinline__ void accumulate_8_elements_8bit(
    uint64_t values_quant, // 8 packed uint8_t values
    float scale,
    float bias,
    const float* a,
    float* sums_f) {
  // Load A using float4 for potentially better memory bandwidth.
  float4 a_vec_0 = *(reinterpret_cast<const float4*>(a));
  float4 a_vec_1 = *(reinterpret_cast<const float4*>(a + 4));

  // Extract, dequantize, and accumulate 8 float values.
  float v[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint8_t q_val = (values_quant >> (i * 8)) & 0xFF;
    v[i] = float(q_val) * scale + bias;
  }

  // Accumulate using fmaf (fused multiply-add).
  sums_f[0] = fmaf(v[0], a_vec_0.x, sums_f[0]);
  sums_f[1] = fmaf(v[1], a_vec_0.y, sums_f[1]);
  sums_f[2] = fmaf(v[2], a_vec_0.z, sums_f[2]);
  sums_f[3] = fmaf(v[3], a_vec_0.w, sums_f[3]);
  sums_f[4] = fmaf(v[4], a_vec_1.x, sums_f[4]);
  sums_f[5] = fmaf(v[5], a_vec_1.y, sums_f[5]);
  sums_f[6] = fmaf(v[6], a_vec_1.z, sums_f[6]);
  sums_f[7] = fmaf(v[7], a_vec_1.w, sums_f[7]);
}

template <
    int rows_per_block,
    int n_per_thread,
    int group_size,
    bool has_bias,
    typename T,
    typename Q>
__global__ void qmv_8bit_kernel(
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
  const int groups_per_row = cuda::ceil_div(k, group_size);
  const int groups_per_block = rows_per_block * groups_per_row;

  // The start of row that this block handles.
  int row_head = block.group_index().x * rows_per_block;

  // The row that this warp handles.
  int row = row_head + warp.meta_group_rank(); // [0, n)
  if (row >= n) {
    return;
  }

  // Scales and biases are loaded in shared memory.
  extern __shared__ char shared_buffer[];
  T* scales_smem = reinterpret_cast<T*>(shared_buffer);
  [[maybe_unused]] T* biases_smem = nullptr;
  if constexpr (has_bias) {
    biases_smem = scales_smem + groups_per_block;
  }

  // Load scales/biases used by current block.
  // TODO: Use vectorized read when groups_per_row % 2 == 0.
  constexpr int block_size = rows_per_block * WARP_SIZE;
  for (int i = block.thread_rank(); i < groups_per_block; i += block_size) {
    int t_row = i / groups_per_row;
    int t_col = i % groups_per_row;
    int t_n = row_head + t_row;
    if (t_n < n) {
      int64_t idx = static_cast<int64_t>(t_n) * groups_per_row + t_col;
      scales_smem[i] = scales[idx];
      if constexpr (has_bias) {
        biases_smem[i] = biases[idx];
      }
    }
  }
  block.sync();

  // Advance w/scales/biases to current row.
  w += static_cast<int64_t>(row) * k;
  scales_smem += warp.meta_group_rank() * groups_per_row;
  if constexpr (has_bias) {
    biases_smem += warp.meta_group_rank() * groups_per_row;
  }

  // Advance x/out to current row of x.
  int x_row = block.group_index().y; // [0, m)
  x += x_row * k;
  out += x_row * n;

  // Accumulations of current row.
  float sums[n_per_thread] = {};

  const int lane_offset = warp.thread_rank() * n_per_thread;

  constexpr int k_per_iter = WARP_SIZE * n_per_thread;  // Elements processed per warp per iteration (e.g., 32*8 = 256)
  int k_id = 0; // Current position along the K dimension

  for (; k_id + k_per_iter <= k; k_id += k_per_iter) {
    const Q* current_b_ptr = w + lane_offset + k_id;
    uint64_t value = *reinterpret_cast<const uint64_t*>(current_b_ptr);

    int current_meta_k = (lane_offset + k_id) / group_size;
    T scale = scales_smem[current_meta_k];
    T zp{0};
    if constexpr (has_bias) {
      zp = biases_smem[current_meta_k];
    }

    accumulate_8_elements_8bit(value, scale, zp, x + lane_offset + k_id, sums);
  }

#if 0
  // Handle the tail elements along K dimension for this thread.
  // This loop handles the final iteration if k is not a multiple of k_per_iter.
  // Since K % n_per_thread == 0 is enforced, each thread
  // processes a full set of n_per_thread if it has work left.
  if (lane_offset + k_id < k) {  // Check if this thread has remaining elements
    const uint8_t* current_b_ptr = w + lane_offset + k_id;
    uint64_t value = *reinterpret_cast<const uint64_t*>(current_b_ptr);

    // Calculate k_block index for the tail part
    int current_meta_k = (lane_offset + k_id) / group_size;
    T scale = b_scale_vec_thread[current_meta_k];
    T zp = 0;
    if constexpr (has_bias) {
      zp = b_zp_vec_thread[current_meta_k];
    }
    // Pointer to A data for the tail part
    const T* current_a_ptr = x + lane_offset + k_id;
    // Perform dequantization and accumulation
    accumulate_8_elements_8bit(value, scale, zp, current_a_ptr, sums);
  }
#endif

  // Result for current warp.
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < n_per_thread; ++i) {
    sum += sums[i];
  }
  sum = cg::reduce(warp, sum, cg::plus<float>{});

  if (warp.thread_rank() == 0) {
    out[row] = static_cast<T>(sum);
  }
}

template <int group_size, typename T, typename Q, typename F>
void qmv_8bit(
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
  constexpr int n_per_thread = 8;

  const int groups_per_row = cuda::ceil_div(k, group_size);

  dim3 num_blocks(cuda::ceil_div(n, rows_per_block), m);
  dim3 block_dims(WARP_SIZE, rows_per_block);

  size_t smem_bytes = (biases ? 2 : 1) * sizeof(T) * groups_per_row * rows_per_block;

  void* args[] = { &x, &w, &scales, &biases, &out, &n, &k };
  dispatch_bool(biases, [&](auto has_bias) {
    auto* kernel = &qmv_8bit_kernel<
        rows_per_block,
        n_per_thread, 
        group_size,
        has_bias.value,
        cuda_type_t<T>,
        cuda_type_t<Q>>;
    launch_kernel(
        reinterpret_cast<void*>(kernel),
        num_blocks,
        block_dims,
        {},
        smem_bytes,
        args);
  });
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
  cudaLaunchKernelExC(&config, func, args);
}

int main(int argc, char** argv) {
  int m = 1;
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

  using Element = float;
  using Quant = uint8_t;

  constexpr int group_size = 16;
  constexpr bool fp_quant = cute::sizeof_bits_v<Quant> <= 8 && cutlass::has_negative_zero_v<Quant>;

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
  cu::qmv_8bit<group_size>(
      d_A.data().get(),
      d_B.data().get(),
      d_S.data().get(),
      fp_quant ? nullptr : d_Z.data().get(),
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
  // Timing iterations
  const int timing_iterations = 100;
  const double tflops = (2.0 * m * n * k) * 1e-12;
  GPU_Clock timer;
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    cu::qmv_8bit<group_size>(
        d_A.data().get(),
        d_B.data().get(),
        d_S.data().get(),
        fp_quant ? nullptr : d_Z.data().get(),
        d_D.data().get(),
        m, n, k,
        launch_kernel);
  }
  double cute_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUTE:    [%5.2f]TFlop/s  (%6.4f)ms\n", tflops / cute_time, cute_time*1000);

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
  printf("CUBLAS:  [%5.2f]TFlop/s  (%6.4f)ms\n", tflops / cublas_time, cublas_time*1000);

  printf("Speedup: [%5.2f]x\n", cublas_time / cute_time);
#endif

  return 0;
}
