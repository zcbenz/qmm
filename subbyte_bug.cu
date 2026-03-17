#include <cute/tensor.hpp>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace cute;

template <typename ProblemShape, typename CtaTiler, typename T,
          typename StrideX, typename SmemLayoutX, typename TiledCopyX>
__global__ void copy_to_smem_kernel(
    ProblemShape shape_MN, CtaTiler cta_tiler,
    const T* X, StrideX dX, SmemLayoutX sX_layout, TiledCopyX copy_x,
    float* Y) {
  int thread_idx = int(threadIdx.x);
  auto [m_coord, n_coord, l_coord] = static_cast<uint3>(blockIdx);

  Tensor mX = make_tensor(make_gmem_ptr(X), shape_MN, dX);    // (M, N)
  Tensor mY = make_tensor(make_gmem_ptr(Y), shape_MN, dX);    // (M, N)
  Tensor gX = local_tile(mX, cta_tiler, make_coord(m_coord)); // (BLK_M,BLK_N,n)
  Tensor gY = local_tile(mY, cta_tiler, make_coord(m_coord)); // (BLK_M,BLK_N,n)

  __shared__ float smemX[cosize_v<SmemLayoutX>];
  Tensor sX = make_tensor(make_smem_ptr(smemX), sX_layout); // (BLK_M,BLK_N)

  ThrCopy thr_copy_x = copy_x.get_slice(thread_idx);
  Tensor tXgX = thr_copy_x.partition_S(gX);  // (CPY,CPY_M,CPY_N,n)
  Tensor tXgY = thr_copy_x.partition_S(gY);  // (CPY,CPY_M,CPY_N,n)
  Tensor tXsX = thr_copy_x.partition_D(sX);  // (CPY,CPY_M,CPY_N)
  Tensor tXrX = make_fragment_like<T>(tXsX); // (CPY,CPY_M,CPY_N)
  Tensor tXrX_dq = make_fragment_like(tXsX); // (CPY,CPY_M,CPY_N)

  auto TILE_MAX  = size<3>(tXgX);
  for (int tile = 0; tile < TILE_MAX; ++tile) {
    copy(copy_x, tXgX(_,_,_,tile), tXrX);
    CUTE_UNROLL
    for (int i = 0; i < size(tXrX); ++i) {
      if constexpr (sizeof_bits_v<T> < 8) {
        tXrX_dq(i) = float(tXrX(i).get());
      } else {
        tXrX_dq(i) = float(tXrX(i));
      }
    }
    copy(tXrX_dq, tXsX);
    __syncthreads();
    copy(tXsX, tXgY(_,_,_,tile));
    __syncthreads();
  }
}

template <typename T>
void copy_to_smem(T* X, float* Y, int m, int n) {
  auto prob_shape = make_shape(m, n);
  auto dX = make_stride(n, Int<1>{});
  auto bM = Int<8>{};
  auto bN = Int<8>{};
  auto cta_tiler = make_shape(bM, bN);
  auto sX_layout = make_layout(make_shape(bM, bN), LayoutRight{});
  constexpr int bits = 2 * sizeof_bits_v<T>;
  TiledCopy copy_x = make_tiled_copy(Copy_Atom<UniversalCopy<uint_bit_t<bits>>, T>{},
                                     make_layout(Shape<_8,_4>{}, LayoutRight{}),
                                     make_layout(Shape<_1,_2>{}));
  dim3 num_blocks(size(ceil_div(m, bM)));
  dim3 block_dims(size(copy_x));
  copy_to_smem_kernel<<<num_blocks, block_dims>>>(
      prob_shape, cta_tiler, X, dX, sX_layout, copy_x, Y);
}

int main(int argc, char** argv) {
  int m = 8, n = 8;
  using T = cutlass::uint4b_t;

  constexpr int elems_per_byte = 8 / sizeof_bits_v<T>;
  thrust::device_vector<uint8_t> storage_X(m * n / elems_per_byte);
  T* X = reinterpret_cast<T*>(storage_X.data().get());
  cutlass::reference::device::BlockFillSequential(X, m * n, T(1), T(0));

  printf("X:\n");
  thrust::host_vector<uint8_t> h_X = storage_X;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n / elems_per_byte; ++j) {
      if constexpr (sizeof_bits_v<T> < 8) {
        uint8_t packed = h_X[i * n / elems_per_byte + j];
        printf("%2d %2d ", int(packed & 0x0F), int((packed >> 4) & 0x0F));
      } else {
        printf("%2d ", int(h_X[i * n + j]));
      }
    }
    printf("\n");
  }

  thrust::device_vector<float> storage_Y(m * n);
  float* Y = storage_Y.data().get();

  copy_to_smem(X, Y, m, n);
  CUTE_CHECK_LAST();

  printf("Y:\n");
  thrust::host_vector<float> h_Y = storage_Y;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      printf("%2.0f ", float(h_Y[i * n + j]));
    }
    printf("\n");
  }

  return 0;
}