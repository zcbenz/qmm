#include <cublas_v2.h>
#include <cute/layout.hpp>
#include <cutlass/arch/mma.h>
#include <cutlass/matrix_shape.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/util/GPU_Clock.hpp>
#include <cutlass/util/reference/device/gemm.h>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/reference/device/tensor_compare.h>
#include <cutlass/util/mixed_dtype_utils.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace cutlass {

using uint3b_t = integer_subbyte<3, false>;
using uint5b_t = integer_subbyte<5, false>;

template <typename T, int N, FloatRoundStyle Round>
struct NumericArrayConverter<T, uint3b_t, N, Round> {
  static_assert(N % 8 == 0);

  using result_type = Array<T, N>;
  using source_type = Array<uint3b_t, N>;

  CUTLASS_HOST_DEVICE
  static result_type convert(const source_type& source) {
    result_type result;
    auto* s_base = reinterpret_cast<const uint8_t*>(&source);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 8; ++i) {
      auto* s = s_base + i * 3;
      result[i * 8] = T(s[0] & 0x07);
      result[i * 8 + 1] = T((s[0] & 0x38) >> 3);
      result[i * 8 + 2] = T((s[0] & 0xc0) >> 6) + T((s[1] & 0x01) << 2);
      result[i * 8 + 3] = T((s[1] & 0x0e) >> 1);
      result[i * 8 + 4] = T((s[1] & 0x70) >> 4);
      result[i * 8 + 5] = T((s[1] & 0x80) >> 7) + T((s[2] & 0x03) << 1);
      result[i * 8 + 6] = T((s[2] & 0x1c) >> 2);
      result[i * 8 + 7] = T((s[2] & 0xe0) >> 5);
    }
    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(const source_type& s) const {
    return convert(s);
  }
};

template <typename T, int N, FloatRoundStyle Round>
struct NumericArrayConverter<T, uint5b_t, N, Round> {
  static_assert(N % 8 == 0);

  using result_type = Array<T, N>;
  using source_type = Array<uint5b_t, N>;

  CUTLASS_HOST_DEVICE
  static result_type convert(const source_type& source) {
    result_type result;
    auto* s_base = reinterpret_cast<const uint8_t*>(&source);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 8; ++i) {
      auto* s = s_base + i * 5;
      result[i * 8] = T(s[0] & 0x1f);
      result[i * 8 + 1] = T((s[0] & 0xe0) >> 5) + T((s[1] & 0x03) << 3);
      result[i * 8 + 2] = T((s[1] & 0x7c) >> 2);
      result[i * 8 + 3] = T((s[1] & 0x80) >> 7) + T((s[2] & 0x0f) << 1);
      result[i * 8 + 4] = T((s[2] & 0xf0) >> 4) + T((s[3] & 0x01) << 4);
      result[i * 8 + 5] = T((s[3] & 0x3e) >> 1);
      result[i * 8 + 6] = T((s[3] & 0xc0) >> 6) + T((s[4] & 0x07) << 2);
      result[i * 8 + 7] = T((s[4] & 0xf8) >> 3);
    }
    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(const source_type& s) const {
    return convert(s);
  }
};

template <typename T, int N, FloatRoundStyle Round>
struct NumericArrayConverter<T, uint6b_t, N, Round> {
  static_assert(N % 4 == 0);

  using result_type = Array<T, N>;
  using source_type = Array<uint6b_t, N>;

  CUTLASS_HOST_DEVICE
  static result_type convert(const source_type& source) {
    result_type result;
    auto* s_base = reinterpret_cast<const uint8_t*>(&source);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 4; ++i) {
      auto* s = s_base + i * 3;
      result[i * 4] = T(s[0] & 0x3f);
      result[i * 4 + 1] = T((s[0] >> 6) & 0x03) + T((s[1] & 0x0f) << 2);
      result[i * 4 + 2] = T((s[1] >> 4) & 0x0f) + T((s[2] & 0x03) << 4);
      result[i * 4 + 3] = T((s[2] >> 2) & 0x3f);
    }
    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(const source_type& s) const {
    return convert(s);
  }
};

} // namespace cutlass

namespace cute {

// Required by tiled copy for 3/5/6-bit weights.
struct uint24_t {
  std::array<std::uint8_t, 3> bytes;
};
struct uint40_t {
  std::array<std::uint8_t, 5> bytes;
};
struct uint48_t {
  std::array<std::uint8_t, 6> bytes;
};

template <>
struct uint_bit<24> {
  using type = uint24_t;
};
template <>
struct uint_bit<40> {
  using type = uint40_t;
};
template <>
struct uint_bit<48> {
  using type = uint48_t;
};

} // namespace cute

namespace cute_gemm {

using namespace cute;

template <typename Quant>
constexpr bool has_bias_v = !cutlass::has_negative_zero_v<Quant>;

template <typename Element, typename SmemLayoutA, typename SmemLayoutB>
struct SharedStorage {
  ArrayEngine<Element, cosize_v<SmemLayoutA>> A;
  ArrayEngine<Element, cosize_v<SmemLayoutB>> B;
};

__device__ __forceinline__ void
cute_vectorized_dequant(auto w, auto s, auto z, auto out) {
  using Element = typename decltype(out)::value_type;
  using Quant = typename decltype(w)::value_type;
  // Scale must be one element.
  CUTE_STATIC_ASSERT_V(cosize(s.layout()) == Int<1>{});
  CUTE_STATIC_ASSERT_V(cosize(z.layout()) == Int<1>{});
  // Quant must be contiguous.
  auto layout = coalesce(w.layout());
  CUTE_STATIC_ASSERT_V(stride(layout) == Int<1>{});
  // Use cutlass for conversions.
  constexpr int N = size(layout);
  auto& w_vec = *(reinterpret_cast<const cutlass::Array<Quant, N>*>(raw_pointer_cast(w.data())));
  Element scale{s[0]};
  cutlass::NumericArrayConverter<Element, Quant, N> converter;
  auto w_dq = converter(w_vec) * scale;
  if constexpr (has_bias_v<Quant>) {
    Element zero_point{z[0]};
    w_dq = w_dq + zero_point;
  }
  copy(make_tensor(make_rmem_ptr<Element>(&w_dq), out.layout()), out);
}

__device__ __forceinline__ void
cute_naive_dequant(auto w, auto s, auto z, auto out) {
  using Element = typename decltype(out)::value_type;
  using Quant = typename decltype(w)::value_type;
  using Scale = typename decltype(s)::value_type;
  transform(w, out, [](Quant q) { return Element(q); } );
  transform(out, s, out, [](Element e, Scale s) { return e * Element(s); });
  if constexpr (has_bias_v<Quant>) {
    transform(out, z, out, plus{});
  }
}

__device__ __forceinline__ void
cute_dequant(auto w, auto s, auto z, auto out) {
  if constexpr (stride(coalesce(w.layout())) == Int<1>{} &&
                is_static_v<decltype(s.layout())>) {
    cute_vectorized_dequant(w, s, z, out);
  } else {
    cute_naive_dequant(w, s, z, out);
  }
}

template <bool HasKResidue, typename ProblemShape, typename CtaTiler,
          typename Element, typename Quant, typename Scale,
          typename StrideA, typename SmemLayoutA, typename TiledCopyA,
          typename StrideB, typename SmemLayoutB, typename TiledCopyB,
          typename StrideC, typename LayoutS, typename TiledMma>
__global__ void qmm_naive_kernel(
    ProblemShape shape_MNKL, CtaTiler cta_tiler,
    const Element* A, StrideA dA, SmemLayoutA sA_layout, TiledCopyA copy_a,
    const Quant*   B, StrideB dB, SmemLayoutB sB_layout, TiledCopyB copy_b,
          Element* C, StrideC dC,
    const Scale* S, const Element* Z, LayoutS S_layout,
    TiledMma mma) {
  CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));
  CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));
  CUTE_STATIC_ASSERT_V(congruent(select<0,2,3>(shape_MNKL), dA));
  CUTE_STATIC_ASSERT_V(congruent(select<1,2,3>(shape_MNKL), dB));
  CUTE_STATIC_ASSERT_V(congruent(select<0,1,3>(shape_MNKL), dC));

  int thread_idx = int(threadIdx.x);
  auto [m_coord, n_coord, l_coord] = static_cast<uint3>(blockIdx);

  auto m_max_coord = size<0>(shape_MNKL) - size<0>(cta_tiler) * m_coord; // M - BLK_M * m_coord
  auto n_max_coord = size<1>(shape_MNKL) - size<1>(cta_tiler) * n_coord; // N - BLK_N * n_coord

  // Shift tensor so we handle residue of K in the 0th tile.
  auto shape_K = size<2>(shape_MNKL);
  auto bK = size<2>(cta_tiler);
  auto k_residue = shape_K - bK * ceil_div(shape_K, bK);
  if constexpr (HasKResidue) {
    A += k_residue * get<1>(dA);
    B += k_residue * get<1>(dB) * cuda::std::min(8, sizeof_bits_v<Quant>) / 8;
    S += k_residue * stride<1>(S_layout);
    Z += k_residue * stride<1>(S_layout);
  }

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

  // Shared memory buffers.
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<Element, SmemLayoutA, SmemLayoutB>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout); // (BLK_M,BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout); // (BLK_N,BLK_K)

  // Partition the copying of A/B/C tiles across the threads.
  ThrCopy thr_copy_a = copy_a.get_slice(thread_idx);
  Tensor tAgA = thr_copy_a.partition_S(gA); // (ACPY,ACPY_M,ACPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA); // (ACPY,ACPY_M,ACPY_K)
  Tensor tArA = make_fragment_like(tAsA);   // (ACPY,ACPY_M,ACPY_K)

  ThrCopy thr_copy_b = copy_b.get_slice(thread_idx);
  Tensor tBgB = thr_copy_b.partition_S(gB);        // (BCPY,BCPY_N,BCPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(sB);        // (BCPY,BCPY_N,BCPY_K)
  Tensor tBrB = make_fragment_like<Quant>(tBsB);   // (BCPY,BCPY_M,BCPY_K)
  Tensor tBrB_dq = make_fragment_like(tBsB);       // (BCPY,BCPY_M,BCPY_K)
  Tensor tBgS = thr_copy_b.partition_S(gS);        // (BCPY,BCPY_N,BCPY_K,k)
  Tensor tBrS = make_fragment_like(tBgS(_,_,_,0)); // (BCPY,BCPY_N,BCPY_K)
  Tensor tBgZ = thr_copy_b.partition_S(gZ);        // (BCPY,BCPY_N,BCPY_K,k)
  Tensor tBrZ = make_fragment_like(tBgZ(_,_,_,0)); // (BCPY,BCPY_N,BCPY_K)

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
  Tensor cC = make_identity_tensor(make_shape(size<0>(gC), size<1>(gC))); // (M,N)
  Tensor tAcA = thr_copy_a.partition_S(cA); // (CPY,CPY_M,CPY_K)
  Tensor tBcB = thr_copy_b.partition_S(cB); // (CPY,CPY_N,CPY_K)
  Tensor tCcC = thr_mma.partition_C(cC);    // (MMA,MMA_M,MMA_N)
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
    copy_if(copy_a, tApA, tAgA(_,_,_,tile), tArA);
    copy_if(copy_b, tBpB, tBgB(_,_,_,tile), tBrB);
    copy(tBgS(_,_,_,tile), tBrS);
    copy(tBgZ(_,_,_,tile), tBrZ);
  };
  // RMEM => SMEM.
  auto store_smem = [&]() {
    __syncthreads();
    copy(tArA, tAsA);
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tBrB); ++k) {
      CUTE_UNROLL
      for (int n = 0; n < size<1>(tBrB); ++n) {
        cute_dequant(tBrB(_,n,k), tBrS(_,n,k), tBrZ(_,n,k), tBrB_dq(_,n,k));
      }
    }
    copy(tBrB_dq, tBsB);
    __syncthreads();
  };

  // Clear the rmem tiles to account for predicated off loads.
  if constexpr (HasKResidue) {
    clear(tArA);
    clear(tBrB);
    clear(tBrS);
    clear(tBrZ);
  }

  // Prefetch first tile.
  if constexpr (HasKResidue) {
    Tensor tAgA_k = tAgA(_,_,_,0);
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tArA); ++k) {
      if (get<1>(tAcA(0,0,k)) >= -k_residue) {
        copy_if(copy_a, tApA(_,k), tAgA_k(_,_,k), tArA(_,_,k));
      }
    }
    Tensor tBgB_k = tBgB(_,_,_,0);
    Tensor tBgS_k = tBgS(_,_,_,0);
    Tensor tBgZ_k = tBgZ(_,_,_,0);
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tBrB); ++k) {
      if (get<1>(tBcB(0,0,k)) >= -k_residue) {
        copy_if(copy_b, tBpB(_,k), tBgB_k(_,_,k), tBrB(_,_,k));
        copy(tBgS_k(_,_,k), tBrS(_,_,k));
        copy(tBgZ_k(_,_,k), tBrZ(_,_,k));
      }
    }
  } else {
    fetch_gmem(0);
  }

  // Clear accumulators.
  clear(tCrC);

  // Loop over CTA tiles.
  auto K_TILE_MAX = size<3>(tAgA);
  for (int tile = 0; tile < K_TILE_MAX; ++tile) {
    store_smem();
    if constexpr (HasKResidue) {
      // Avoid fetching full 0th-tile when there is residue.
      if (K_TILE_MAX > 1) {
        fetch_gmem((tile + 1 < K_TILE_MAX) ? tile + 1 : tile);
      }
    } else {
      fetch_gmem((tile + 1 < K_TILE_MAX) ? tile + 1 : tile);
    }
    gemm(mma, tCsA, tCsB, tCrC);
  }

  // Epilogue.
  CUTE_UNROLL
  for (int i = 0; i < size(tCrC); ++i) {
    if ((get<0>(tCcC(i)) < m_max_coord) && (get<1>(tCcC(i)) < n_max_coord)) {
      tCgC(i) = Element(tCrC(i));
    }
  }
}

template <bool KMajor>
inline constexpr auto make_matrix_stride(auto m, auto k) {
  if constexpr (KMajor) {
    return cute::make_stride(k, cute::Int<1>{}, m * k);
  } else {
    return cute::make_stride(cute::Int<1>{}, m, m * k);
  }
}

template <bool KMajor = true>
inline constexpr auto make_smem_layout(auto bM, auto bK) {
  // TODO: Calculate swizzle based on tile shape.
  if constexpr (KMajor) {
    auto swizzle = composition(Swizzle<3,3,3>{},
                               Layout<Shape <_8,Shape <_8, _8>>,
                                      Stride<_8,Stride<_1,_64>>>{});
    return tile_to_shape(swizzle, make_shape(bM, bK));
  } else {
    auto swizzle = composition(Swizzle<3,3,3>{},
                               Layout<Shape<_64,_1>, Stride<_1,_64>>{});
    return tile_to_shape(swizzle, make_shape(bM, bK));
  }
}

template <int TileM, bool SM80, typename Element>
inline constexpr auto make_tiled_mma() {
  using Atom = std::conditional_t<
      SM80,
      std::conditional_t<
          std::is_same_v<Element, half_t>,
          SM80_16x8x16_F32F16F16F32_TN,
          std::conditional_t<
              std::is_same_v<Element, bfloat16_t>,
              SM80_16x8x16_F32BF16BF16F32_TN,
              UniversalFMA<float>
          >
      >,
      UniversalFMA<float, Element, Element>>;
  if constexpr (!SM80 || std::is_same_v<Element, float>) {
    return make_tiled_mma(Atom{}, Layout<Shape<_16,_8,_1>>{});
  } else {
    if constexpr (TileM >= 32) {
      return make_tiled_mma(Atom{}, Layout<Shape<_2,_2,_1>>{}, Tile<_32,_32,_16>{});
    } else {
      return make_tiled_mma(Atom{}, Layout<Shape<_1,_4,_1>>{}, Tile<_16,_32,_16>{});
    }
  }
}

template <typename T, bool KMajor = true, bool HasKResidue = false>
inline auto make_tiled_copy(auto num_threads, auto bM, auto bK) {
  // TODO: Only do 1-element read for the tile of residue.
  auto n_read = Int<HasKResidue ? 1 : 8>{};
  auto atom = Copy_Atom<UniversalCopy<uint_bit_t<n_read * sizeof_bits_v<T>>>, T>{};
  if constexpr (KMajor) {
    auto k_threads = bK / n_read;
    return make_tiled_copy(
        atom,
        make_layout(make_shape(Int<num_threads / k_threads>{}, k_threads), LayoutRight{}),
        make_layout(make_shape(Int<1>{}, n_read)));
  } else {
    auto m_threads = bM / n_read;
    return make_tiled_copy(
        atom,
        make_layout(make_shape(m_threads, Int<num_threads / m_threads>{}), LayoutLeft{}),
        make_layout(make_shape(n_read, Int<1>{})));
  }
}

template <bool KMajor>
inline constexpr auto make_scales_layout(auto n, auto k, auto l, auto group_size) {
  if constexpr (KMajor) {
    return make_layout(
        make_shape(n, make_shape(group_size, k / group_size), l),
        make_stride(k / group_size, Stride<_0,_1>{}, n * k / group_size));
  } else {
    return make_layout(
        make_shape(make_shape(group_size, n / group_size), k, l),
        make_stride(Stride<_0,_1>{}, n / group_size, n * k / group_size));
  }
}

template <int TileM = 16, bool KMajor = true, bool SM80 = true, bool HasKResidue = false,
          typename Element, typename Quant, typename Scale>
void qmm_naive(
    const Element* A,
    const Quant*   B,
    const Scale*   S,
    const Element* Z,
    Element* C,
    int m, int n, int k, int l,
    bool broadcast_b,
    auto group_size,
    auto&& launch_kernel) {
  // Define shapes (dynamic).
  auto prob_shape = make_shape(m, n, k, l); // (M,N,K,L)

  // Define TN strides (mixed).
  auto dA = make_stride(k, Int<1>{}, m * k);  // (dM,dK,dL)
  auto dB = make_matrix_stride<KMajor>(n, k); // (dN,dK,dL)
  auto dC = make_stride(n, Int<1>{}, m * n);  // (dM,dN,dL)

  // Define layout of scales/biases (mixed).
  auto S_layout = make_scales_layout<KMajor>(n, k, l, group_size);

  // Handle broadcasting.
  if (broadcast_b) {
    get<2>(dB) = 0;
    get<2>(stride(S_layout)) = 0;
  }

  // Define CTA tile sizes (static).
  auto bM = Int<TileM>{};
  auto bN = Int<(!SM80 && group_size > 64) ? 64 : 128>{};
  auto bK = Int<max(64, group_size)>{};
  auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M,BLK_N,BLK_K)

  // Define MMA.
  TiledMMA mma = make_tiled_mma<TileM, SM80, Element>();
  auto num_threads = size(mma);

  // Define the A/B smem layouts (static).
  auto sA_layout = make_smem_layout(bM, bK);
  auto sB_layout = make_smem_layout<KMajor>(bN, bK);

  // Atoms.
  TiledCopy copy_a = make_tiled_copy<Element, true, HasKResidue>(num_threads, bM, bK);
  TiledCopy copy_b = make_tiled_copy<Quant, KMajor>(num_threads, bN, bK);

  auto* kernel = &qmm_naive_kernel<
      HasKResidue, decltype(prob_shape), decltype(cta_tiler),
      Element, Quant, Scale,
      decltype(dA), decltype(sA_layout), decltype(copy_a),
      decltype(dB), decltype(sB_layout), decltype(copy_b),
      decltype(dC), decltype(S_layout), decltype(mma)>;

  // Set L1 to be SMEM only.
  size_t smem_bytes = sizeof(SharedStorage<Element, decltype(sA_layout), decltype(sB_layout)>);
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

  dim3 num_blocks(size(ceil_div(m, bM)), size(ceil_div(n, bN)), l);
  dim3 block_dims(num_threads);
  void* args[] = {
      &prob_shape, &cta_tiler,
      &A, &dA, &sA_layout, &copy_a,
      &B, &dB, &sB_layout, &copy_b,
      &C, &dC,
      &S, &Z, &S_layout,
      &mma};
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

template <typename Element>
void cutlass_gemm_tt(int m, int n, int k, int l,
                     Element* A, Element* B, Element* C) {
  cutlass::reference::device::Gemm<Element,
                                   cutlass::layout::RowMajor,
                                   Element,
                                   cutlass::layout::RowMajor,
                                   Element,
                                   cutlass::layout::RowMajor,
                                   float,
                                   Element> gemm_ref;
  cutlass::TensorRef ref_A(A, cutlass::layout::RowMajor::packed({m, k}));
  cutlass::TensorRef ref_B(B, cutlass::layout::RowMajor::packed({k, n}));
  cutlass::TensorRef ref_C(C, cutlass::layout::RowMajor::packed({m, n}));
  gemm_ref({m, n, k},
           1.f,
           ref_A,
           ref_B,
           0.f,
           ref_C,
           ref_C);
  CUTE_CHECK_LAST();
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
  using Quant = cutlass::uint2b_t;
  constexpr int TileM = 16;
  constexpr bool SM80 = true;
  constexpr bool KMajor = false;
  constexpr bool HasKResidue = true;
  constexpr int group_size = 64;

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
  if constexpr (cute_gemm::has_bias_v<Quant>) {
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
      d_S.data().get(), d_S.size(), Element(0.f), Element(1.f));
  cutlass::reference::device::BlockFillSequential(
      d_Z.data().get(), d_Z.size(), Element(0.f), Element(0.f));
#endif

  using namespace cute;
  cudaStream_t stream = nullptr;
  auto make_weight_layout = [&]<bool KMajor>() {
    if constexpr (KMajor) {
      return make_layout(make_shape(n, k, l), make_stride(k, Int<1>{}, n * k));
    } else {
      return make_layout(make_shape(k, n, l), make_stride(n, Int<1>{}, n * k));
    }
  };
  auto make_scales_layout = [&]<bool KMajor>() {
    if constexpr (KMajor) {
      return make_layout(make_shape(n, k / group_size, l), make_stride(k / group_size, Int<1>{}, n * k / group_size));
    } else {
      return make_layout(make_shape(k, n / group_size, l), make_stride(n / group_size, Int<1>{}, n * k / group_size));
    }
  };
  cutlass::dequantize(
      d_B_dq.data().get(),
      d_B.data().get(),
      make_weight_layout.operator()<KMajor>(),
      d_S.data().get(),
      d_Z.data().get(),
      make_scales_layout.operator()<KMajor>(),
      group_size,
      stream);

  // Run once
  cute_gemm::qmm_naive<TileM, KMajor, SM80, HasKResidue>(
      d_A.data().get(),
      d_B.data().get(),
      d_S.data().get(),
      d_Z.data().get(),
      d_D.data().get(),
      m, n, k, l,
      false,
      Int<group_size>{},
      launch_kernel);
  CUTE_CHECK_LAST();

  // Verify
  cublas_gemm(
      'T', KMajor ? 'N' : 'T',
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
    cute_gemm::qmm_naive<TileM, KMajor, SM80, HasKResidue>(
        d_A.data().get(),
        d_B.data().get(),
        d_S.data().get(),
        d_Z.data().get(),
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
        'T', KMajor ? 'N' : 'T',
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
