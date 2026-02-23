CUDA_COMPUTE_CAPABILITY = $(shell __nvcc_device_query)$(shell [ $(shell __nvcc_device_query) -ge 90 ] && echo "a" || true)
CUDA_FLAGS = -w --expt-relaxed-constexpr -std=c++20 -lcublas -gencode arch=compute_$(CUDA_COMPUTE_CAPABILITY),code=sm_$(CUDA_COMPUTE_CAPABILITY)
CUTLASS_FLAGS = -DNDEBUG -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing
INCLUDE_FLAGS = -I cutlass/include -I cutlass/tools/util/include -I cutlass/tools/profiler/include
FLAGS = $(CUDA_FLAGS) $(CUTLASS_FLAGS) $(INCLUDE_FLAGS)

.PHONY: all
all: build/gemm build/qmm_sm90

build/qmm: qmm.cu
	mkdir -p build
	nvcc $< $(FLAGS) -o $@

build/qmm_sm90: qmm_sm90.cu
	mkdir -p build
	nvcc $< $(FLAGS) -o $@

build/gemm: gemm.cu
	mkdir -p build
	nvcc $< $(FLAGS) -o $@

build/cutlass_gemm: cutlass_gemm.cu
	mkdir -p build
	nvcc $< $(FLAGS) -o $@

.PHONY: clean
clean:
	rm -r build