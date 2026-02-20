CUDA_COMPUTE_CAPABILITY = $(shell __nvcc_device_query)$(shell [ $(shell __nvcc_device_query) -ge 90 ] && echo "a" || true)
CUDA_FLAGS = -w --expt-relaxed-constexpr -std=c++20 -lcublas -gencode arch=compute_$(CUDA_COMPUTE_CAPABILITY),code=sm_$(CUDA_COMPUTE_CAPABILITY)
INCLUDE_FLAGS = -I cutlass/include -I cutlass/tools/util/include -I cutlass/tools/profiler/include
FLAGS = $(CUDA_FLAGS) $(INCLUDE_FLAGS)

.PHONY: all
all: build/qmm build/gemm

build/qmm: qmm.cu
	mkdir -p build
	nvcc $< $(FLAGS) -o $@

build/gemm: gemm.cu
	mkdir -p build
	nvcc $< $(FLAGS) -o $@

.PHONY: clean
clean:
	rm -r build