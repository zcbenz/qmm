ARCH_FLAGS = -gencode arch=compute_$(shell __nvcc_device_query)a,code=sm_$(shell __nvcc_device_query)a
C_FLAGS = -w --expt-relaxed-constexpr -std=c++20 -lcublas
INCLUDE_FLAGS = -I cutlass/include -I cutlass/tools/util/include
FLAGS = $(INCLUDE_FLAGS) $(ARCH_FLAGS) $(CUDA_FLAGS) $(C_FLAGS)

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