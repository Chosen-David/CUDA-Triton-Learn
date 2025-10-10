NVCC      ?= nvcc
SM        ?= 86

NVCCFLAGS := -O3 -std=c++17 -gencode arch=compute_$(SM),code=sm_$(SM)
INCLUDES  := -I../benchmark -I../../../cuda_hgemv/src/common -I../../../FastGEMV

TESTS     := test_gemv test_multi_gemv
TEST      ?= test_gemv
KERNELS   := ../benchmark/thread_smem.cu \
             ../benchmark/warp1_smem.cu \
             ../benchmark/warp2_smem.cu \
             ../benchmark/warp4_smem.cu \
             ../benchmark/warp8_smem.cu \
             ../benchmark/warp16_smem.cu

all: run

$(TESTS): %: %.cu $(KERNELS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $^ -o $@

run: $(TEST)
	./$(TEST)

run_%: %
	./$*

clean:
	rm -f $(TESTS)

.PHONY: all run clean run_%
