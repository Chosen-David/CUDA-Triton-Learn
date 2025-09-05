# Makefile (standalone, no PyTorch)
TARGET = test_sigmoid
NVCC   = nvcc
SM     = 86

NVCCFLAGS = -O3 -std=c++17 -gencode arch=compute_$(SM),code=sm_$(SM)
# 如需更快（牺牲一点精度）可打开：
# NVCCFLAGS += --use_fast_math

SRC = test_sigmoid.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -DBUILD_STANDALONE $< -o $@

clean:
	rm -f $(TARGET)

.PHONY: all clean
