# ===============================
# Makefile for add_vec_cute.cu
# ===============================

NVCC        := nvcc
NVCC_FLAGS  := -O3 -std=c++17 -arch=sm_86

LIBS        := -lcublas

TARGET      := test_add
SRC         := test_add.cu

# 默认目标
all: build

build: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS)  -o $@ $^ $(LIBS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o

