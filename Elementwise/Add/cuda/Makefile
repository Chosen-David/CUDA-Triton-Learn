NVCC        := nvcc
NVCC_FLAGS  := -O3 -std=c++17 -lineinfo

# Change this to your GPU's SM if known; sm_86 used in tests
NVCC_FLAGS  += -arch=sm_86

TARGET      := add_bench
SRCS        := add_bench.cu add.cu

all: build

build: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

run_f16_vec8: $(TARGET)
	./$(TARGET) f16 vec8 16777216 10

run_f16_vec2: $(TARGET)
	./$(TARGET) f16 vec2 16777216 10

run_f32_vec4: $(TARGET)
	./$(TARGET) f32 vec4 16777216 10

clean:
	rm -f $(TARGET) *.o

