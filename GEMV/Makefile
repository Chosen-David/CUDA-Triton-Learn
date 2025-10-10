# ====== User knobs ======
# 默认架构：sm_86（可通过 make SM=90 等覆盖）
SM           ?= 86
CUDA_PATH    ?= /usr/local/cuda
NVCC         ?= $(CUDA_PATH)/bin/nvcc

# 构建类型：release / debug
BUILD        ?= release

# ====== Paths ======
ROOT_DIR     := $(abspath .)
BENCH_DIR    := $(ROOT_DIR)/benchmark
TEST_DIR     := $(ROOT_DIR)/test
INC_DIRS     := -I$(ROOT_DIR) -I$(BENCH_DIR) -I$(ROOT_DIR)/include

BUILD_DIR    := $(ROOT_DIR)/build
BIN          := $(BUILD_DIR)/test_gemv

# ====== Sources ======
BENCH_SRCS   := $(BENCH_DIR)/fast_gemv.cu \
                $(BENCH_DIR)/thread_smem.cu \
                $(BENCH_DIR)/warp1_smem.cu \
                $(BENCH_DIR)/warp2_smem.cu \
                $(BENCH_DIR)/warp4_smem.cu \
                $(BENCH_DIR)/warp8_smem.cu \
                $(BENCH_DIR)/warp16_smem.cu

CORE_SRCS    := $(ROOT_DIR)/src/smem_log.cu

TEST_SRCS    := $(TEST_DIR)/test_gemv.cu
SRCS         := $(BENCH_SRCS) $(TEST_SRCS) $(CORE_SRCS)
OBJS         := $(patsubst $(ROOT_DIR)/%.cu,$(BUILD_DIR)/%.o,$(SRCS))

# ====== Flags ======
COMMON_CUDA  := -std=c++17 -arch=sm_$(SM) $(INC_DIRS) -Xcompiler -fPIC
ifeq ($(BUILD),release)
  NVCCFLAGS := -O3 -lineinfo $(COMMON_CUDA)
else
  NVCCFLAGS := -O0 -g -G $(COMMON_CUDA)
endif

LDFLAGS      :=
LDLIBS       := -lcudart

DIRS_NEEDED  := $(sort $(dir $(OBJS)))
$(shell mkdir -p $(DIRS_NEEDED))

# ====== Rules ======
.PHONY: all run clean info

all: $(BIN)

$(BIN): $(OBJS)
	@echo "[LINK] $@"
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILD_DIR)/%.o: $(ROOT_DIR)/%.cu
	@mkdir -p $(dir $@)
	@echo "[NVCC] $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

run: $(BIN)
	@echo "========== RUN =========="
	@$(BIN)

clean:
	@echo "[CLEAN] $(BUILD_DIR)"
	@rm -rf $(BUILD_DIR)

info:
	@echo "NVCC   = $(NVCC)"
	@echo "SM     = $(SM)"
	@echo "BUILD  = $(BUILD)"
	@echo "BIN    = $(BIN)"
