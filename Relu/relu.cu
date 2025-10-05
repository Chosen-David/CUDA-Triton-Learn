#include "../include/utils.h"

// FP32
// Relu x: N, y: N y=max(0,x)
// grid(N/256), block(K=256)
__global__ void relu_f32(float *x, float *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    y[idx] = fmaxf(0.0f, x[idx]);
}
