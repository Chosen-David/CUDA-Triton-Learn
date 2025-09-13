#include "../include/utils.h"

//避免数值溢出
#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)


__global__ void sigmoid_f32_naive(float *x, float *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float v = x[idx];
    v = fminf(fmaxf(v, MIN_EXP_F32), MAX_EXP_F32);
    y[idx] = 1.0f / (1.0f + expf(-v));
  }
}
//  FP16
__global__ void sigmoid_f16_naive(half *x, half *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const half f = __float2half(1.0f);
  if (idx < N) {
    half v = x[idx];
    v = __hmin(__hmax(v, MIN_EXP_F16), MAX_EXP_F16);
    y[idx] = f / (f + hexp(-v));
  }
}

// Sigmoid x: N, y: N y=1/(1+exp(-x)) Vec4
// grid(N/256), block(256/4)
__global__ void sigmoid_f32_vec4(float *x, float *y, int N) {
    int idx=(threadIdx.x+blockIdx.x*blockDim.x)*4;
    float4 regx=FLOAT4(x[idx]);
    float4 regy;

    regx.x=fminf(fmax(regx.x,MIN_EXP_F32),MAX_EXP_F32);
    regx.y=fminf(fmax(regx.y,MIN_EXP_F32),MAX_EXP_F32);
    regx.z=fminf(fmax(regx.z,MIN_EXP_F32),MAX_EXP_F32);
    regx.w=fminf(fmax(regx.w,MIN_EXP_F32),MAX_EXP_F32);  
    
    regy.x=1.0f/(1+expf(-regx.x));
    regy.y=1.0f/(1+expf(-regx.y));
    regy.z=1.0f/(1+expf(-regx.z));
    regy.w=1.0f/(1+expf(-regx.w));

    if(idx<N){
        FLOAT4(y[idx])=regy;
    }
}

__global__ void sigmoid_f16_vec2(half *x, half *y, int N) {
    int idx=(threadIdx.x+blockIdx.x*blockDim.x)*2;
    half2 regx=HALF2(x[idx]);
    half2 regy;
    half2 one2=__half2half2(__float2half(1.0f));
    half2 MAX_EXP_F16_2=__half2half2(MAX_EXP_F16);
    half2 MIN_EXP_F16_2=__half2half2(MIN_EXP_F16);
    regx=__hmin2(__hmax2(regx,MIN_EXP_F16_2),MAX_EXP_F16_2);

    regy=one2/(one2+h2exp(-regx));
    if(idx<N){
        HALF2(y[idx])=regy;
    }
}

