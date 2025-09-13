//学习参考：https://github.com/xlite-dev/LeetCUDA/blob/main/kernels/elementwise/elementwise.cu
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>


#define FLOAT4(v) *reinterpret_cast<float4 *>(&v)//也可以用[0]解引用
#define HALF2(v) *reinterpret_cast<half2 *>(&v)
#define HALF8(v) *reinterpret_cast<float4 *>(&v)

// ElementWise Add + Vec4
// grid(N/256), block(256/4)
// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void add_vec4(float* a,float* b,float* c,int N){
    int idx=4*(threadIdx.x+blockDim.x*blockIdx.x);
    if(idx<N){
        float4 reg_a=FLOAT4(a[idx]);
        float4 reg_b=FLOAT4(b[idx]);
        float4 reg_c;
        reg_c.x=reg_a.x+reg_b.x;
        reg_c.y=reg_a.y+reg_b.y;
        reg_c.z=reg_a.z+reg_b.z;
        reg_c.w=reg_a.w+reg_b.w;
        FLOAT4(c[idx])=reg_c;
    }
}

// FP16
// ElementWise Add grid(N/256),
// block(256) a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void add_vec2_f16(half* a,half* b,half* c,int N){
    int idx=2*(threadIdx.x+blockDim.x*blockIdx.x);
    if(idx<N){
        half2 reg_a=HALF2(a[idx]);
        half2 reg_b=HALF2(b[idx]);
        half2 reg_c=__hadd2(reg_a,reg_b);//源代码是分别__hadd
        HALF2(c[idx])=reg_c;
    }
}

//可以进一步向量化访存，float4 = 16B = half8 (注意没有half8 但是你有float4)
__global__ void add_vec8_f16(half* a,half* b,half* c,int N){
    int idx=8*(threadIdx.x+blockDim.x*blockIdx.x);
    half pack_a[8],pack_b[8],pack_c[8];
    HALF8(pack_a[0])=HALF8(a[idx]);//注意本质是取pack_a[0]的地址然后连续放8个half
    HALF8(pack_b[0])=HALF8(b[idx]);//注意HALF8是给地址用的
    for(int i=0;i<8;i+=2){
        HALF2(pack_c[i])=__hadd2(HALF2(pack_a[i]),HALF2(pack_b[i]));//注意HALF2可以重新解释为half2的值
    }
    if((idx+7)<N){
        HALF8(c)=HALF8(pack_c[idx]);
    }


}


