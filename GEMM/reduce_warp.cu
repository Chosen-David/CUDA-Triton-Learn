#include "error.cuh"
#include <stdio.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 100;
const int N = 100000000;
const int M = sizeof(real) * N;
const int BLOCK_SIZE = 128;
const unsigned FULL_MASK = 0xffffffff;


void timing(const real *d_x, const int method);

int main(void)
{
    real *h_x = (real *) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        h_x[n] = 1.23;
    }
    real *d_x;
    CHECK(cudaMalloc(&d_x, M));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

    printf("\nusing syncwarp:\n");
    timing(d_x, 0);
    printf("\nusing shfl:\n");
    timing(d_x, 1);
    printf("\nusing cooperative group:\n");
    timing(d_x, 2);

    free(h_x);
    CHECK(cudaFree(d_x));
    return 0;
}

void __global__ reduce_syncwarp(const real *d_x, real *d_y, const int N){
    const int tid=threadIdx.x;
    const int bid=blockIdx.x;
    const int n=bid*blockDim.x+tid;
    extern __shared__ real s_y[];
    s_y[tid]=(n<N)?d_x[n]:0.0;
    __syncthreads();
    for(int offset=blockDim.x>>2;offset>=32;offset>>=1){
        if(tid<offset) s_y[tid]+=s_y[tid+offset];
        __syncthreads();
    }
    for(int offset=16;offset>0;offset>>=1){
        if(tid<offset) s_y[tid]+=s_y[tid+offset];
        __syncwarp();
    }
    if(tid==0){
        atomicAdd(d_y,s_y[0]);
    }
}

void __global__ reduce_shfl(const real *d_x, real *d_y, const int N){
    const int tid=threadIdx.x;
    const int bid=blockDim.x;
    const int n=bid*blockDim.x+tid;
    extern __shared__ real s_y[];
    s_y[tid] =(n<N)?d_x[n]:0.0;
    __syncthreads();
     for(int offset=blockDim.x>>2;offset>=32;offset>>=1){
        if(tid<offset) s_y[tid]+=s_y[tid+offset];
        __syncthreads();
    }
    real y=s_y[tid];
    for(int offset=16;offset>0;offset>>=1){
        if(tid<offset) y+=__shfl_down_sync(FULL_MASK,y,offset);
        __syncwarp();
    }
    if(tid==0){
        atomicAdd(d_y,s_y[0]);
    }

}

void __global__ reduce_cp(const real *d_x, real *d_y, const int N){
    const int tid=threadIdx.x;
    const int bid=blockDim.x;
    const int n=bid*blockDim.x+tid;
    extern __shared__ real s_y[];
    s_y[tid] =(n<N)?d_x[n]:0.0;
    __syncthreads();
     for(int offset=blockDim.x>>2;offset>=32;offset>>=1){
        if(tid<offset) s_y[tid]+=s_y[tid+offset];
        __syncthreads();
    }
    real y=s_y[tid];
    thread_block_tile<32> g=tiled_partition<32>(this_thread_block());
    for(int offset=g.size>>1;offset>0;offset>>=1){
        y+=g.shfl_down(y,i);
    }
    if(tid==0){
        atomicAdd(d_y,s_y[0]);
    }
}