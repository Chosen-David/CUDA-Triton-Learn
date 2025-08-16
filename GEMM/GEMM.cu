#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

float testError(void);
float testPerformance(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat);

void cpuSgemm(
    float *a, float *b, float *c, const int M, const int N, const int K) {

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = psum;
        }
    }
}
__global__ void naiveSgemm(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {
        int m = blockIdx.y * blockDim.y + threadIdx.y;
        int n = blockIdx.x * blockDim.x + threadIdx.x;
        if (m < M && n < N) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = psum;
        }

}


__global__ void sgemm_V1(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {
        const int BM=128;
        const int BN=128;
        const int BK=8;
        const int TM=8;
        const int TN=8；

        const int bx=blockIdx.x;
        const int by=blockIdx.y;
        const int tx=threadIdx.x;
        const int ty=threadIdx.y;
        const int tid=ty*blockDim.x+tx;

        __shared__ float s_a[BM][BK];
        __shared__ float s_b[BK][BN];
        float r_c[TM][TN]={0.0};
        int load_a_smem_m=tid>>1;
        int load_a_smem_k=(tid&1)<<2;
        int load_b_smem_k=tid>>5;
        int load_b_smem_n=(tid&31)<<2;

        int load_a_gmem_m=by*BM+load_a_smem_m;
        int load_b_gmem_n=bx*BN+load_b_smem_n;
        for(int bk=0;bk<(K+BK-1)/BK;bk++){
            int load_a_gmem_k=bk*BK+load_a_smem_k;
            int load_a_gmem_addr=OFFSET(load_a_gmem_m,load_a_gmem_k,K);
            int load_b_gmem_k=bk*BK+load_b_smem_k;
            int load_b_gmem_addr=OFFSET(load_b_gmem_k,load_b_gmem_n,N);
            FLOAT4(s_a[load_a_smem_m][load_a_smem_k])=FLOAT4(a[load_a_gmem_addr]);
            FLOAT4(s_b[load_b_smem_k][load_b_smem_n])=FLOAT4(b[load_b_gmem_addr]);
            //可以使用r_load_a[0~3]分别缓存左右两边的数据
            //然后分别s_a[load_a_smem_k+ 0~3][load_a_smem_m]=r_load_a[0~3];

            __syncthreads();
            #pragma unroll
           for(int k=0;k<BK;k++){
            //在这里先r_comp_a/b[0/4]对于共享内存进行加载

            #pragma unroll
            for(int m=0;m<TM;m++){
                #pragma unroll
                for(int n=0;n<TN;n++){
                    r_c[m][n]+=s_a[ty*TM+m][k]*s_b[k][tx*TN+n];
                    //使用r_comp_a和xx_b相乘
                }
            }

           }
           __syncthreads();
           //那么这里就需要两次了，每次加载TM/2下addr和addr+BN/2的数据
           //第二次的，TM/2+BM/2下addr和addr+BN/2的数据
            #pragma unroll
           for(int i=0;i<TM;i++){
            int store_c_gmem_m=by*BM+ty*TM+i;
                for(int j=0;j<TN;j++){
                    int store_c_gmem_n=bx*BN+TN*tx+j;
                    int  store_c_gmem_addr=OFFSET(store_c_gmem_m,store_c_gmem_n,N);
                    FLOAT4(c[store_c_gmem_addr])=FLOAT4(r_c[i][j]);
                }

           }

    }




        



}