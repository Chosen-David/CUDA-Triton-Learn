#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

float testError(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K);
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

__global__ void sgemm_V2(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {
       const int BM=128;
        const int BN=128;
        const int BK=8;
        const int TM=8;
        const int TN=8;

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
        float r_load_a[4];
        float r_load_b[4];
        float r_comp_a[TM];
        float r_comp_b[TN];

        //这里对于bk=0单独提取加入s_a和s_b的第0维度，进行循环前从gmem->r_load->smem[0]中

        for(int bk=0;bk<(K+BK-1)/BK;bk++){
            int load_a_gmem_k=bk*BK+load_a_smem_k;
            int load_a_gmem_addr=OFFSET(load_a_gmem_m,load_a_gmem_k,K);
            int load_b_gmem_k=bk*BK+load_b_smem_k;
            int load_b_gmem_addr=OFFSET(load_b_gmem_k,load_b_gmem_n,N);
            FLOAT4(r_load_a[0])=FLOAT4(a[load_a_gmem_addr]);
            FLOAT4(r_load_b[0])=FLOAT4(b[load_b_gmem_addr]);
            //bk这时候从1开始，继续gmem->r_load，但是先别急着->smem
            s_a[load_a_smem_k+0][load_a_smem_m]=r_load_a[0];
            s_a[load_a_smem_k+1][load_a_smem_m]=r_load_a[1];
            s_a[load_a_smem_k+2][load_a_smem_m]=r_load_a[2];
            s_a[load_a_smem_k+3][load_a_smem_m]=r_load_a[3];
            FLOAT4(s_b[load_b_smem_k][load_b_smem_n])=FLOAT4(r_load_b[0]);
  

            __syncthreads();
            //这里算smem->r_c
            #pragma unroll
           for(int k=0;k<BK;k++){
       
                FLOAT4(r_comp_a[0])=FLOAT4(s_a[k][ty*TM/2]);
                FLOAT4(r_comp_a[4])=FLOAT4(s_a[k][ty*TM/2+BM/2]);
                FLOAT4(r_comp_b[0])=FLOAT4(s_b[k][ty*TN/2]);
                FLOAT4(r_comp_b[4])=FLOAT4(s_b[k][ty*TN/2+BN/2]);
                

                #pragma unroll
                for(int m=0;m<TM;m++){
                    #pragma unroll
                    for(int n=0;n<TN;n++){
                        r_c[m][n]+=r_comp_a[m]*r_comp_b[n];
   
                    }
                }
           }

            //在计算的同时，将r_load->smem，因为双缓冲区，smem[0]->r_c计算和r_load->smem[1]的加载可以同时进行


           __syncthreads();
        }
        //把r_c->gmem
            #pragma unroll
           for(int i=0;i<TM/2;i++){
                int store_c_gmem_m=by*BM+ty*TM/2+i;
                int store_c_gmem_n=bx*BN+TN*tx/2;
                int store_c_gmem_addr=OFFSET(store_c_gmem_m,store_c_gmem_n,N);
                FLOAT4(c[store_c_addr])=FLOAT4(r_c[i][0]);
                FLOAT4(c[store_c_addr+BN/2])=FLOAT4(r_c[i][4]);

            }
            #pragma unroll
           for(int i=0;i<TM/2;i++){
                int store_c_gmem_m=by*BM+ty*TM/2+BM/2+i;
                int store_c_gmem_n=bx*BN+TN*tx/2;
                int store_c_gmem_addr=OFFSET(store_c_gmem_m,store_c_gmem_n,N);
                FLOAT4(c[store_c_addr])=FLOAT4(r_c[i+TM/2][0]);
                FLOAT4(c[store_c_addr+BN/2])=FLOAT4(r_c[i+TM/2][4]);
            }
}
