#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>
using namespace nvcuda;

template<const int WMMA_M=16,
         const int WMMA_N=16,
         const int WMMA_K=16>
__global__ void hgemm_v1_wmma_m16n16k16_naive_kernel(const half *A, const half *B, half *C, int M, int N, int K) {
    // Implementation of the kernel
    const int NUM_K_TILES=div_ceil(K, WMMA_K);
    const int load_gmem_a_m=blockIdx.y*WMMA_M;
    const int load_gmem_b_n=blockIdx.x*WMMA_N;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
    wmma::fill_fragment(C_frag, 0.0f);
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> B_frag;
#pragma unroll
    for(int i=0;i<NUM_K_TILES;i++){
        const int load_gmem_a_k=i*WMMA_K;
        const int load_gmem_b_k=i*WMMA_K;
        wmma::load_matrix_sync(A_frag, A+load_gmem_a_m*K+load_gmem_a_k, K);
        wmma::load_matrix_sync(B_frag, B+load_gmem_b_k*N+load_gmem_b_n, N);
        
        //算C=A*B+C
        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    }
    wmma::store_matrix_sync(C+load_gmem_a_m*N+load_gmem_b_n, C_frag, N, wmma::mem_row_major);

}

void hgemm_v1_wmma_m16n16k16_naive(const half *A, const half *B, half *C, int M, int N, int K) {
    // Launch the kernel
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    const dim3 block(32);
    //注意grid的访问顺序
    const dim3 grid(div_ceil(N, WMMA_N), div_ceil(M, WMMA_M));
    hgemm_v1_wmma_m16n16k16_naive_kernel<WMMA_M,WMMA_N,WMMA_K><<<grid, block>>>(A, B, C, M, N, K);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}