#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "cuda_runtime_api.h"
#include <algorithm>
#include <algorithm>
#include <assert.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef K_STAGE
#define K_STAGE 3
#endif

#ifndef TILE_K
#define TILE_K 1024u
#endif

#ifndef TILE_M
#define TILE_M 4u
#endif

//假设x方向1个block(一个block一个warp)，那么搬运32 * THREADS_COPY_ELEM = 256 列，然后直接warpreduce写回就行
//假设y方向8个block，那么搬运8 * TILE_M = 64行
//假设k方向TILE个
/*
对于每个tid，
从vec搬运1 * THREAD_COPY_ELEM
从mat搬运8 * THREAD_COPY_ELEM -> 循环到取完
存到reg里面算出tid_sum，

循环到搬完vec的一个
*/

#define BLOCK_COLS 1024
#define BLOCK_ROWS 8

#define THREAD_COPY_ELEM 8

#define WARP_COLS (THREAD_COPY_ELEM * WARP_SIZE)
#define WARP_ROWS 1

#define THREADS_PER_BLOCK (BLOCK_ROWS * BLOCK_COLS / THREAD_COPY_ELEM)

#define THREADS_PER_TILE_K (TILE_K / THREAD_COPY_ELEM)

//其实我这里就是多少次搬运完一个stage的所有的数据（TILE_K个）
#define REG_VEC_TILES (THREADS_PER_TILE_K / THREADS_PER_BLOCK)
//其实这里就是y方向分配多少个block
#define REG_MAT_TILES 4


#ifndef SHARED_MEM_MAX_ROWS
#define SHARED_MEM_MAX_ROWS 64
#endif

#define CHUNK_K (TILE_K / THREAD_COPY_ELEM)

// all cache
#define CP_ASYNC_CA(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global.L2::256B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

//only L2 cache
#define CP_ASYNC_CG(dst, src, Bytes) \
  asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], %2;\n" \
               :: "l"((unsigned long long)(dst)), \
                  "l"((unsigned long long)(src)), "n"(Bytes))


#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))

#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)

#define LDS2R_X4(R0, R1, R2, R3, addr) \
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n" \
                  : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)    \
                  : "r"(addr))

//运行前检查

static_assert(TILE_K % (THREAD_COPY_ELEM * THREADS_PER_BLOCK) == 0,
              "TILE_K 需要是一个block搬运元素的整数倍");




// device 上的累加函数
__device__ __forceinline__
void compute(half2 &acc, const half2 vec_vals[4], const half2 mat_vals[4]) {
    acc = __hfma2(vec_vals[0], mat_vals[0], acc);
    acc = __hfma2(vec_vals[1], mat_vals[1], acc);
    acc = __hfma2(vec_vals[2], mat_vals[2], acc);
    acc = __hfma2(vec_vals[3], mat_vals[3], acc);
}

__device__ __forceinline__
void load_vec_smem2reg(
    half2 dst[4],        
    half *smem_stage_ptr
) {
    int addr = __cvta_generic_to_shared(smem_stage_ptr);
    uint32_t r0, r1, r2, r3;
    LDS2R_X4(r0, r1, r2, r3, addr);
    dst[0] = *reinterpret_cast<half2*>(&r0);
    dst[1] = *reinterpret_cast<half2*>(&r1);
    dst[2] = *reinterpret_cast<half2*>(&r2);
    dst[3] = *reinterpret_cast<half2*>(&r3);
}

__device__ __forceinline__
void load_mat_gmem2reg(half2 dst[4], float4 mat_val) {
    dst[0] = *(half2*)&mat_val.x;
    dst[1] = *(half2*)&mat_val.y;
    dst[2] = *(half2*)&mat_val.z;
    dst[3] = *(half2*)&mat_val.w;
}






extern "C" __global__
void gemv_smem(half* __restrict__ mat,   // [M x K], row-major
               half* __restrict__ vec,   // [K]
               half* __restrict__ res,   // [M]
               unsigned int K,           // K
               unsigned int M) {


                            
  // each thread load num_per_thread elements from global
  size_t tid = threadIdx.x; 
  const size_t warp_id = tid / WARP_SIZE;
  const size_t lane_id = tid % WARP_SIZE;

  size_t row = (blockIdx.y * blockDim.y + threadIdx.y) * TILE_M;
  // size_t num_stages = (K + TILE_K - 1) / TILE_K;
  size_t row_stage_num = (M + TILE_M - 1) / TILE_M;

  // size_t start_idx = threadIdx.x;
  float4* mat4 = reinterpret_cast<float4*>(mat);
  // float4* vec4 = reinterpret_cast<float4*>(vec);
  __shared__ half smem_raw[K_STAGE - 1][TILE_K];

  //4个half2正好是一个tid算的量
  half2 reg_vec[2][REG_VEC_TILES][4];
  half2 reg_mat[2][REG_MAT_TILES][4];
  half2 reg_res[REG_VEC_TILES][REG_MAT_TILES][2];

  const half* vec_warp_ptr = &vec[row * K + (blockDim.x * blockIdx.x + warp_id * WARP_SIZE) * THREAD_COPY_ELEM];
  
  size_t stage_store_smem = 0;
  float4 *vec_lane_ptr = nullptr;
  // float4 *mat_lane_ptr = nullptr; 

  for(size_t row_stage = 0; row_stage < row_stage_num; row_stage++){
    
    //先所有线程共同加载第一个buffer
    #pragma unroll
    for(size_t stage_load_id = 0; stage_load_id < (K_STAGE - 1); ++stage_load_id){
      //g2s
      vec_lane_ptr = (float4 *)(vec_warp_ptr + (lane_id / THREADS_PER_TILE_K) * TILE_K + stage_load_id * TILE_K) + 
                      (lane_id % THREADS_PER_TILE_K);
      
      size_t vec_smem_ptr = __cvta_generic_to_shared(&smem_raw[stage_load_id][0]) + 
          lane_id % THREADS_PER_TILE_K;

      CP_ASYNC_CG(vec_smem_ptr, vec_lane_ptr, THREAD_COPY_ELEM * sizeof(half));

      CP_ASYNC_COMMIT_GROUP();

    }
    
    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();

    //标记存到哪以及用到哪
    size_t stage_store_id = (K_STAGE - 1);
    size_t stage_use_id = 0;
    size_t reg_store_id = 0;
    size_t reg_use_id = 0;

    //第一次寄存器进入
    //每个tid从smem里面取数据到自己的reg，搬运完stage_use_id所对应smem
    size_t offset = (tid % THREADS_PER_TILE_K) * THREAD_COPY_ELEM;
    #pragma unroll
    for (size_t i = 0; i < REG_VEC_TILES; i++) {
        load_vec_smem2reg(
            reg_vec[reg_store_id][i],
            &smem_raw[stage_use_id][i * TILE_K + offset]
        );
    }


    float4 mat_val = mat4[row * (K >> 3) + offset];

    load_mat_gmem2reg(reg_mat[reg_store_id][row], mat_val);

    reg_store_id = (reg_store_id + 1) % 2; //这里寄存器第一个区域已经存满了
    stage_use_id = (stage_use_id + 1) % (K_STAGE - 1); //这里第一个stage也被加载完了

    #pragma unroll
    //你已经预加载了前K_STAGE-1个了，所以注意起始位置
    //注意你应该设置K正好整除TILE_K
    for(size_t tile_k = TILE_K * (K_STAGE - 1); tile_k < K; tile_k += TILE_K){
      //载入第二个寄存器，算第一个。注意reg_store_id已经更新了
        #pragma unroll
        for (size_t i = 0; i < REG_VEC_TILES; i++) {
            load_vec_smem2reg(
                reg_vec[reg_store_id][i],
                &smem_raw[stage_use_id][i * THREAD_COPY_ELEM * THREADS_PER_BLOCK + offset]
            );
        }

        mat_val = mat4[row * (K >> 3) + offset];

        load_mat_gmem2reg(
            reg_mat[reg_store_id][row],
            mat_val
        );

        #pragma unroll
        for (size_t i = 0; i < REG_VEC_TILES; i++) {
            half2 acc = reg_res[i][row][reg_use_id];
            compute(acc, reg_vec[reg_use_id][i], reg_mat[reg_use_id][row]);
            reg_res[i][row][reg_use_id] = acc;
        }

        //加载下一轮buffer到smem
        #pragma unroll
        for(size_t stage_load_id = 0; stage_load_id < (K_STAGE - 1); ++stage_load_id){
          //g2s
          vec_lane_ptr = (float4 *)(vec_warp_ptr + (lane_id / THREADS_PER_TILE_K) * K + stage_store_smem * TILE_K) + 
                          (lane_id % THREADS_PER_TILE_K);
          size_t vec_smem_ptr = __cvta_generic_to_shared(&smem_raw[stage_load_id][0]) + 
              lane_id % THREADS_PER_TILE_K;

          CP_ASYNC_CG(vec_smem_ptr, vec_lane_ptr, THREAD_COPY_ELEM * sizeof(half));
          CP_ASYNC_COMMIT_GROUP();

        }
        
        CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
        __syncthreads();

        //让多级流水buffer流动起来
        stage_use_id = (stage_use_id + 1) % K_STAGE;
        stage_store_id = (stage_store_id + 1) % K_STAGE;
        reg_use_id = (reg_use_id + 1) % 2;
        reg_store_id = (reg_use_id + 1) % 2;

        //第一次寄存器进入
        #pragma unroll
        for (size_t i = 0; i < REG_VEC_TILES; i++) {
            load_vec_smem2reg(
                reg_vec[reg_store_id][i],
                smem_raw[stage_use_id],
                offset
            );
        }

        mat_val = mat4[row * (K >> 3) + offset];
        load_mat_gmem2reg(
            reg_mat[reg_store_id][row],
            mat_val
        );

        //算第二个寄存器
        #pragma unroll
        for (size_t i = 0; i < REG_VEC_TILES; i++) {
            half2 acc = reg_res[i][row][reg_use_id];
            compute(acc, reg_vec[reg_use_id][i], reg_mat[reg_use_id][row]);
            reg_res[i][row][reg_use_id] = acc;
        }

        reg_store_id = (reg_store_id + 1) % 2;
        reg_use_id = (reg_use_id + 1) % 2;

    }

    //尾部处理
    #pragma unroll
    for(size_t stage = 0; stage < (K_STAGE - 2); ++stage){
      #pragma unroll
      for(size_t k_step = 0; k_step < (TILE_K / THREAD_COPY_ELEM); ++k_step){
          //smem2reg
          #pragma unroll
          for (size_t i = 0; i < REG_VEC_TILES; i++) {
              load_vec_smem2reg(
                  reg_vec[reg_store_id][i],
                  smem_raw[stage_use_id],
                  offset
              );
          }


          mat_val = mat4[row * (K >> 3) + offset];
          load_mat_gmem2reg(
              reg_mat[reg_store_id][row],
              mat_val
          );

          #pragma unroll
          for (size_t i = 0; i < REG_VEC_TILES; i++) {
              half2 acc = reg_res[i][row][reg_use_id];
              compute(acc, reg_vec[reg_use_id][i], reg_mat[reg_use_id][row]);
              reg_res[i][row][reg_use_id] = acc;
          }

          reg_store_id = (reg_store_id + 1) % 2;
          reg_use_id = (reg_use_id + 1) % 2;
          if (k_step == 0){
              stage_use_id = (stage_use_id + 1) % K_STAGE;
              CP_ASYNC_WAIT_GROUP(0);
              __syncthreads();
          }
      }
    }

      #pragma unroll
      for (size_t k_step = 1; k_step < CHUNK_K; ++k_step) {         
          //-------------------------------load smem to reg-----------------------------------
          #pragma unroll
          for (size_t i = 0; i < REG_VEC_TILES; i++) {
              load_vec_smem2reg(
                  reg_vec[reg_store_id][i],
                  smem_raw[stage_use_id],
                  offset
              );
          }

          mat_val = mat4[row * (K >> 3) + offset];
          load_mat_gmem2reg(
              reg_mat[reg_store_id][row],
              mat_val
          );

          reg_store_id = (reg_store_id + 1) % 2;
          reg_use_id = (reg_use_id + 1) % 2;
        }

        #pragma unroll
        for (size_t i = 0; i < REG_VEC_TILES; i++) {
            half2 acc = reg_res[i][row][reg_use_id];
            compute(acc, reg_vec[reg_use_id][i], reg_mat[reg_use_id][row]);
            reg_res[i][row][reg_use_id] = acc;
        }

        __syncthreads();

        //写回全局内存
        #pragma unroll
        for(size_t i = 0; i < REG_VEC_TILES; i++){
            res[row] += __low2half(__hadd2(reg_res[i][row][0],reg_res[i][row][1])) + __high2half(__hadd2(reg_res[i][row][0],reg_res[i][row][1]));
        }

  }
}


inline void launch_gemv_with_smem_max(
    half* d_mat, half* d_vec, half* d_res,
    unsigned int M, unsigned int K,
    dim3 grid, dim3 block,
    cudaStream_t stream = 0) {

  //运算前的检查
、

  // const int shared_bytes =
  //     static_cast<int>(TILE_K) * static_cast<int>(K_STAGE) * sizeof(half);

  gemv_smem<<<grid, block>>>(
      d_mat, d_vec, d_res, K, M);
}
