#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>
#define THREAD_PER_BLOCK 256
__global__ void reduce(float* a,float* a_d){
    float* input_begin=a+blockDim.x*blockIdx.x;
    int tid=threadIdx.x;
    for(int i=1;i<blockDim.x;i*=2){
        if(tid%(i*2)==0){
            input_begin[tid]+=input_begin[tid+i];
        }
        __syncthreads();
    }
    
    if(tid==0){
        a_d[blockIdx.x]=input_begin[0];
    }



}

bool check(float *out,float *res,int n){
    for(int i=0;i<n;i++){
        if(abs(out[i]-res[i])>0.005)
            return false;
    }
    return true;
}
int main(){
    int N=32*1024*1024;
    float* a=(float* )malloc(N*sizeof(float*));
    float* a_d;
    cudaMalloc((float**)&a_d,N*sizeof(float*));
    int block_num=(N-1)/THREAD_PER_BLOCK+1;
    float* out=(float*)malloc(block_num*sizeof(float*));
    float*out_d;
    float* res=(float*)malloc(block_num*sizeof(float*));
    cudaMalloc((float**)&out_d,block_num*sizeof(float*));
    for(int i=0;i<N;i++){
        a[i]=1;
    }
    
    for(int i=0;i<block_num;i++){
        float cur=0;
        for(int j=0;j<THREAD_PER_BLOCK;j++){
            cur+=a[i*THREAD_PER_BLOCK+j];
        }
        res[i]=cur;
    }
    cudaMemcpy(a_d,a,N*sizeof(float*),cudaMemcpyHostToDevice);

    

    dim3 Grid(block_num,1);
    dim3 Block(THREAD_PER_BLOCK,1);
    reduce<<<Grid,Block>>>(a_d,out_d);
    cudaMemcpy(out,out_d,block_num*sizeof(float*),cudaMemcpyDeviceToHost);
    if(check(out,res,block_num))printf("the ans is right\n");
    else{
        printf("the ans is wrong\n");
        for(int i=0;i<block_num;i++){
            printf("%lf ",out[i]);
        }
        printf("\n");
    }

    cudaFree(a_d);
    cudaFree(out_d);






}