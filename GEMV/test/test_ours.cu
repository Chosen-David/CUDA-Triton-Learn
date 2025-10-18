#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

#include "../gemv_smem.cuh"  // 假设你上面的代码保存在此头文件中

#define CUDA_CHECK(expr) do {                                \
    cudaError_t err = (expr);                                \
    if (err != cudaSuccess) {                                \
        std::cerr << "CUDA error at " << __FILE__ << ":"     \
                  << __LINE__ << " - "                       \
                  << cudaGetErrorString(err) << std::endl;   \
        exit(EXIT_FAILURE);                                  \
    }                                                        \
} while(0)


// ============================= CPU参考实现 =============================
void gemv_cpu(const std::vector<half>& mat, const std::vector<half>& vec,
              std::vector<half>& res, unsigned int M, unsigned int K) {
    for (unsigned int i = 0; i < M; ++i) {
        float acc = 0.0f;
        for (unsigned int j = 0; j < K; ++j) {
            float a = __half2float(mat[i * K + j]);
            float b = __half2float(vec[j]);
            acc += a * b;
        }
        res[i] = __float2half(acc);
    }
}


// ============================= 测试函数 =============================
int main() {
    const unsigned int M = 128;   // 矩阵行数
    const unsigned int K = 1024;  // 矩阵列数（注意要整除 TILE_K=1024）

    std::vector<half> h_mat(M * K);
    std::vector<half> h_vec(K);
    std::vector<half> h_res(M, __float2half(0.0f));
    std::vector<half> h_ref(M, __float2half(0.0f));

    // 初始化随机数据
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : h_mat) x = __float2half(dist(gen));
    for (auto &x : h_vec) x = __float2half(dist(gen));

    // 分配 GPU 内存
    half *d_mat, *d_vec, *d_res;
    CUDA_CHECK(cudaMalloc(&d_mat, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_vec, K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_res, M * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_mat, h_mat.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_res, 0, M * sizeof(half)));

    // ============================= 启动 kernel =============================
    dim3 block(THREADS_PER_BLOCK, 1);
    dim3 grid(1, (M + TILE_M - 1) / TILE_M);

    // 热身运行一次
    launch_gemv_with_smem_max(d_mat, d_vec, d_res, M, K, grid, block);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计时
    const int warmup = 5;
    const int repeat = 20;
    for (int i = 0; i < warmup; ++i) {
        launch_gemv_with_smem_max(d_mat, d_vec, d_res, M, K, grid, block);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeat; ++i) {
        launch_gemv_with_smem_max(d_mat, d_vec, d_res, M, K, grid, block);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count() / repeat;

    // 拷回结果
    CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, M * sizeof(half), cudaMemcpyDeviceToHost));

    // ============================= CPU验证 =============================
    gemv_cpu(h_mat, h_vec, h_ref, M, K);

    // 计算误差
    float max_diff = 0.0f, mse = 0.0f;
    for (unsigned int i = 0; i < M; ++i) {
        float diff = fabs(__half2float(h_res[i]) - __half2float(h_ref[i]));
        mse += diff * diff;
        max_diff = std::max(max_diff, diff);
    }
    mse /= M;

    std::cout << "✅ Kernel average time: " << elapsed << " ms" << std::endl;
    std::cout << "✅ Max abs diff: " << max_diff << ", MSE: " << mse << std::endl;
    std::cout << "✅ Performance: " 
              << 2.0 * M * K / (elapsed * 1e6) << " GFLOPS" << std::endl;

    CUDA_CHECK(cudaFree(d_mat));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_res));
    return 0;
}


/*
nvcc -O3 -arch=sm_86 test_ours.cu -o test_ours
./test_ours

*/