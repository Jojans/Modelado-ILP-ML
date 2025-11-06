#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <curand.h>
#include <curand_kernel.h>

using real = float;

__global__ void gemm_gpu_kernel(int M, int N, int K, const real* A, const real* B, real* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        real sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

void rand_matrix(std::vector<real>& mat) {
    for (auto& x : mat)
        x = static_cast<real>((rand() / (float)RAND_MAX) * 2.0f - 1.0f);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Uso: gemm_gpu M N K\n";
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);

    size_t sizeA = M * K;
    size_t sizeB = K * N;
    size_t sizeC = M * N;

    std::vector<real> hA(sizeA), hB(sizeB), hC(sizeC);
    rand_matrix(hA);
    rand_matrix(hB);

    real *dA, *dB, *dC;
    cudaMalloc(&dA, sizeA * sizeof(real));
    cudaMalloc(&dB, sizeB * sizeof(real));
    cudaMalloc(&dC, sizeC * sizeof(real));

    cudaMemcpy(dA, hA.data(), sizeA * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), sizeB * sizeof(real), cudaMemcpyHostToDevice);

    dim3 block(16, 16); 
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();

    gemm_gpu_kernel<<<grid, block>>>(M, N, K, dA, dB, dC);
    cudaDeviceSynchronize();

    auto t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = t2 - t1;

    cudaMemcpy(hC.data(), dC, sizeC * sizeof(real), cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    std::cout << elapsed.count() << std::endl;
    return 0;
}