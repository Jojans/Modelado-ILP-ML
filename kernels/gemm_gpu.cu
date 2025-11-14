#include <iostream>
#include <vector>
#include <cuda_runtime.h>

using real = float;

__global__ void gemm_naive(int M, int N, int K,
                           const real* A,
                           const real* B,
                           real* C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        real sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void rand_matrix(int rows, int cols, std::vector<real>& mat) {
    for (auto &x : mat)
        x = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Uso: gemm_naive <M> <N> <K>\n";
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);

    size_t sizeA = (size_t)M * K;
    size_t sizeB = (size_t)K * N;
    size_t sizeC = (size_t)M * N;

    std::vector<real> hA(sizeA), hB(sizeB), hC(sizeC);
    rand_matrix(M, K, hA);
    rand_matrix(K, N, hB);

    real *dA, *dB, *dC;
    cudaMalloc(&dA, sizeA * sizeof(real));
    cudaMalloc(&dB, sizeB * sizeof(real));
    cudaMalloc(&dC, sizeC * sizeof(real));

    cudaMemcpy(dA, hA.data(), sizeA * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), sizeB * sizeof(real), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gemm_naive<<<grid, block>>>(M, N, K, dA, dB, dC);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(hC.data(), dC, sizeC * sizeof(real), cudaMemcpyDeviceToHost);

    // --- GFLOPS ---
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = flops / (ms * 1e6);

    // --- Bandwidth (bytes leídos y escritos) ---
    double bytes_read_A = sizeA * sizeof(real);
    double bytes_read_B = sizeB * sizeof(real);
    double bytes_write_C = sizeC * sizeof(real);
    double total_bytes = bytes_read_A + bytes_read_B + bytes_write_C;

    double seconds = ms / 1000.0;
    double bandwidth_GBps = (total_bytes / seconds) / 1e9;

    std::cout << "Tiempo (s): " << seconds << "\n";
    std::cout << "GFLOPS: " << gflops << "\n";
    std::cout << "Bandwidth aproximado (GB/s): " << bandwidth_GBps << "\n";

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0; // nvcc gemm_gpu.cu -o gemm_gpu.exe
} 