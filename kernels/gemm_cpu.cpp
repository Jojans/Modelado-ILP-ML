#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

using real = float;

void gemm_naive(int M, int N, int K, const real* A, const real* B, real* C) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            real s = 0;
            for (int k = 0; k < K; k++) {
                s += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = s;
        }
    }
}

void rand_matrix(int rows, int cols, real* mat, unsigned seed = 123) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<real> dist(-1.0, 1.0);
    for (int i = 0; i < rows * cols; i++)
        mat[i] = dist(rng);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Uso: gemm_cpu <M> <N> <K>\n";
        return 1;
    }

    int M = std::stoi(argv[1]);
    int N = std::stoi(argv[2]);
    int K = std::stoi(argv[3]);

    std::vector<real> A(M * K), B(K * N), C(M * N);
    rand_matrix(M, K, A.data());
    rand_matrix(K, N, B.data());

    auto t0 = std::chrono::high_resolution_clock::now();
    gemm_naive(M, N, K, A.data(), B.data(), C.data());
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << elapsed << std::endl;

    return 0;
}