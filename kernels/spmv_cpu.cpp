#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <omp.h>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Uso: spmv_cpu <rows> <nnz>\n";
        return 1;
    }

    int rows = atoi(argv[1]);
    int nnz  = atoi(argv[2]);

    std::vector<int>   row_ptr(rows + 1);
    std::vector<int>   col_idx(nnz);
    std::vector<float> values(nnz);
    std::vector<float> x(rows, 1.0f);
    std::vector<float> y(rows, 0.0f);

    int nnz_per_row = nnz / rows;
    for (int i = 0; i <= rows; ++i)
        row_ptr[i] = i * nnz_per_row;

    for (int i = 0; i < nnz; ++i) {
        col_idx[i] = i % rows;
        values[i]  = 1.0f;
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        float sum = 0.0f;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            sum += values[j] * x[col_idx[j]];
        }
        y[i] = sum;
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t2 - t1).count();

    double flops = 2.0 * nnz;
    double gflops = flops / (elapsed * 1e9);

    // Trafico de memoria aproximado (bytes):
    // values[nnz] -> nnz * 4 bytes
    // col_idx[nnz] -> nnz * 4 bytes
    // x[] -> cada acceso a x -> 4 bytes
    // y[] -> una escritura por fila -> 4 bytes
    size_t bytes =
          nnz * sizeof(float)      // values[]
        + nnz * sizeof(int)        // col_idx[]
        + nnz * sizeof(float)      // acceso a x[col]
        + rows * sizeof(float);    // escritura final de y

    double bandwidth = bytes / (elapsed * 1e9); 

    std::cout << "Tiempo (s): " << elapsed << "\n";
    std::cout << "GFLOPS:    " << gflops << "\n";
    std::cout << "Bandwidth aproximado (GB/s): " << bandwidth << "\n";

    return 0; // g++ spmv_cpu.cpp -O2 -o spmv_cpu
}