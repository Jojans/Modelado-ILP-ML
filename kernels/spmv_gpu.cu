#include <iostream>
#include <cuda_runtime.h>
#include <vector>

__global__ void spmv_kernel(int rows, const int* row_ptr, const int* col_idx, 
                            const float* values, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) {
        float sum = 0.0f;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j)
            sum += values[j] * x[col_idx[j]];
        y[i] = sum;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Uso: spmv_gpu <rows> <nnz>\n";
        return 1;
    }

    int rows = atoi(argv[1]);
    int nnz  = atoi(argv[2]);

    std::vector<int> h_row_ptr(rows + 1);
    std::vector<int> h_col_idx(nnz);
    std::vector<float> h_values(nnz);
    std::vector<float> h_x(rows, 1.0f);
    std::vector<float> h_y(rows, 0.0f);

    int nnz_per_row = nnz / rows;
    for (int i = 0; i <= rows; ++i) h_row_ptr[i] = i * nnz_per_row;
    for (int i = 0; i < nnz; ++i) {
        h_col_idx[i] = i % rows;
        h_values[i] = 1.0f;
    }

    int *d_row_ptr, *d_col_idx;
    float *d_values, *d_x, *d_y;

    cudaMalloc(&d_row_ptr, (rows + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(float));
    cudaMalloc(&d_x, rows * sizeof(float));
    cudaMalloc(&d_y, rows * sizeof(float));

    cudaMemcpy(d_row_ptr, h_row_ptr.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), rows * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    int blockSize = 256;
    int gridSize = (rows + blockSize - 1) / blockSize;
    spmv_kernel<<<gridSize, blockSize>>>(rows, d_row_ptr, d_col_idx, d_values, d_x, d_y);
    cudaEventRecord(stop);

    cudaMemcpy(h_y.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << (milliseconds / 1000.0) << std::endl; // segundos

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}