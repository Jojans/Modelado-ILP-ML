#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define PI 3.14159265358979323846

struct Complex {
    float real;
    float imag;
};

__device__ Complex complexAdd(Complex a, Complex b) {
    return {a.real + b.real, a.imag + b.imag};
}

__device__ Complex complexSub(Complex a, Complex b) {
    return {a.real - b.real, a.imag - b.imag};
}

__device__ Complex complexMul(Complex a, Complex b) {
    return {
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    };
}

__global__ void fftKernel(Complex *data, int n, int step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n / 2) {
        int j = i % step;
        int k = i - j;

        Complex t = data[k + j];
        Complex u = data[k + j + step];

        float angle = -PI * j / step;
        Complex w = {cosf(angle), sinf(angle)};
        Complex v = complexMul(w, u);

        data[k + j]         = complexAdd(t, v);
        data[k + j + step]  = complexSub(t, v);
    }
}

void fftGPU(Complex *h_data, int N) {
    Complex *d_data;
    cudaMalloc(&d_data, N * sizeof(Complex));
    cudaMemcpy(d_data, h_data, N * sizeof(Complex), cudaMemcpyHostToDevice);

    for (int step = 1; step < N; step *= 2) {
        int threads = 256;
        int blocks = (N / 2 + threads - 1) / threads;
        fftKernel<<<blocks, threads>>>(d_data, N, step);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_data, d_data, N * sizeof(Complex), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Uso: fft_gpu N\n"); // N debe ser una potencia de 2, ej: 256, 512, 1024, 2048, 4096
        return 1;
    }
    int N = atoi(argv[1]);

    Complex *data = (Complex *)malloc(N * sizeof(Complex));
    for (int i = 0; i < N; i++) {
        data[i].real = sinf(2.0f * PI * i / N);
        data[i].imag = 0.0f;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    fftGPU(data, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    double seconds = ms / 1000.0;

    double flops = 5.0 * N * log2((double)N);
    double gflops = flops / (seconds * 1e9);

    double bytes = 16.0 * N * log2((double)N);
    double gbs = bytes / (seconds * 1e9);

    printf("Tiempo (s): %f\n", seconds);
    printf("GFLOPS: %f\n", gflops);
    printf("Bandwidth aproximado (GB/s): %f\n", gbs);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(data);

    return 0; // nvcc fft_gpu.cu -o fft_gpu.exe
}