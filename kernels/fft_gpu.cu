#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define PI 3.14159265358979323846

struct Complex {
    float real;
    float imag;
};

__device__ Complex complexAdd(Complex a, Complex b) {
    Complex c;
    c.real = a.real + b.real;
    c.imag = a.imag + b.imag;
    return c;
}

__device__ Complex complexSub(Complex a, Complex b) {
    Complex c;
    c.real = a.real - b.real;
    c.imag = a.imag - b.imag;
    return c;
}

__device__ Complex complexMul(Complex a, Complex b) {
    Complex c;
    c.real = a.real * b.real - a.imag * b.imag;
    c.imag = a.real * b.imag + a.imag * b.real;
    return c;
}

__global__ void fftKernel(Complex *data, int n, int step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int m = step * 2;

    if (i < n / 2) {
        int j = i % step;
        int k = i - j;
        Complex t = data[k + j];
        Complex u = data[k + j + step];
        float angle = -PI * j / step;
        Complex w = {cosf(angle), sinf(angle)};
        Complex v = complexMul(w, u);

        data[k + j] = complexAdd(t, v);
        data[k + j + step] = complexSub(t, v);
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
        printf("Uso: fft_gpu N\n");
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
    printf("%f\n", ms / 1000.0f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(data);
    return 0;
}