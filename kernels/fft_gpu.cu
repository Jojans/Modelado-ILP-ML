#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>
#include <cufft.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

int main() {
    int N = 1 << 20;  // mismo tamaño que FFTW
    cout << "Ejecutando cuFFT con N=" << N << " puntos..." << endl;

    vector<cufftComplex> h_data(N);
    for (int i = 0; i < N; ++i) {
        h_data[i].x = sin(2.0 * M_PI * i / N);
        h_data[i].y = 0.0f;
    }

    cufftComplex* d_data;
    cudaMalloc(&d_data, sizeof(cufftComplex) * N);
    cudaMemcpy(d_data, h_data.data(), sizeof(cufftComplex) * N, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);

    auto start = chrono::high_resolution_clock::now();
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    auto end = chrono::high_resolution_clock::now();

    double elapsed = chrono::duration<double>(end - start).count();
    cout << fixed << "Tiempo cuFFT: " << elapsed << " s" << endl;

    cudaMemcpy(h_data.data(), d_data, sizeof(cufftComplex) * N, cudaMemcpyDeviceToHost);
    cout << "Resultado (primeros 5):\n";
    for (int i = 0; i < 5; ++i)
        cout << h_data[i].x << " + " << h_data[i].y << "i\n";

    cufftDestroy(plan);
    cudaFree(d_data);
    return 0;
}