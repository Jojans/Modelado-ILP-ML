#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <fftw3.h>
#include <chrono>
#include <vector>
#include <iomanip>

using namespace std;

int main() {
    int N = 1 << 20; // 1 millón de puntos (2^20)
    cout << "Ejecutando FFTW con N=" << N << " puntos..." << endl;

    vector<float> in_real(N);
    for (int i = 0; i < N; ++i)
        in_real[i] = sin(2.0 * M_PI * i / N);  // señal de prueba

    fftwf_complex* out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(N, in_real.data(), out, FFTW_ESTIMATE);

    auto t0 = chrono::high_resolution_clock::now();
    fftwf_execute(plan);
    auto t1 = chrono::high_resolution_clock::now();

    double elapsed = chrono::duration<double>(t1 - t0).count();
    cout << fixed << setprecision(6);
    cout << "Tiempo FFTW: " << elapsed << " s" << endl;

    cout << "Resultado (primeros 5):\n";
    for (int i = 0; i < 5; ++i)
        cout << out[i][0] << " + " << out[i][1] << "i\n";

    fftwf_destroy_plan(plan);
    fftwf_free(out);
    return 0;
}