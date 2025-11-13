#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <algorithm>

using namespace std;

// Radix-2 Cooley-Tukey FFT
void fft(vector<complex<double>>& a) {
    int n = a.size();
    int logn = 0;
    while ((1 << logn) < n) logn++;

    // Bit reversal permutation
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
            swap(a[i], a[j]);
    }

    // Iterative FFT
    for (int len = 2; len <= n; len <<= 1) {
        double ang = -2 * M_PI / len;
        complex<double> wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            complex<double> w(1);
            for (int j = 0; j < len / 2; j++) {
                complex<double> u = a[i + j];
                complex<double> v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Uso: fft_cpu <N>\n";
        return 1;
    }

    int N = atoi(argv[1]);
    if ((N & (N - 1)) != 0) {
        cerr << "N debe ser una potencia de 2\n";
        return 1;
    }

    vector<complex<double>> data(N);
    for (int i = 0; i < N; i++) {
        data[i] = complex<double>(sin(2 * M_PI * i / N), 0.0);
    }

    auto t0 = chrono::high_resolution_clock::now();
    fft(data);
    auto t1 = chrono::high_resolution_clock::now();

    double elapsed = chrono::duration<double>(t1 - t0).count();
    cout << elapsed << endl;
    return 0;
}