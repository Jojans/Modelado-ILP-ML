#include <iostream>
#include <vector>
#include <random>
#include <chrono>
using namespace std;

void spmv_cpu(int rows, const vector<float>& values, const vector<int>& col_idx,
              const vector<int>& row_ptr, const vector<float>& x, vector<float>& y) {
    for (int i = 0; i < rows; ++i) {
        float sum = 0.0f;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j)
            sum += values[j] * x[col_idx[j]];
        y[i] = sum;
    }
}