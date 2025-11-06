#include <cuda_runtime.h>
#include <iostream>

using real = float;
#define TILE 16

__global__ void gemm_tiled(const real* A, const real* B, real* C, int M, int N, int K){
    __shared__ real sA[TILE][TILE], sB[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    real acc = 0.0f;
    for (int t=0; t < (K + TILE -1)/TILE; ++t){
        int aRow = row, aCol = t*TILE + threadIdx.x;
        int bRow = t*TILE + threadIdx.y, bCol = col;
        sA[threadIdx.y][threadIdx.x] = (aRow < M && aCol < K) ? A[aRow*K + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < K && bCol < N) ? B[bRow*N + bCol] : 0.0f;
        __syncthreads();
        for (int i=0;i<TILE;i++) acc += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row*N + col] = acc;
}

void launch_gemm(const float* dA, const float* dB, float* dC, int M,int N,int K){
    dim3 block(TILE,TILE);
    dim3 grid( (N + TILE-1)/TILE, (M + TILE-1)/TILE );
    gemm_tiled<<<grid, block>>>(dA,dB,dC,M,N,K);
}