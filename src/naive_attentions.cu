#include <cuda_runtime.h>
#include <math.h>
#include "attention.cuh"


// S = Q @ K^T / sqrt(d)
#define BLOCK 16
#define BLOCK_1D 256

__global__ void compute_scores_kernel(const float* Q, const float* K, float*  S, int N, int d
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;


    if (row >= N || col >= N)
    {
        return;
    }

    float scale = 1.0f / sqrtf((float)d);
    float sum = 0.0f;

    for (int i = 0; i < d; i++)
    {
        sum += Q[row * d + i] * K[col * d + i];
    }

    S[row * N + col] = sum * scale;
}



__global__ void softmax_kernel(const float* S,float* P,int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N)
    {
        return;
    }

    const float* row_ptr = S + row * N;
    float* out_ptr = P + row * N;

    // find row max
    float max_val = -FLT_MAX;
    for (int j = 0; j < N; j++)
    {
        max_val = fmaxf(max_val, row_ptr[j]);
    }

    // compute exp and accumulate sum
    float sum = 0.0f;
    for (int j = 0; j < N; j++) 
    {
        out_ptr[j] = expf(row_ptr[j] - max_val);
        sum += out_ptr[j];
    }

    // Pass 3: normalize
    for (int j = 0; j < N; j++)
    {
        out_ptr[j] /= sum;
    }
}


// Kernel 3: Output O = P @ V
__global__ void compute_output_kernel(const float* P, const float* V, float* O, int N, int d
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= d) 
    {
        return;
    }

    float sum = 0.0f;
    for (int j = 0; j < N; j++)
    {
        sum += P[row * N + j] * V[j * d + col];
    }

    O[row * d + col] = sum;
}


void naive_attention(const float* d_Q, const float* d_K, const float* d_V, float* d_O, int N, int d, float* time_ms) {

    // Allocate intermediate N×N matrices on device
    float *d_S, *d_P;
    cudaMalloc(&d_S, N * N * sizeof(float));
    cudaMalloc(&d_P, N * N * sizeof(float));

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    dim3 block2d(BLOCK, BLOCK);
    dim3 grid_scores((N + BLOCK - 1) / BLOCK,(N + BLOCK - 1) / BLOCK);


    dim3 block1d(BLOCK_1D);
    dim3 grid_softmax((N + BLOCK_1D - 1) / BLOCK_1D);


    dim3 grid_output((d + BLOCK - 1) / BLOCK,(N + BLOCK - 1) / BLOCK);


    // Time all three kernels together
    cudaEventRecord(start);

    compute_scores_kernel<<<grid_scores, block2d>>>(d_Q, d_K, d_S, N, d);
    softmax_kernel<<<grid_softmax, block1d>>>(d_S, d_P, N);
    compute_output_kernel<<<grid_output, block2d>>>(d_P, d_V, d_O, N, d);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(time_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_S);
    cudaFree(d_P);
}
