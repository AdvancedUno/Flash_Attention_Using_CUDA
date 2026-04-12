#include <cuda_runtime.h>
#include <math.h>
#include "attention.cuh"

#define TILE_SIZE 32
#define D_MAX 128 

__global__ void flash_attention_kernel(const float* Q, const float* K, const float* V, float* O, int N, int d) {


}




void flash_attention(const float* d_Q, const float* d_K, const float* d_V, float* d_O, int N, int d, float* time_ms) {
    dim3 block(TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    flash_attention_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, N, d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(time_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}