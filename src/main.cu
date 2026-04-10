#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "attention.cuh"
#include "timer.cuh"
#include "matrix.cuh"
#include "validate.cuh"


// Number of warmup runs before timing
#define WARMUP_RUNS  3
#define TIMED_RUNS   10

void run_naive(int N, int d) {

    printf("Naive Attention");


    // Allocate and initialize matrices
    float *d_Q, *d_K, *d_V, *d_O;

    float* h_Q = alloc_and_fill(N, d, &d_Q);
    float* h_K = alloc_and_fill(N, d, &d_K);
    float* h_V = alloc_and_fill(N, d, &d_V);


    cudaMalloc(&d_O, N * d * sizeof(float));

    //Correctness check
    if (N <= 512) {
        float* h_O_gpu = (float*)malloc(N * d * sizeof(float));
        float* h_O_ref = (float*)malloc(N * d * sizeof(float));

        float dummy_time;
        naive_attention(d_Q, d_K, d_V, d_O, N, d, &dummy_time);
        cudaMemcpy(h_O_gpu, d_O, N * d * sizeof(float), cudaMemcpyDeviceToHost);

        cpu_attention(h_Q, h_K, h_V, h_O_ref, N, d);
        check_correctness(h_O_gpu, h_O_ref, N, d, "NaiveAttn");

        free(h_O_gpu);
        free(h_O_ref);
    }



    // Warmup
    float dummy_time;
    for (int i = 0; i < WARMUP_RUNS; i++)
    {
        naive_attention(d_Q, d_K, d_V, d_O, N, d, &dummy_time);
    }


    // This is when to check time
    float total_ms = 0.0f;
    for (int i = 0; i < TIMED_RUNS; i++) {
        float t;
        naive_attention(d_Q, d_K, d_V, d_O, N, d, &t);
        total_ms += t;
    }


    float avg_ms = total_ms / TIMED_RUNS;


    printf("Avg time: %.3f ms\n", avg_ms);



    // Clean ups
    free(h_Q); 
    free(h_K); 
    free(h_V);

    cudaFree(d_Q); 
    cudaFree(d_K); 
    cudaFree(d_V); 
    cudaFree(d_O);
}

int main() {

    printf("Naive Kernel\n");
    run_naive(64, 64);
    run_naive(256, 64);
    run_naive(512, 64);
    run_naive(1024, 64);
    run_naive(2048, 64);
    run_naive(4096, 64);







    return 0;
}
