#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "attention.cuh"
#include "timer.cuh"
#include "matrix.cuh"
#include "validate.cuh"


// Number of warmup runs before timing
#define WARMUP_RUNS 3
#define TIMED_RUNS 10

typedef void (*AttnFn)(const float*, const float*, const float*, float*, int, int, float*);

float benchmark(AttnFn fn, const float* d_Q, const float* d_K, const float* d_V, float* d_O, int N, int d) {
    float t;


    for (int i = 0; i < WARMUP_RUNS; i++)
    {
        fn(d_Q, d_K, d_V, d_O, N, d, &t);
    }
        

    float total = 0.0f;
    for (int i = 0; i < TIMED_RUNS; i++) {
        fn(d_Q, d_K, d_V, d_O, N, d, &t);
        total += t;
    }
    return total / TIMED_RUNS;
}

void print_metrics(const char* name, float avg_ms, int N, int d,
                   double mem_bytes) {
    double flops = 4.0 * N * N * d + 5.0 * N * N;
    double gflops = flops / (avg_ms * 1e-3) / 1e9;
    double bw_gbs = mem_bytes / (avg_ms * 1e-3) / 1e9;
    double intensity = flops / mem_bytes;

    printf("  %-18s | %8.3f ms | %7.2f GFLOP/s | " "%7.2f GB/s | %6.2f FLOP/byte\n", name, avg_ms, gflops, bw_gbs, intensity);
}


void run_comparison(int N, int d) {

    printf("│  N=%-4d  d=%-3d                                              │\n", N, d);

    // Allocate
    float *d_Q; 
    float *d_K;
    float *d_V;
    float *d_O;

    float* h_Q = alloc_and_fill(N, d, &d_Q);
    float* h_K = alloc_and_fill(N, d, &d_K);
    float* h_V = alloc_and_fill(N, d, &d_V);
    cudaMalloc(&d_O, N * d * sizeof(float));

    // Correctness (small N only)
    if (N <= 512) {
        float* h_O_gpu = (float*)malloc(N * d * sizeof(float));
        float* h_O_ref = (float*)malloc(N * d * sizeof(float));
        cpu_attention(h_Q, h_K, h_V, h_O_ref, N, d);

        float dummy;
        naive_attention(d_Q, d_K, d_V, d_O, N, d, &dummy);
        cudaMemcpy(h_O_gpu, d_O, N * d * sizeof(float), cudaMemcpyDeviceToHost);
        check_correctness(h_O_gpu, h_O_ref, N, d, "Naive    ");

        flash_attention(d_Q, d_K, d_V, d_O, N, d, &dummy);
        cudaMemcpy(h_O_gpu, d_O, N * d * sizeof(float), cudaMemcpyDeviceToHost);
        check_correctness(h_O_gpu, h_O_ref, N, d, "Flash    ");

        free(h_O_gpu);
        free(h_O_ref);
        printf("\n");
    }

    // Memory traffic estimates
    // Naive: reads Q,K writes S, reads S writes P, reads P,V writes O
    double naive_mem = sizeof(float) * (2.0*N*d +       // read Q, K
        N*N +           // write S
        2.0*N*N +       // read S, write P
        N*N + N*d +     // read P, V
        N*d             // write O
    );
    // Flash: reads Q,K,V once each, writes O once — no N×N traffic
    double flash_mem = sizeof(float) * (
        3.0*N*d +       // read Q, K, V
        N*d             // write O
    );

    // Benchmark
    float naive_ms = benchmark(naive_attention, d_Q, d_K, d_V, d_O, N, d);
    float flash_ms = benchmark(flash_attention, d_Q, d_K, d_V, d_O, N, d);

    printf("  %-18s | %8s ms | %7s GFLOP/s | " "%7s GB/s | %6s FLOP/byte\n", "Kernel", "Time", "Compute", "Bandwidth", "Intensity");
    printf("  %-18s-+-%8s----+-%7s----------+-" "%7s------+-%6s-----------\n", "------------------","--------","-------","-------","------");

    print_metrics("Naive Attention", naive_ms, N, d, naive_mem);
    print_metrics("Flash Attention", flash_ms, N, d, flash_mem);

    printf("\n  Speedup: %.2fx\n", naive_ms / flash_ms);
    printf("  NxN matrix size avoided: %.1f MB\n", (float)(N * N * sizeof(float)) / 1e6);

    // Cleanup
    free(h_Q); free(h_K); free(h_V);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);

}

int main() {
    printf("Flash Attention CUDA\n");

    srand(422);

    // Correctness + small scale
    run_comparison(64,   64);
    run_comparison(256,  64);
    run_comparison(512,  64);

    // Performance sweep
    run_comparison(1024, 64);
    run_comparison(2048, 64);
    run_comparison(4096, 64);

    // Also test d=128
    run_comparison(1024, 128);
    run_comparison(2048, 128);

    return 0;
}
