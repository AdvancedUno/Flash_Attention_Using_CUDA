// validate.cuh
#pragma once
#include <math.h>
#include <stdio.h>

// CPU reference implementation of attention
void cpu_attention(float* Q, float* K, float* V, float* O, int N, int d) {
    
    float* S = new float[N * N];

    // S = Q @ K^T / sqrt(d)
    float scale = 1.0f / sqrtf((float)d);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            
            for (int k = 0; k < d; k++)
            {
                sum += Q[i * d + k] * K[j * d + k];
            }

            S[i * N + j] = sum * scale;
        }
    }

    // softmax
    for (int i = 0; i < N; i++) {

        float max_val = -1e9f;
        for (int j = 0; j < N; j++)
        {
            max_val = fmaxf(max_val, S[i * N + j]);
        }
            

        float sum = 0.0f;
        for (int j = 0; j < N; j++) 
        {
            S[i * N + j] = expf(S[i * N + j] - max_val);
            sum += S[i * N + j];
        }


        for (int j = 0; j < N; j++)
        {
            S[i * N + j] /= sum;
        }
    }

    // O = softmax(S) @ V
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < d; k++) {
            float sum = 0.0f;
            for (int j = 0; j < N; j++)
            {
                sum += S[i * N + j] * V[j * d + k];
            }
            O[i * d + k] = sum;
        }
    }

    free(S);
}


// Compute max absolute error between two arrays
inline float max_abs_error(float* A, float* B, int N, int d) {
    float max_err = 0.0f;
    for (int i = 0; i < N * d; i++)
    {
        max_err = fmaxf(max_err, fabsf(A[i] - B[i]));
    }
        
    return max_err;
}

// Returns average absolute error
inline float avg_abs_error(float* A, float* B, int N, int d) {

    float total = 0.0f;

    for (int i = 0; i < N * d; i++)
    {
        total += fabsf(A[i] - B[i]);
    }   
    
    return total / (N * d);
}

// Print pass or fail with error values
inline bool check_correctness(float* O_gpu, float* O_ref,int N, int d, const char* kernel_name, float threshold = 1e-3f) {

    float max_err = max_abs_error(O_gpu, O_ref, N, d);
    float avg_err = avg_abs_error(O_gpu, O_ref, N, d);

    bool pass = max_err < threshold;

    printf("[%s] Max error: %.6f | Avg error: %.6f | %s\n", kernel_name, max_err, avg_err, pass ? "PASS" : "FAIL");

    return pass;
}
