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