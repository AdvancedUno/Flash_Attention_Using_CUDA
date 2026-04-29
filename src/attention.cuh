// attention.cuh
#pragma once

// naive attention kernel
void naive_attention(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int N, 
    int d,
    float* time_ms
);

// flash attention kernel
void flash_attention(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int N, 
    int d,
    float* time_ms
);
