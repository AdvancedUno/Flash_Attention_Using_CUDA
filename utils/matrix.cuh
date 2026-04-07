#pragma once
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

// Fill a host matrix with random floats
void random_fill(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++)
    {
        mat[i] = (float) rand();
    }  
}

// Allocate a matrix on host and device, fill host with random values and copy to device. 
float* alloc_and_fill(int rows, int cols, float** d_ptr) {
    
    int size = rows * cols * sizeof(float);
    float* h_ptr = (float*)malloc(size);

    random_fill(h_ptr, rows, cols);

    cudaMalloc(d_ptr, size);
    cudaMemcpy(*d_ptr, h_ptr, size, cudaMemcpyHostToDevice);

    return h_ptr;
}
