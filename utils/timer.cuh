#pragma once
#include <cuda_runtime.h>
#include <stdio.h>

struct CudaTimer {
    cudaEvent_t start, stop;

    CudaTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~CudaTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start);
    }


    float Stop()
    {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);


        float ms = 0.0f;
        
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};
