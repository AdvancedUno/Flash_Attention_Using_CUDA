#include <cuda_runtime.h>
#include <math.h>
#include "attention.cuh"

#define TILE_SIZE 32
#define D_MAX 128 

__global__ void flash_attention_kernel(const float* Q, const float* K, const float* V, float* O, int N, int d) {


    int q_row = blockIdx.x * TILE_SIZE + threadIdx.x;

    // shared memory for K and V tiles
    __shared__ float K_tile[TILE_SIZE][D_MAX];
    __shared__ float V_tile[TILE_SIZE][D_MAX];


    float q_reg[D_MAX] = {};
    if (q_row < N)
    {
        for (int i = 0; i < d; i++)
        {
        	q_reg[i] = Q[q_row * d + i];
        }
    }



    float O_acc[D_MAX] = {};
    float m = -1e9f;
    float l = 0.0f;

    float scale = 1.0f / sqrtf((float)d);


    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;



    for (int tile = 0; tile < num_tiles; tile++) {

        int kv_row = tile * TILE_SIZE + threadIdx.x;

        if (threadIdx.x < TILE_SIZE && kv_row < N) 
        {
            for (int i = 0; i < d; i++)
            {
                K_tile[threadIdx.x][i] = K[kv_row * d + i];
            }
        } 
        else if (threadIdx.x < TILE_SIZE) 
        {
            for (int i = 0; i < d; i++)
            {
                K_tile[threadIdx.x][i] = 0.0f;
            }
        }
        

        // wait for K tile load
        __syncthreads();


        if (threadIdx.x < TILE_SIZE && kv_row < N) 
        {
            for (int i = 0; i < d; i++)
            {
                V_tile[threadIdx.x][i] = V[kv_row * d + i];
            }
                
        } 
        else if (threadIdx.x < TILE_SIZE) 
        {
            for (int i = 0; i < d; i++)
            {
                V_tile[threadIdx.x][i] = 0.0f;
            }
        }
        
        // wait for V tile
        __syncthreads(); 


        // each thread processes its own query row
        if (q_row < N) 
        {

            // compute partial scores
            float scores[TILE_SIZE];
            for (int j = 0; j < TILE_SIZE; j++) 
            {
                float dot = 0.0f;
                for (int i = 0; i < d; i++)
                {
                	dot += q_reg[i] * K_tile[j][i];
                }
                
                scores[j] = dot * scale;
            }

            // online softmax for findinh new max over current tile scores
            float m_new = m;
            for (int j = 0; j < TILE_SIZE; j++)
            {
                m_new = fmaxf(m_new, scores[j]);
            }
               

            // rescale factor for correcting previous accumulations
            float rescale = expf(m - m_new);

            // update running denominator
            float l_new = l * rescale;
            for (int j = 0; j < TILE_SIZE; j++)
            {
                l_new += expf(scores[j] - m_new);
            }

            // rescale output accumulator
            for (int i = 0; i < d; i++) {
                O_acc[i] *= rescale;
                
                for (int j = 0; j < TILE_SIZE; j++)
                {
                    O_acc[i] += expf(scores[j] - m_new) * V_tile[j][i];
                }
            }

            // Update running statistics
            m = m_new;
            l = l_new;
        }

        __syncthreads();
    }
    

    if (q_row < N) 
    {
        for (int i = 0; i < d; i++)
        {
            O[q_row * d + i] = O_acc[i] / l;
        }
    }
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
