#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define THREADS_PER_BLOCK 256

// CUDA kernel for Monte Carlo integration
__global__ void monteCarloKernel(float *results, int samples, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    curandState state;
    curand_init(seed, idx, 0, &state);

    float sum = 0.0f;
    for (int i = idx; i < samples; i += totalThreads) {
        float x = curand_uniform(&state); // x in (0,1]
        float y = x * x; // Integrate y = x^2
        sum += y;
    }
    results[idx] = sum;
}

int main() {
    int samples = 10000000;
    int blocks = 128;
    int threads = THREADS_PER_BLOCK;
    int totalThreads = blocks * threads;

    float *d_results, *h_results;
    h_results = (float*)malloc(totalThreads * sizeof(float));
    cudaMalloc(&d_results, totalThreads * sizeof(float));

    monteCarloKernel<<<blocks, threads>>>(d_results, samples, time(NULL));
    cudaDeviceSynchronize();

    cudaMemcpy(h_results, d_results, totalThreads * sizeof(float), cudaMemcpyDeviceToHost);

    double total = 0.0;
    for (int i = 0; i < totalThreads; ++i) {
        total += h_results[i];
    }
    double estimate = total / samples; // [0,1]区间，面积就是均值

    printf("Monte Carlo estimate of integral of x^2 over [0,1]: %.6f\n", estimate);
    printf("Theoretical value: 1/3 = %.6f\n", 1.0/3.0);

    cudaFree(d_results);
    free(h_results);
    return 0;
}