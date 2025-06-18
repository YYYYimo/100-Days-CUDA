#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define THREADS_PER_BLOCK 256

__device__ float gelu(float x) {
    // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    const float sqrt_2_over_pi = 0.7978845608f;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
}

__global__ void geluKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = gelu(data[idx]);
    }
}

void applyGeluCuda(float* h_data, int N) {
    float *d_data;
    size_t size = N * sizeof(float);
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    geluKernel<<<blocks, THREADS_PER_BLOCK>>>(d_data, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

int main() {
    int N = 10;
    float h_data[10] = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f, -3.0f, 4.0f};
    printf("Input:\n");
    for (int i = 0; i < N; ++i) printf("%.4f ", h_data[i]);
    printf("\n");

    applyGeluCuda(h_data, N);

    printf("After GELU:\n");
    for (int i = 0; i < N; ++i) printf("%.4f ", h_data[i]);
    printf("\n");
    return 0;
}