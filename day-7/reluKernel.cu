#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h> // For fmaxf in kernel (optional, can use if/else)

// CUDA Error Handling Macro
static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("CUDA error: %s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

/**
 * @brief CUDA kernel to compute ReLU for each element in an array.
 *        Modifies the data in-place.
 * 
 * @param data Pointer to data (device memory). This array will be modified.
 * @param N Number of elements in the array.
 */
__global__ void reluKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        if (data[idx] < 0.0f) {
            data[idx] = 0.0f;
        }
        // Alternative using fmaxf:
        // data[idx] = fmaxf(0.0f, data[idx]);
    }
}

/**
 * @brief Host function to manage CUDA operations for applying ReLU.
 *        This function modifies h_data in-place.
 * 
 * @param h_data Pointer to data array in host memory. This array will be modified.
 * @param N Number of elements in the array.
 */
void applyReluCuda(float* h_data, int N) {
    if (N <= 0) {
        printf("Number of elements must be positive.\n");
        return;
    }

    float *d_data; 
    size_t size_bytes = (size_t)N * sizeof(float);

    // 1. Allocate memory on device
    HANDLE_ERROR(cudaMalloc((void**)&d_data, size_bytes));

    // 2. Copy data from host to device
    printf("Copying data from host to device...\n");
    HANDLE_ERROR(cudaMemcpy(d_data, h_data, size_bytes, cudaMemcpyHostToDevice));

    // 3. Set kernel launch parameters
    int threadsPerBlock = 256; // Can be tuned based on GPU
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Launching reluKernel with gridDim=%d, blockDim=%d for %d elements...\n",
           blocksPerGrid, threadsPerBlock, N);
    
    // 4. Launch kernel (in-place operation)
    reluKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N); 

    // Check for kernel launch errors
    HANDLE_ERROR(cudaGetLastError());
    // Wait for all device operations to complete
    HANDLE_ERROR(cudaDeviceSynchronize());
    printf("Kernel execution finished.\n");

    // 5. Copy results from device to host
    printf("Copying data from device to host...\n");
    HANDLE_ERROR(cudaMemcpy(h_data, d_data, size_bytes, cudaMemcpyDeviceToHost));

    // 6. Free device memory
    HANDLE_ERROR(cudaFree(d_data));
}

int main() {
    int N = 1024; // Example array size
    float* h_array = (float*)malloc(N * sizeof(float));

    if (!h_array) {
        fprintf(stderr, "Failed to allocate host memory for h_array\n");
        return 1;
    }

    // Initialize host array with some positive and negative values
    printf("Initializing host array...\n");
    for (int i = 0; i < N; ++i) {
        h_array[i] = (float)(i % 10) - 5.0f; // Example values: -5.0, -4.0, ..., 0.0, 1.0, ..., 4.0
    }

    printf("Original values (first 10 and last 5 if N is large enough):\n");
    for (int i = 0; i < (N < 15 ? N : 10); ++i) {
        printf("h_array[%d] = %.2f\n", i, h_array[i]);
    }
    if (N > 10) {
        printf("...\n");
        for (int i = N - (N < 15 ? 0 : 5) ; i < N; ++i) {
             if (i >= 10) printf("h_array[%d] = %.2f\n", i, h_array[i]);
        }
    }

    // Apply ReLU function on GPU
    applyReluCuda(h_array, N);

    printf("\nValues after ReLU (first 10 and last 5 if N is large enough):\n");
    for (int i = 0; i < (N < 15 ? N : 10); ++i) {
        float original_val = (float)(i % 10) - 5.0f;
        printf("h_array[%d] = %.2f (original: %.2f)\n", i, h_array[i], original_val);
    }
    if (N > 10) {
        printf("...\n");
        for (int i = N - (N < 15 ? 0 : 5) ; i < N; ++i) {
            if (i >= 10) {
                float original_val = (float)(i % 10) - 5.0f;
                printf("h_array[%d] = %.2f (original: %.2f)\n", i, h_array[i], original_val);
            }
        }
    }

    // Free host memory
    free(h_array);

    // Optional: Reset CUDA device
    // HANDLE_ERROR(cudaDeviceReset());

    printf("\nReLU example finished.\n");
    return 0;
}