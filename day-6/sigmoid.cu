#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h> // For expf on host and for kernel

// CUDA Error Handling Macro
static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("CUDA error: %s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

/**
 * @brief CUDA kernel to compute sigmoid for each element in an array.
 * 
 * @param input Pointer to input data (device memory).
 * @param output Pointer to output data (device memory).
 * @param N Number of elements in the array.
 */
__global__ void sigmoidKernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

/**
 * @brief Host function to manage CUDA operations for applying sigmoid.
 *        This function modifies h_data in-place.
 * 
 * @param h_data Pointer to data array in host memory. This array will be modified.
 * @param N Number of elements in the array.
 */
void applySigmoidCuda(float* h_data, int N) {
    if (N <= 0) {
        printf("Number of elements must be positive.\n");
        return;
    }

    float *d_data_input; 
    // float *d_data_output; // If not modifying in-place
    size_t size_bytes = (size_t)N * sizeof(float);

    // 1. Allocate memory on device
    HANDLE_ERROR(cudaMalloc((void**)&d_data_input, size_bytes));
    // HANDLE_ERROR(cudaMalloc((void**)&d_data_output, size_bytes)); // If using separate output buffer

    // 2. Copy data from host to device
    printf("Copying data from host to device...\n");
    HANDLE_ERROR(cudaMemcpy(d_data_input, h_data, size_bytes, cudaMemcpyHostToDevice));

    // 3. Set kernel launch parameters
    int threadsPerBlock = 256; // Can be tuned based on GPU
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Launching sigmoidKernel with gridDim=%d, blockDim=%d for %d elements...\n",
           blocksPerGrid, threadsPerBlock, N);
    
    // 4. Launch kernel (in-place operation shown here)
    // If using separate output buffer: sigmoidKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data_input, d_data_output, N);
    sigmoidKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data_input, d_data_input, N); 

    // Check for kernel launch errors
    HANDLE_ERROR(cudaGetLastError());
    // Wait for all device operations to complete
    HANDLE_ERROR(cudaDeviceSynchronize());
    printf("Kernel execution finished.\n");

    // 5. Copy results from device to host
    printf("Copying data from device to host...\n");
    // If using separate output buffer: HANDLE_ERROR(cudaMemcpy(h_data, d_data_output, size_bytes, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_data, d_data_input, size_bytes, cudaMemcpyDeviceToHost));

    // 6. Free device memory
    HANDLE_ERROR(cudaFree(d_data_input));
    // HANDLE_ERROR(cudaFree(d_data_output)); // If using separate output buffer
}

int main() {
    int N = 1024; // Example array size
    float* h_array = (float*)malloc(N * sizeof(float));

    if (!h_array) {
        fprintf(stderr, "Failed to allocate host memory for h_array\n");
        return 1;
    }

    // Initialize host array
    printf("Initializing host array...\n");
    for (int i = 0; i < N; ++i) {
        h_array[i] = (float)(i % 10) - 5.0f; // Example values (negative, zero, positive)
    }

    printf("Original values (first 5 and last 5 if N is large enough):\n");
    for (int i = 0; i < (N < 10 ? N : 5); ++i) {
        printf("h_array[%d] = %.4f\n", i, h_array[i]);
    }
    if (N > 5) {
        for (int i = N - (N < 10 ? 0 : 5) ; i < N; ++i) {
             if (i >= 5) printf("h_array[%d] = %.4f\n", i, h_array[i]);
        }
    }

    // Apply sigmoid function on GPU
    applySigmoidCuda(h_array, N);

    printf("\nValues after sigmoid (first 5 and last 5 if N is large enough):\n");
    for (int i = 0; i < (N < 10 ? N : 5); ++i) {
        printf("h_array[%d] = %.4f (original: %.4f, sigmoid: %.4f)\n", i, h_array[i], (float)(i % 10) - 5.0f, 1.0f / (1.0f + expf(-((float)(i % 10) - 5.0f))));
    }
     if (N > 5) {
        for (int i = N - (N < 10 ? 0 : 5) ; i < N; ++i) {
            if (i >= 5) printf("h_array[%d] = %.4f (original: %.4f, sigmoid: %.4f)\n", i, h_array[i], (float)(i % 10) - 5.0f, 1.0f / (1.0f + expf(-((float)(i % 10) - 5.0f))));
        }
    }

    // Free host memory
    free(h_array);

    // Optional: Reset CUDA device
    // HANDLE_ERROR(cudaDeviceReset());

    printf("\nSigmoid example finished.\n");
    return 0;
}