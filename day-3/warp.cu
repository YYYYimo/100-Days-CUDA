#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h> // For malloc/free
#include <math.h>   // For fabs

// Helper function to check for CUDA errors
static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

// Kernel 1: Basic Warp Execution and SIMT
// Demonstrates how threads in a warp execute the same instruction.
// Prints information for the first few threads to show warp/lane IDs.
__global__ void basicWarpExecutionKernel(float *output, int N) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIdx < N) {
        output[globalIdx] = (float)globalIdx * 2.0f; // Simple operation

        // Print info for the first warp (first 32 threads if blockDim.x >= 32)
        // To keep output manageable, only print for a few threads.
        if (globalIdx < 8) { // Limit printing
            int warpId = threadIdx.x / warpSize; // warpSize is a built-in constant (32)
            int laneId = threadIdx.x % warpSize;
            printf("BasicWarp: GlobalIdx=%d, BlockIdx=%d, ThreadIdx=%d, WarpId=%d, LaneId=%d, Output=%.1f\n",
                   globalIdx, blockIdx.x, threadIdx.x, warpId, laneId, output[globalIdx]);
        }
    }
}

// Kernel 2: Demonstrating Warp Divergence
// Threads in a warp take different paths based on a condition.
__global__ void divergentWarpKernel(const int *input, float *output, int N) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIdx < N) {
        if (input[globalIdx] % 2 == 0) { // Condition causing divergence
            output[globalIdx] = (float)input[globalIdx] * 1.5f;
        } else {
            output[globalIdx] = (float)input[globalIdx] * 0.5f;
        }

        if (globalIdx < 8) { // Limit printing
            int warpId = threadIdx.x / warpSize;
            int laneId = threadIdx.x % warpSize;
            printf("DivergentWarp: GlobalIdx=%d, Input=%d, PathTaken=%s, Output=%.1f\n",
                   globalIdx, input[globalIdx], (input[globalIdx] % 2 == 0 ? "IF" : "ELSE"), output[globalIdx]);
        }
    }
}

// Kernel 3: Avoiding Warp Divergence using Predication/Arithmetic
// Modifies Kernel 2 to reduce or eliminate divergence.
__global__ void nonDivergentWarpKernel(const int *input, float *output, int N) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIdx < N) {
        bool isEven = (input[globalIdx] % 2 == 0);
        // Arithmetic equivalent of the if-else, avoids explicit branching for the hardware
        // Note: True predication is a hardware feature. This C++ code expresses the logic.
        // The compiler might generate predicated instructions if possible.
        float valIfEven = (float)input[globalIdx] * 1.5f;
        float valIfOdd = (float)input[globalIdx] * 0.5f;
        
        output[globalIdx] = isEven ? valIfEven : valIfOdd;
        // Alternative arithmetic approach (less readable but sometimes used):
        // output[globalIdx] = (float)isEven * valIfEven + (float)(!isEven) * valIfOdd;


        if (globalIdx < 8) { // Limit printing
             printf("NonDivergentWarp: GlobalIdx=%d, Input=%d, Output=%.1f\n",
                   globalIdx, input[globalIdx], output[globalIdx]);
        }
    }
}

// Kernel 4: Shared Memory and Block-Level Synchronization
// Demonstrates use of shared memory and __syncthreads().
// __syncthreads() synchronizes ALL threads in a block, not just one warp.
// This is relevant as warps within a block might need to coordinate.
__global__ void sharedMemoryAndSyncKernel(const float *input, float *output, int N) {
    __shared__ float sharedData[64]; // Example shared memory, size of 2 warps if blockDim.x = 64

    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x; // Index within the block

    // Load data into shared memory (assuming blockDim.x <= 64)
    if (localIdx < blockDim.x && globalIdx < N) {
        sharedData[localIdx] = input[globalIdx];
    }

    // Synchronize all threads in the block to ensure sharedData is populated
    __syncthreads();

    if (globalIdx < N) {
        // Example operation using shared memory: e.g., a simple stencil or average
        float result = 0.0f;
        if (localIdx > 0 && localIdx < blockDim.x - 1) {
            result = (sharedData[localIdx - 1] + sharedData[localIdx] + sharedData[localIdx + 1]) / 3.0f;
        } else {
            result = sharedData[localIdx]; // Boundary threads
        }
        output[globalIdx] = result;

        if (globalIdx < 8 && localIdx < 8) { // Limit printing
            printf("SharedMemSync: GlobalIdx=%d, LocalIdx=%d, LoadedToShared=%.1f, Output=%.1f\n",
                   globalIdx, localIdx, (localIdx < blockDim.x ? sharedData[localIdx] : -1.0f) , output[globalIdx]);
        }
    }
}


int main() {
    const int N = 128; // Total number of elements
    const int dataSizeBytes = N * sizeof(float);
    const int dataSizeIntBytes = N * sizeof(int);

    // Host arrays
    float *h_output1, *h_output2, *h_output3, *h_output4;
    int   *h_input_int;
    float *h_input_float;

    h_output1 = (float*)malloc(dataSizeBytes);
    h_output2 = (float*)malloc(dataSizeBytes);
    h_output3 = (float*)malloc(dataSizeBytes);
    h_output4 = (float*)malloc(dataSizeBytes);
    h_input_int = (int*)malloc(dataSizeIntBytes);
    h_input_float = (float*)malloc(dataSizeBytes);

    if (!h_output1 || !h_output2 || !h_output3 || !h_output4 || !h_input_int || !h_input_float) {
        printf("Host memory allocation failed\n");
        return 1;
    }

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_input_int[i] = i;
        h_input_float[i] = (float)i * 1.1f;
    }

    // Device arrays
    float *d_output1, *d_output2, *d_output3, *d_output4;
    int   *d_input_int;
    float *d_input_float;

    HANDLE_ERROR(cudaMalloc((void**)&d_output1, dataSizeBytes));
    HANDLE_ERROR(cudaMalloc((void**)&d_output2, dataSizeBytes));
    HANDLE_ERROR(cudaMalloc((void**)&d_output3, dataSizeBytes));
    HANDLE_ERROR(cudaMalloc((void**)&d_output4, dataSizeBytes));
    HANDLE_ERROR(cudaMalloc((void**)&d_input_int, dataSizeIntBytes));
    HANDLE_ERROR(cudaMalloc((void**)&d_input_float, dataSizeBytes));

    // Copy input data to device
    HANDLE_ERROR(cudaMemcpy(d_input_int, h_input_int, dataSizeIntBytes, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_input_float, h_input_float, dataSizeBytes, cudaMemcpyHostToDevice));

    // --- Kernel Launch Configurations ---
    // For simplicity, let's use a block size that's a multiple of warpSize (32)
    // A common choice is 128, 256, etc.
    int threadsPerBlock = 128; // 4 warps per block
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("=== 1. Basic Warp Execution Kernel ===\n");
    printf("Launching with %d blocks, %d threads/block (%d warps/block)\n", blocksPerGrid, threadsPerBlock, threadsPerBlock / 32);
    basicWarpExecutionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_output1, N);
    HANDLE_ERROR(cudaDeviceSynchronize()); // Wait for kernel to complete
    // (Optionally copy d_output1 to host and print/verify)
    printf("\n");

    printf("=== 2. Divergent Warp Kernel ===\n");
    divergentWarpKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input_int, d_output2, N);
    HANDLE_ERROR(cudaDeviceSynchronize());
    printf("\n");

    printf("=== 3. Non-Divergent Warp Kernel ===\n");
    nonDivergentWarpKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input_int, d_output3, N);
    HANDLE_ERROR(cudaDeviceSynchronize());
    printf("\n");

    printf("=== 4. Shared Memory and Sync Kernel ===\n");
    // For sharedMemoryAndSyncKernel, block size should not exceed shared memory array size (64 in kernel)
    int sharedMemThreadsPerBlock = 64; // 2 warps, matching sharedData size
    int sharedMemBlocksPerGrid = (N + sharedMemThreadsPerBlock - 1) / sharedMemThreadsPerBlock;
    printf("Launching with %d blocks, %d threads/block for shared memory demo\n", sharedMemBlocksPerGrid, sharedMemThreadsPerBlock);
    sharedMemoryAndSyncKernel<<<sharedMemBlocksPerGrid, sharedMemThreadsPerBlock>>>(d_input_float, d_output4, N);
    HANDLE_ERROR(cudaDeviceSynchronize());
    printf("\n");

    // Cleanup
    cudaFree(d_output1); cudaFree(d_output2); cudaFree(d_output3); cudaFree(d_output4);
    cudaFree(d_input_int); cudaFree(d_input_float);
    free(h_output1); free(h_output2); free(h_output3); free(h_output4);
    free(h_input_int); free(h_input_float);

    printf("CUDA Warp Demonstration Finished.\n");
    return 0;
}