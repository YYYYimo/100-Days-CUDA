#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

static void HandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

#define TILE_DIM 16
__global__ void matrixMulTiledKernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float tileA[TILE_DIM][TILE_DIM];
    __shared__ float tileB[TILE_DIM][TILE_DIM];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;
    float Cvalue = 0.0f;
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        if (row < M && (t * TILE_DIM + tx) < K) {
            tileA[ty][tx] = A[row * K + (t * TILE_DIM + tx)];
        } else {
            tileA[ty][tx] = 0.0f;
        }
        if ((t * TILE_DIM + ty) < K && col < N) {
            tileB[ty][tx] = B[(t * TILE_DIM + ty) * N + col];
        } else {
            tileB[ty][tx] = 0.0f; // Padding if out of bounds
        }
        __syncthreads();
        for (int i = 0; i < TILE_DIM; ++i) {
            Cvalue += tileA[ty][i] * tileB[i][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = Cvalue;
    }
}

void verifyMatrixMul(const float *A, const float *B, const float *C_gpu, int M, int N, int K) {
    printf("Verifying matrix multiplication...\n");
    bool pass = true;
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            float C_cpu_val = 0.0f;
            for (int k_idx = 0; k_idx < K; ++k_idx) {
                C_cpu_val += A[r * K + k_idx] * B[k_idx * N + c];
            }
            if (fabs(C_cpu_val - C_gpu[r * N + c]) > 1e-4) { // Tolerance for float comparison
                printf("Mismatch at C[%d][%d]: CPU=%.5f, GPU=%.5f\n", r, c, C_cpu_val, C_gpu[r * N + c]);
                pass = false;
                // return; // Early exit on first mismatch
            }
        }
    }
    if (pass) {
        printf("Verification PASSED!\n");
    } else {
        printf("Verification FAILED.\n");
    }
}

int main() {
    // Define matrix dimensions
    // For simplicity, let M, N, K be multiples of TILE_DIM, but the kernel handles non-multiples with padding.
    int M = 2 * TILE_DIM; // e.g., 32
    int N = 3 * TILE_DIM; // e.g., 48
    int K = 4 * TILE_DIM; // e.g., 64

    printf("Matrix Dimensions: C(%d x %d) = A(%d x %d) * B(%d x %d)\n", M, N, M, K, K, N);
    printf("Tile Dimension: %d x %d\n", TILE_DIM, TILE_DIM);

    // Allocate host memory
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C_gpu = (float*)malloc(M * N * sizeof(float)); // For GPU result

    if (!h_A || !h_B || !h_C_gpu) {
        printf("Host memory allocation failed\n");
        return 1;
    }

    // Initialize host matrices (e.g., with simple values)
    for (int i = 0; i < M * K; ++i) h_A[i] = (float)(i % 10) + 1.0f; // Simple pattern
    for (int i = 0; i < K * N; ++i) h_B[i] = (float)(i % 7) + 0.5f; // Simple pattern

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    HANDLE_ERROR(cudaMalloc((void**)&d_A, M * K * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&d_B, K * N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&d_C, M * N * sizeof(float)));
    // Copy matrices from host to device
    HANDLE_ERROR(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // Define grid and block dimensions for the kernel launch
    // Each block will be TILE_DIM x TILE_DIM threads
    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    // Grid dimensions: number of tiles needed to cover matrix C
    dim3 numBlocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    printf("Launching kernel with gridDim=(%d, %d, %d) and blockDim=(%d, %d, %d)\n",
           numBlocks.x, numBlocks.y, numBlocks.z,
           threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);

    // Launch the matrix multiplication kernel
    matrixMulTiledKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    HANDLE_ERROR(cudaGetLastError()); // Check for errors during kernel launch
    HANDLE_ERROR(cudaDeviceSynchronize()); // Wait for the kernel to complete

    // Copy the result matrix C from device to host
    HANDLE_ERROR(cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Optional: Verify the result on the CPU
    verifyMatrixMul(h_A, h_B, h_C_gpu, M, N, K);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C_gpu);

    printf("Matrix multiplication finished.\n");
    return 0;
}