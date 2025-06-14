#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>

static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("CUDA error: %s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

#define TILE_DIM 16

__global__ void matrixTransposeTiledKernel(const float *A, float *B, int M_A, int N_A) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int globalReadRow = blockIdx.y * TILE_DIM + ty;
    int globalReadCol = blockIdx.x * TILE_DIM + tx;

    if (globalReadRow < M_A && globalReadCol < N_A) {
        tile[ty][tx] = A[globalReadRow * N_A + globalReadCol];
    }

    __syncthreads();

    int globalWriteRow = blockIdx.x * TILE_DIM + ty;
    int globalWriteCol = blockIdx.y * TILE_DIM + tx;

    if (globalWriteRow < N_A && globalWriteCol < M_A) {
        B[globalWriteRow * M_A + globalWriteCol] = tile[tx][ty];
    }
}

void verifyTranspose(const float *original, const float *transposed, int M_orig, int N_orig) {
    printf("Verifying matrix transposition...\n");
    bool pass = true;
    for (int r_trans = 0; r_trans < N_orig; ++r_trans) {
        for (int c_trans = 0; c_trans < M_orig; ++c_trans) {
            float expected_val = original[c_trans * N_orig + r_trans];
            float actual_val = transposed[r_trans * M_orig + c_trans];
            if (fabs(expected_val - actual_val) > 1e-5) {
                printf("Mismatch at Transposed[%d][%d]: Expected=%.5f, GPU=%.5f (Original[%d][%d])\n",
                       r_trans, c_trans, expected_val, actual_val, c_trans, r_trans);
                pass = false;
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
    int M = 2 * TILE_DIM + TILE_DIM / 2;
    int N = 3 * TILE_DIM - TILE_DIM / 3;

    printf("Original Matrix A Dimensions: %d rows x %d cols\n", M, N);
    printf("Transposed Matrix B Dimensions: %d rows x %d cols\n", N, M);
    printf("Tile Dimension: %d x %d\n", TILE_DIM, TILE_DIM);

    float *h_A = (float*)malloc(M * N * sizeof(float));
    float *h_B_gpu = (float*)malloc(N * M * sizeof(float));

    if (!h_A || !h_B_gpu) {
        printf("Host memory allocation failed\n");
        return 1;
    }

    printf("Initializing matrix A...\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            h_A[i * N + j] = (float)(i * 100 + j);
        }
    }

    float *d_A, *d_B;
    HANDLE_ERROR(cudaMalloc((void**)&d_A, M * N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&d_B, N * M * sizeof(float)));

    printf("Copying matrix A from host to device...\n");
    HANDLE_ERROR(cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 numBlocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    printf("Launching kernel with gridDim=(%d, %d) and blockDim=(%d, %d)\n",
           numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);

    matrixTransposeTiledKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, M, N);
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    printf("Copying transposed matrix B from device to host...\n");
    HANDLE_ERROR(cudaMemcpy(h_B_gpu, d_B, N * M * sizeof(float), cudaMemcpyDeviceToHost));

    verifyTranspose(h_A, h_B_gpu, M, N);

    cudaFree(d_A);
    cudaFree(d_B);

    free(h_A);
    free(h_B_gpu);

    printf("Matrix transposition finished.\n");
    return 0;
}