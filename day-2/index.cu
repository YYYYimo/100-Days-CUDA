#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

// ����ӷ��ں� - չʾ2D����
__global__ void matrixAdd2D(float *A, float *B, float *C, int width, int height) {
    // 2D�߳�����
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // xά�ȣ���
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // yά�ȣ���
    
    if (col < width && row < height) {
        int idx = row * width + col;  // ת��Ϊ1D����
        C[idx] = A[idx] + B[idx];
        
        // ��ӡ������Ϣ����ǰ�����̣߳�
        if (col < 4 && row < 4) {
            printf("Thread(%d,%d): blockIdx(%d,%d), threadIdx(%d,%d), globalIdx=%d\n",
                   col, row, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, idx);
        }
    }
}

// 3D�����ӷ��ں� - չʾ3D����
__global__ void tensor3DAdd(float *A, float *B, float *C, int width, int height, int depth) {
    // 3D�߳�����
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // ���
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // �߶�  
    int z = blockIdx.z * blockDim.z + threadIdx.z;  // ���
    
    if (x < width && y < height && z < depth) {
        int idx = z * (width * height) + y * width + x;  // 3D��1D����ת��
        C[idx] = A[idx] + B[idx];
        
        // ��ӡ3D������Ϣ
        if (x < 2 && y < 2 && z < 2) {
            printf("3D Thread(%d,%d,%d): blockIdx(%d,%d,%d), threadIdx(%d,%d,%d), globalIdx=%d\n",
                   x, y, z, blockIdx.x, blockIdx.y, blockIdx.z, 
                   threadIdx.x, threadIdx.y, threadIdx.z, idx);
        }
    }
}

// ��ϸ���������ں�
__global__ void indexAnalysis(int *output, int N) {
    // ����ȫ���߳�ID
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (globalThreadId < N) {
        // ��������Ϣ�洢���������
        output[globalThreadId] = globalThreadId;
        
        // ��ӡ��ϸ������Ϣ����ǰ16���̣߳�
        if (globalThreadId < 16) {
            printf("\n=== Thread %d Analysis ===\n", globalThreadId);
            printf("blockIdx.x = %d\n", blockIdx.x);
            printf("blockDim.x = %d\n", blockDim.x);
            printf("threadIdx.x = %d\n", threadIdx.x);
            printf("gridDim.x = %d\n", gridDim.x);
            printf("Global Thread ID = %d * %d + %d = %d\n", 
                   blockIdx.x, blockDim.x, threadIdx.x, globalThreadId);
        }
    }
}

// չʾwarp����
__global__ void warpAnalysis() {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = threadIdx.x / 32;  // warp��С�̶�Ϊ32
    int laneId = threadIdx.x % 32;  // ��warp�ڵ�λ��
    
    if (globalId < 64) {  // ֻ��ӡǰ64���߳�
        printf("Global:%d, Block:%d, Thread:%d, Warp:%d, Lane:%d\n",
               globalId, blockIdx.x, threadIdx.x, warpId, laneId);
    }
}

int main() {
    printf("=== CUDA ����ϵͳ��� ===\n\n");
    
    // 1. һά��������
    printf("1. һά��������:\n");
    int N = 32;
    int *d_output;
    cudaMalloc(&d_output, N * sizeof(int));
    
    int threadsPerBlock = 8;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("��������: %d blocks, %d threads per block\n", blocksPerGrid, threadsPerBlock);
    indexAnalysis<<<blocksPerGrid, threadsPerBlock>>>(d_output, N);
    cudaDeviceSynchronize();
    
    // 2. Warp����
    printf("\n2. Warp ����:\n");
    printf("Block size: 64, ÿ��warp 32���߳�\n");
    warpAnalysis<<<1, 64>>>();
    cudaDeviceSynchronize();
    
    // 3. ��ά����ӷ�
    printf("\n3. ��ά��������:\n");
    int width = 8, height = 6;
    int matSize = width * height * sizeof(float);
    
    float *h_A = (float*)malloc(matSize);
    float *h_B = (float*)malloc(matSize);
    float *h_C = (float*)malloc(matSize);
    float *d_A, *d_B, *d_C;
    
    // ��ʼ������
    for(int i = 0; i < width * height; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    
    cudaMalloc(&d_A, matSize);
    cudaMalloc(&d_B, matSize);
    cudaMalloc(&d_C, matSize);
    
    cudaMemcpy(d_A, h_A, matSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matSize, cudaMemcpyHostToDevice);
    
    // 2D��������
    dim3 blockSize2D(4, 3);  // 4x3 = 12���߳�ÿ��
    dim3 gridSize2D((width + blockSize2D.x - 1) / blockSize2D.x,
                    (height + blockSize2D.y - 1) / blockSize2D.y);
    
    printf("2D����: Grid(%d,%d), Block(%d,%d)\n", 
           gridSize2D.x, gridSize2D.y, blockSize2D.x, blockSize2D.y);
    
    matrixAdd2D<<<gridSize2D, blockSize2D>>>(d_A, d_B, d_C, width, height);
    cudaDeviceSynchronize();
    
    // 4. ��ά�����ӷ�
    printf("\n4. ��ά��������:\n");
    int w = 4, h = 3, d = 2;
    int tensorSize = w * h * d * sizeof(float);
    
    float *h_A3D = (float*)malloc(tensorSize);
    float *h_B3D = (float*)malloc(tensorSize);
    float *h_C3D = (float*)malloc(tensorSize);
    float *d_A3D, *d_B3D, *d_C3D;
    
    for(int i = 0; i < w * h * d; i++) {
        h_A3D[i] = 1.0f;
        h_B3D[i] = 2.0f;
    }
    
    cudaMalloc(&d_A3D, tensorSize);
    cudaMalloc(&d_B3D, tensorSize);
    cudaMalloc(&d_C3D, tensorSize);
    
    cudaMemcpy(d_A3D, h_A3D, tensorSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B3D, h_B3D, tensorSize, cudaMemcpyHostToDevice);
    
    // 3D��������
    dim3 blockSize3D(2, 2, 2);  // 2x2x2 = 8���߳�ÿ��
    dim3 gridSize3D((w + blockSize3D.x - 1) / blockSize3D.x,
                    (h + blockSize3D.y - 1) / blockSize3D.y,
                    (d + blockSize3D.z - 1) / blockSize3D.z);
    
    printf("3D����: Grid(%d,%d,%d), Block(%d,%d,%d)\n", 
           gridSize3D.x, gridSize3D.y, gridSize3D.z,
           blockSize3D.x, blockSize3D.y, blockSize3D.z);
    
    tensor3DAdd<<<gridSize3D, blockSize3D>>>(d_A3D, d_B3D, d_C3D, w, h, d);
    cudaDeviceSynchronize();
    
    // �����ڴ�
    cudaFree(d_output);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_A3D); cudaFree(d_B3D); cudaFree(d_C3D);
    free(h_A); free(h_B); free(h_C);
    free(h_A3D); free(h_B3D); free(h_C3D);
    
    printf("\n=== ����ϵͳ��ʾ��� ===\n");
    return 0;
}