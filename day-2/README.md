# Day 2: CUDA Indexing System - Comprehensive Thread Index Demonstration

## What This Program Does

### Overview
This program (`index.cu`) provides a comprehensive demonstration of CUDA's indexing system through four different computational scenarios, showcasing how threads are organized and indexed in 1D, 2D, and 3D configurations.

### Core Functionality

#### 1. **1D Index Analysis** (`indexAnalysis`)
- Demonstrates basic thread indexing with 32 elements across 4 blocks
- Shows detailed breakdown of `blockIdx.x`, `blockDim.x`, `threadIdx.x`, and `gridDim.x`
- Calculates global thread ID using: `globalThreadId = blockIdx.x * blockDim.x + threadIdx.x`

#### 2. **Warp Analysis** (`warpAnalysis`)
- Explores GPU's fundamental execution unit (warp = 32 threads)
- Calculates warp ID and lane position within each warp
- Uses single block with 64 threads to demonstrate 2 warps

#### 3. **2D Matrix Indexing** (`matrixAdd2D`)
- Implements matrix addition using 2D thread blocks and grids
- Maps 2D coordinates (row, col) to 1D memory addresses
- Configuration: 8¡Á6 matrix with 4¡Á3 thread blocks

#### 4. **3D Tensor Indexing** (`tensor3DAdd`)
- Demonstrates 3D indexing for tensor operations
- Shows complex 3D-to-1D address calculation: `idx = z * (width * height) + y * width + x`
- Configuration: 4¡Á3¡Á2 tensor with 2¡Á2¡Á2 thread blocks

## Technical Highlights

**Index Calculation Formulas:**
```cuda
// 1D: Global thread position
globalId = blockIdx.x * blockDim.x + threadIdx.x

// 2D: Matrix coordinates to linear memory
idx = row * width + col

// 3D: Tensor coordinates to linear memory  
idx = z * (width * height) + y * width + x
```

**Memory Management:**
- Proper GPU memory allocation and deallocation
- Host-device data transfers
- Complete error handling and resource cleanup

## Learning Outcomes

This program serves as a complete reference for understanding:
- CUDA's hierarchical thread organization (thread ¡ú block ¡ú grid)
- Index calculations across different dimensionalities
- Warp-level parallelism concepts
- Memory layout patterns for multi-dimensional data

The program outputs detailed index information for each scenario, making it an excellent educational tool for mastering CUDA's indexing fundamentals.