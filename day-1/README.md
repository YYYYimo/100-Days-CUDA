# Day 1: CUDA Basics - Hello World and Vector Addition

## What I Learned Today

### 1. Hello World from GPU (`hello-from-gpu.cu`)
- Created my first CUDA kernel function using `__global__` qualifier
- Learned basic CUDA execution model with `<<<blocks, threads>>>`
- Used `threadIdx.x` to identify thread index
- Implemented GPU-CPU synchronization with `cudaDeviceSynchronize()`

**Key Concepts:**
- Kernel launches are asynchronous
- Thread identification within blocks
- Host-device synchronization

### 2. Vector Addition (`vectorAdd.cu`)
- Implemented parallel vector addition on GPU
- Learned CUDA memory management:
  - `cudaMalloc()` for device memory allocation
  - `cudaMemcpy()` for host-device data transfer
  - `cudaFree()` for memory cleanup
- Calculated proper grid and block dimensions
- Added comprehensive error handling for CUDA operations

**Key Concepts:**
- Memory allocation patterns (host vs device)
- Thread indexing: `blockIdx.x * blockDim.x + threadIdx.x`
- Boundary checking to prevent out-of-bounds access
- Performance timing with CPU clocks

## Code Structure
```
day-1/
©À©¤©¤ hello-from-gpu.cu    # Basic kernel execution
©À©¤©¤ vectorAdd.cu         # Parallel vector operations
©¸©¤©¤ README.md           # This file
```

## Challenges Faced
- Memory management complexity
- Error handling for CUDA operations
- Understanding thread-block organization

## Next Steps
- Explore more complex memory patterns
- Learn about shared memory
- Optimize kernel performance