# CUDA Warp and Thread Hierarchy Demonstration

This CUDA C++ program demonstrates key concepts related to GPU thread hierarchy, focusing on **warps**, their execution behavior, and performance implications.

## Features Demonstrated:

1.  **Basic Warp Execution (`basicWarpExecutionKernel`)**:
    *   Illustrates the Single Instruction, Multiple Threads (SIMT) model where threads within a warp execute instructions in lockstep.
    *   Shows how to calculate and use `warpId` and `laneId`.

2.  **Warp Divergence (`divergentWarpKernel`)**:
    *   Shows how conditional branching (`if-else`) within a warp can lead to threads taking different execution paths.
    *   Highlights the performance impact of divergence, as divergent paths are serialized.

3.  **Avoiding Warp Divergence (`nonDivergentWarpKernel`)**:
    *   Demonstrates techniques (e.g., using ternary operators or arithmetic equivalents) to achieve conditional logic without explicit branching, potentially allowing the compiler to use predicated instructions and maintain warp efficiency.

4.  **Shared Memory and Block-Level Synchronization (`sharedMemoryAndSyncKernel`)**:
    *   Illustrates the use of `__shared__` memory for fast data exchange between threads within the same block.
    *   Shows the use of `__syncthreads()` to synchronize all threads in a block, ensuring data consistency when using shared memory. This is crucial for cooperative tasks among warps within a block.


## Program Structure:

*   **Helper Function (`HandleError`)**: A macro for robust CUDA API error checking.
*   **Kernels**: Four distinct GPU kernels, each designed to highlight a specific concept.
*   **Main Function (`main`)**:
    *   Sets up host and device data.
    *   Launches each kernel with appropriate configurations.
    *   Includes `printf` statements for illustrative output from both host and device.
    *   Provides a textual explanation of occupancy.
    *   Performs cleanup of allocated memory.

## How to Compile and Run:

1.  Ensure you have the NVIDIA CUDA Toolkit installed.
2.  Compile the `program.cu` file (or whatever you named it) using `nvcc`:
    ```bash
    nvcc program.cu -o warp_demo
    ```
3.  Run the executable:
    ```bash
    ./warp_demo
    ```

The program will output information from the kernels, demonstrating the behavior of threads and warps, along with the discussion on occupancy. This output is intended to help visualize and understand these core CUDA concepts.