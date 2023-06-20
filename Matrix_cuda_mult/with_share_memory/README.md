# Matrix Multiplication using CUDA

This code performs matrix multiplication using CUDA to leverage the power of GPU parallelism. It demonstrates the use of shared memory to optimize the computation.

## Prerequisites
- CUDA-enabled GPU
- CUDA Toolkit installed

## Compilation
Compile the code using the following command:
nvcc matrix_multiplication.cu -o matrix_multiplication

## Execution
Run the compiled executable:

./matrix_multiplication

## Code Explanation
The code performs matrix multiplication for various matrix dimensions and thread configurations. Here is a breakdown of the code's main components:

- Kernel Function (`matrixMulKernel`):
  - This is the GPU kernel function responsible for performing matrix multiplication.
  - It uses shared memory (`Mds` and `Nds`) to store portions of the input matrices for efficient access.
  - The computation is divided into tiles of size `TILE_WIDTH x TILE_WIDTH`.
  - Each thread computes an element of the resulting matrix by loading data from shared memory and performing the multiplication.
  - The result is stored in global memory.

- `matrixMul` Function:
  - This function sets up the memory on the GPU, copies the input matrices from host to device, launches the kernel, and copies the result back to the host.
  - It allocates global memory on the device for `d_M`, `d_N`, and `d_P` matrices.
  - The kernel is launched with a 2D grid of blocks, where each block contains `TILE_WIDTH x TILE_WIDTH` threads.
  - After the kernel execution, the result (`d_P`) is copied back to the host memory.

- `main` Function:
  - This function generates random input matrices, calls `matrixMul` for different matrix dimensions and thread configurations, and measures the execution time.
  - It initializes the random number generator and iterates over different matrix dimensions and thread configurations.
  - For each combination, it measures the execution time using the `chrono` library and outputs the result.

## Performance Testing
The code measures the execution time for different matrix dimensions (`dimList`) and thread configurations (`numThreadsList`).
The resulting execution times are printed for each combination.

Note: The code provided uses the default thread configurations specified in the `numThreadsList`. Feel free to modify these values based on your requirements.

---
This is a basic README file to help you understand the code and its usage. Feel free to modify and enhance it based on your specific needs.
