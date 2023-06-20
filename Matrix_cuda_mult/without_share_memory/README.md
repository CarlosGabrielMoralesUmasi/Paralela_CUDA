# Matrix-Matrix Multiplication using CUDA

This code demonstrates matrix-matrix multiplication using CUDA, taking advantage of GPU parallelism to accelerate the computation. It showcases a different approach for implementing matrix multiplication compared to the previous example.

## Prerequisites
- CUDA-enabled GPU
- CUDA Toolkit installed

## Compilation
Compile the code using the following command:
- nvcc matrix_multiplication.cu -o matrix_multiplication

## Execution
Run the compiled executable:
- ./matrix_multiplication

## Code Explanation
The code performs matrix-matrix multiplication for different matrix dimensions and thread configurations. Here is a breakdown of the code's main components:

- Kernel Function (`matrixMulKernel`):
  - This is the GPU kernel function responsible for performing matrix multiplication.
  - Each thread calculates a single element of the resulting matrix by iterating over the corresponding row and column.
  - The computation is performed by summing the products of corresponding elements from matrices M and N.
  - The result is stored in global memory.

- `matrixMul` Function:
  - This function sets up the memory on the GPU, copies the input matrices from the host to the device, launches the kernel, and copies the result back to the host.
  - It allocates global memory on the device for matrices `d_M`, `d_N`, and `d_P`.
  - The kernel is launched with a 2D grid of blocks, where each block contains 256 threads (`dimBlock` is set to `16x16`).
  - After the kernel execution, the result (`d_P`) is copied back to the host memory.

- `main` Function:
  - This function generates random input matrices, calls `matrixMul` for different matrix dimensions and thread configurations, and measures the execution time.
  - It initializes the random number generator and iterates over different matrix dimensions and thread configurations.
  - For each combination, it measures the execution time using the `chrono` library and outputs the result.

## Performance Testing
The code measures the execution time for different matrix dimensions (`dimList`) and thread configurations (`numThreadsList`).
The resulting execution times are printed for each combination.

Note: The code provided uses a fixed thread configuration of 256 threads per block. Feel free to modify this value based on your requirements.

---
This is a basic README file to help you understand the code and its usage. Feel free to modify and enhance it based on your specific needs.

