/*
 * EXAMPLE OF MATRIX-MATRIX MULTIPLICATION: CHAPTER 4
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <math.h>
#include <chrono>
#include <iostream>
#include <iomanip>
using namespace std;
#define CHECK_ERROR(call) { \
	cudaError_t err = call; \
	if (err != cudaSuccess) { \
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
		exit(err); \
	} \
}

__global__
void matrixMulKernel(float* P, float* M, float* N, int Width) {
    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    int Row = blockDim.y * blockIdx.y + threadIdx.y;

    // check that only the threads with both Row and Col values are in within range
    if (Row < Width && Col < Width) {
        float pValue = 0;
        for (int k = 0; k < Width; k++) {
            pValue += M[Row * Width + k] * N[k * Width + Col];
        }
        P[Row * Width + Col] = pValue;
    }
}


void matrixMul(float* h_P, float* h_M, float* h_N, int dim, int numThreads) {

    int size = (dim * dim) * sizeof(float);
    float* d_M, * d_N, * d_P;

    //1. Allocate global memory on the device for d_Pin and d_Pout
    // With this type of allocation it isn't possible acces using higher-dimensional indexing syntax
    // it need to linearize first.
    CHECK_ERROR(cudaMalloc((void**)&d_M, size));
    CHECK_ERROR(cudaMalloc((void**)&d_N, size));
    CHECK_ERROR(cudaMalloc((void**)&d_P, size));    // assume square matricies

    // copy h_Pin to device memory
    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

    //2. Kernel launch code - with 256 threads per block
    
    dim3 dimGrid(ceil(dim / numThreads), ceil(dim / numThreads), 1);
    dim3 dimBlock(16, 16, 1);
    matrixMulKernel << <dimGrid, dimBlock >> > (d_P, d_M, d_N, dim);

    //3. copy d_Pout from the device memory
    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

    // Free device vectors
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

int main(int argc, char* argv[]) {
    int numThreadsList[] = { 32, 64, 128, 256, 512 };
    int numThreadsSize = sizeof(numThreadsList) / sizeof(numThreadsList[0]);

    int dimList[] = { 256, 512, 1024, 2048, 4096, 8192, 16384, 32768 };
    int dimSize = sizeof(dimList) / sizeof(dimList[0]);

    float* h_M, * h_N, * h_P;
    h_M = (float*)malloc(sizeof(float) * dimList[dimSize - 1] * dimList[dimSize - 1]);
    h_N = (float*)malloc(sizeof(float) * dimList[dimSize - 1] * dimList[dimSize - 1]);
    h_P = (float*)malloc(sizeof(float) * dimList[dimSize - 1] * dimList[dimSize - 1]);

    // fill M and N with random float numbers
    srand(time(NULL));
    for (int i = 0; i < dimSize; i++) {
        int dim = dimList[i];

        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                h_M[j * dim + k] = ((((float)rand() / (float)(RAND_MAX)) * 10));
                h_N[j * dim + k] = ((((float)rand() / (float)(RAND_MAX)) * 10));
            }
        }

        for (int j = 0; j < numThreadsSize; j++) {
            int numThreads = numThreadsList[j];

            auto start = chrono::steady_clock::now();
            // perform matrix multiplication
            matrixMul(h_P, h_M, h_N, dim, numThreads);
            auto end = chrono::steady_clock::now(); // Measure end time
            auto duration = chrono::duration_cast<chrono::duration<double, milli>>(end - start); // Calculate duration in milliseconds

            cout << "Para dim " << dim << " con " << numThreads << " threads, se tiene un tiempo de " << duration.count() << " ms" << endl;
        }

        cout << "--------------------------" << endl;
    }

    // Free host memory
    free(h_M);
    free(h_N);
    free(h_P);

    cout << "¡Multiplicación completada con éxito!" << endl;

    return 0;
}




