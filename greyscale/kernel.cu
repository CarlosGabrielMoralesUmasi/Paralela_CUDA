
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

__global__
void colorToGreyscaleConversion(unsigned char* Pout, unsigned char* Pin, int width, int height)
{
    int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    if (colIdx < width && rowIdx < height)
    {
        int pixelIdx = rowIdx * width + colIdx;
        int rgbOffset = pixelIdx * 3; // 3 == CHANNELS
        unsigned char r = Pin[rgbOffset + 0];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];
        float greyScale = 0.21f * r + 0.71f * g + 0.07f * b;
        Pout[rgbOffset + 0] = (int)(greyScale);
        Pout[rgbOffset + 1] = (int)(greyScale);
        Pout[rgbOffset + 2] = (int)(greyScale);
    }
}

void GreyScaleConversion(unsigned char* Pout, unsigned char* Pin, int width, int height, int channel)
{
    unsigned char* d_Pin, * d_Pout;
    int size = width * height * channel * sizeof(unsigned char);

    cudaMalloc((void**)&d_Pin, size);
    cudaMemcpy(d_Pin, Pin, size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_Pout, size);

    dim3 dimGrid(ceil(width / 16.0), ceil(height / 16.0), 1);
    dim3 dimBlock(16, 16, 1);

    colorToGreyscaleConversion << <dimGrid, dimBlock >> > (d_Pout, d_Pin, width, height);

    cudaMemcpy(Pout, d_Pout, size, cudaMemcpyDeviceToHost);

    cudaFree(d_Pin);
    cudaFree(d_Pout);
}

int main()
{
    int w, h, n;
    unsigned char* data = stbi_load("messi1.png", &w, &h, &n, 0);
    unsigned char* oData = new unsigned char[w * h * n];

    GreyScaleConversion(oData, data, w, h, n);

    stbi_write_png("write.png", w, h, n, oData, 0);
    stbi_image_free(data);
    return 0;
}
