# Grayscale Image Conversion

This script converts an image to grayscale using CUDA. It utilizes the "stb_image" library for loading the input image and the "stb_image_write" library for saving the resulting image.

## Requirements

- CUDA-compatible C++ compiler.
- "stb_image" and "stb_image_write" libraries (included in the code).
- CUDA-compatible GPU.

## Installation and Configuration

1. Make sure you have a CUDA-compatible C++ compiler and the required libraries installed.
2. Download the source code and save it to a file with the ".cu" extension (e.g., "conversion.cu").
3. Ensure you have an input image named "messi1.png" in the same directory as the source file.

## Usage

1. Compile the program using the CUDA-enabled C++ compiler. 
2. As in the blur code, I only use visual studio to compile the code (previously installed nvidia cuda for my pc).
## 
3. The program will load the "messi1.png" image, convert it to grayscale, and save the resulting image as "write.png" in the same directory.

## How It Works

The program consists of the following parts:

### 1. `colorToGreyscaleConversion` Function

This function runs on the GPU threads and performs the grayscale conversion for each pixel of the image. It takes a pointer to the input pixels `Pin`, a pointer to the output pixels `Pout`, the width and height of the image.

### 2. `GreyScaleConversion` Function

This function is the main interface for performing the grayscale conversion using CUDA. It takes the pointers to the output `Pout` and input `Pin` pixels, the width and height of the image, and the number of color channels.

In this function, the following steps are performed:
- Memory is allocated on the device for the input pixels (`d_Pin`) and output pixels (`d_Pout`).
- The input pixel data is copied from the host to the device.
- The block and grid dimensions are configured to execute the `colorToGreyscaleConversion` function in parallel on the GPU.
- The output pixels are copied from the device to the host.
- The allocated memory on the device is freed.

### 3. `main` Function

In the `main` function, the following steps are performed:
- The input image "messi1.png" is loaded using the `stbi_load` function from the "stb_image" library. The width (`w`), height (`h`), and number of channels (`n`) values are obtained.
- A new array `oData` is created to store the output pixels.
- The `GreyScaleConversion` function is called to perform the image conversion.
- The `stbi_write_png` function from the "stb_image_write" library is used to save the output image as "write.png".
- The memory allocated for the input image is freed.

## Results
### Image RGB:
![RGB](messi1.png)
### Image Grey:
![GREY](write.png)





