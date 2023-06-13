# Color to Greyscale Conversion

This code demonstrates how to use CUDA to convert a color image to grayscale. The conversion is done by calculating the weighted average of the red, green, and blue components of each pixel in the original image. The result is stored in a new output image.

## Prerequisites
- CUDA Toolkit
- `stb_image.h` and `stb_image_write.h` libraries

## Code Explanation

The code consists of the following files:
- `cuda_runtime.h` and `device_launch_parameters.h`: Header files for CUDA runtime and device launch parameters.
- `stdio.h`: Standard input/output library.
- `stb_image_write.h`: External library for writing images.
- `stb_image.h`: External library for reading images.

The main CUDA kernel function is `colorToGreyscaleConversion`, which performs the conversion using GPU parallelism. It takes the input and output image arrays, as well as the width and height of the image, as arguments. Each thread calculates the grayscale value of a pixel based on the RGB values of the corresponding position in the input image. The grayscale value is then written to the output image.

The `GreyScaleConversion` function handles memory allocation and data transfer between the host (CPU) and the device (GPU). It invokes the `colorToGreyscaleConversion` kernel and copies the resulting image back to the host.

The `main` function demonstrates how to load an image using `stb_image.h`, perform the grayscale conversion using CUDA, and save the resulting image using `stb_image_write.h`.

## Usage
1. Compile the code using the CUDA compiler.
2. Ensure that the input image file path is correctly specified in the code.
3. Run the executable to convert the image to grayscale.
4. The resulting grayscale image will be saved as `write.png`.

## References
- CUDA Toolkit Documentation: [https://docs.nvidia.com/cuda/](https://docs.nvidia.com/cuda/)
- stb_image: [https://github.com/nothings/stb](https://github.com/nothings/stb)
- stb_image_write: [https://github.com/nothings/stb](https://github.com/nothings/stb)

# Image Blurring using CUDA

This code demonstrates how to use CUDA to apply a blur effect to an image. The blur is performed by calculating the average color value of the surrounding pixels for each pixel in the image. The resulting image is stored as output.

## Prerequisites
- CUDA Toolkit
- `lodepng.cpp` library

## Code Explanation

The code consists of the following files:
- `cuda_runtime.h` and `device_launch_parameters.h`: Header files for CUDA runtime and device launch parameters.
- `stdio.h`: Standard input/output library.
- `ctime`: Library for measuring time.
- `iostream`: Standard input/output stream library.
- `lodepng.cpp`: External library for reading and writing PNG images.

The main CUDA kernel function is `blur`, which performs the blur effect using GPU parallelism. It takes the input and output image arrays, as well as the width and height of the image, as arguments. Each thread calculates the average color value of the surrounding pixels for a specific pixel in the image. The resulting color values are written to the output image.

The `filterCPU` function implements the same blur effect using CPU. It iterates over each pixel in the image and calculates the average color value of the surrounding pixels. The resulting color values are written to the output image.

The `filterGPU` function handles memory allocation and data transfer between the host (CPU) and the device (GPU). It invokes the `blur` kernel and copies the resulting image back to the host.

The `main` function demonstrates how to load an image using `lodepng.cpp`, apply the blur effect using both CPU and GPU, and save the resulting images.

## Usage
1. Compile the code using the CUDA compiler.
2. Ensure that the input image file path is correctly specified in the code.
3. Run the executable to apply the blur effect to the image.
4. The resulting images (blurred by CPU and GPU) will be saved as `blurred_CPU.png` and `blurred_GPU.png`, respectively.

## References
- CUDA Toolkit Documentation: [https://docs.nvidia.com/cuda/](https://docs.nvidia.com/cuda/)
- lodepng: [https://lodev.org/lodepng/](https://lodev.org/lodepng/)

