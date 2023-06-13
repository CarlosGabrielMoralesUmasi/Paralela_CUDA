//ccomp_paralela_carlos_morales
#include "lodepng.cpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <ctime>
#include <iostream>

using namespace std;

__global__ void blur(unsigned char* input_image, unsigned char* output_image, int width, int height) {

	const unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
	int x = offset % width;
	int y = (offset - x) / width;
	int fsize = 5; // Filter size

	if (offset < width * height) {

		float output_red = 0;
		float output_green = 0;
		float output_blue = 0;
		int hits = 0;

		for (int ox = -fsize; ox < fsize + 1; ++ox) {
			for (int oy = -fsize; oy < fsize + 1; ++oy) {
				if ((x + ox) > -1 && (x + ox) < width && (y + oy) > -1 && (y + oy) < height) {
					const int currentoffset = (offset + ox + oy * width) * 3;
					output_red += input_image[currentoffset];
					output_green += input_image[currentoffset + 1];
					output_blue += input_image[currentoffset + 2];
					hits++;
				}
			}
		}

		output_image[offset * 3] = output_red / hits;
		output_image[offset * 3 + 1] = output_green / hits;
		output_image[offset * 3 + 2] = output_blue / hits;
	}
}

unsigned char* filterCPU(unsigned char* input_image, unsigned char* output_image, int width, int height) {

	for (int offset = 0; offset <= width * height - 1; offset += 1) {

		int x = offset % width;
		int y = (offset - x) / width;
		int fsize = 7; // Filter size

		float output_red = 0;
		float output_green = 0;
		float output_blue = 0;
		int hits = 0;

		for (int ox = -fsize; ox < fsize + 1; ++ox) {
			for (int oy = -fsize; oy < fsize + 1; ++oy) {
				if ((x + ox) > -1 && (x + ox) < width && (y + oy) > -1 && (y + oy) < height) {
					const int currentoffset = (offset + ox + oy * width) * 3;
					output_red += input_image[currentoffset];
					output_green += input_image[currentoffset + 1];
					output_blue += input_image[currentoffset + 2];
					hits++;
				}
			}
		}

		output_image[offset * 3] = output_red / hits;
		output_image[offset * 3 + 1] = output_green / hits;
		output_image[offset * 3 + 2] = output_blue / hits;
	}

	return output_image;
}


cudaError_t filterGPU(unsigned char* input_image, unsigned char* output_image, int width, int height) {

	unsigned char* dev_input;
	unsigned char* dev_output;
	cudaError_t cudaStatus;


	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for input and output vectors.
	cudaStatus = cudaMalloc((void**)&dev_input, width * height * 3 * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_output, width * height * 3 * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vector from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_input, input_image, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	dim3 blockDims(512, 1, 1);
	dim3 gridDims((unsigned int)ceil((double)(width * height * 3 / blockDims.x)), 1, 1);


	blur << <gridDims, blockDims >> > (dev_input, dev_output, width, height);


	// Check for any errors launching the kernel.
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(output_image, dev_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_input);
	cudaFree(dev_output);

	return cudaStatus;

}


int main(int argc, char** argv) {

	cout << "Blurring the given image..." << endl;

	// Read the arguments
	const char* input_file = "messi1.png";
	const char* output_fileGPU = "blurred_GPU.png";
	const char* output_fileCPU = "blurred_CPU.png";

	vector<unsigned char> in_image;
	unsigned int width, height;

	// Load the data
	unsigned error = lodepng::decode(in_image, width, height, input_file);
	if (error) cout << "decoder error " << error << ": " << lodepng_error_text(error) << endl;
	else {
		cout << "Image loaded." << endl;
	}

	// Prepare the data
	unsigned char* input_image = new unsigned char[(in_image.size() * 3) / 4];
	unsigned char* output_imageGPU = new unsigned char[(in_image.size() * 3) / 4];
	unsigned char* output_imageCPU = new unsigned char[(in_image.size() * 3) / 4];
	int where = 0;
	for (int i = 0; i < in_image.size(); ++i) {
		if ((i + 1) % 4 != 0) {
			input_image[where] = in_image.at(i);
			output_imageGPU[where] = 255;
			output_imageCPU[where] = 255;
			where++;
		}
	}

	// Run the filter on CPU
	clock_t t_CPU_start, t_CPU_stop;
	t_CPU_start = clock();
	cout << endl << "Processing on CPU. " << endl;

	output_imageCPU = filterCPU(input_image, output_imageCPU, width, height);

	cout << "CPU Image blurred with success!" << endl;

	t_CPU_stop = clock();
	double t_CPU = (double)(t_CPU_stop - t_CPU_start) / CLOCKS_PER_SEC;
	cout << "CPU - time: " << t_CPU << endl;


	// Run the filter on GPU
	clock_t t_GPU_start, t_GPU_stop;
	t_GPU_start = clock();
	cout << endl << "Processing with CUDA. " << endl;

	cudaError_t cudaStatus = filterGPU(input_image, output_imageGPU, width, height);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "filer failed!");
		return 1;
	}
	else cout << "GPU Image blurred with success!" << endl;

	t_GPU_stop = clock();
	double t_GPU = (double)(t_GPU_stop - t_GPU_start) / CLOCKS_PER_SEC;
	cout << "GPU - time: " << t_GPU << endl << endl;

	// Prepare data for output
	vector<unsigned char> out_imageGPU;
	vector<unsigned char> out_imageCPU;

	for (int i = 0; i < in_image.size() * 3 / 4; ++i) {
		out_imageGPU.push_back(output_imageGPU[i]);
		out_imageCPU.push_back(output_imageCPU[i]);
		if ((i + 1) % 3 == 0) {
			out_imageGPU.push_back(255);
			out_imageCPU.push_back(255);
		}
	}
	// Output the data
	error = lodepng::encode(output_fileGPU, out_imageGPU, width, height);
	if (error == 0)
		error = lodepng::encode(output_fileCPU, out_imageCPU, width, height);

	//if there's an error, display it
	if (error) cout << "encoder error " << error << ": " << lodepng_error_text(error) << endl;
	else {
		cout << "GPU Image saved as " << output_fileGPU << endl;
		cout << "CPU Image saved as " << output_fileCPU << endl;
	}

	delete[] input_image;
	delete[] output_imageGPU;
	delete[] output_imageCPU;
	return 0;
}
