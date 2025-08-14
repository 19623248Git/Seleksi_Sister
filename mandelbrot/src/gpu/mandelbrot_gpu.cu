#include "mandelbrot_gpu.hpp"
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>


__global__
void mandelbrot_kernel(unsigned char* imageData, int width, int height, int max_iter) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= height || j >= width) {
        return;
    }

    double real_min = -2.0, real_max = 1.0;
    double imag_min = -1.5, imag_max = 1.5;

    double real = real_min + (double)j / (width - 1) * (real_max - real_min);
    double imag = imag_min + (double)i / (height - 1) * (imag_max - imag_min);

    double z_real = 0.0, z_imag = 0.0;
    int iterations = 0;

    while (z_real * z_real + z_imag * z_imag < 4.0 && iterations < max_iter) {
        double temp_real = z_real * z_real - z_imag * z_imag + real;
        z_imag = 2.0 * z_real * z_imag + imag;
        z_real = temp_real;
        iterations++;
    }

    unsigned char r = 0, g = 0, b = 0;
    if (iterations == max_iter) {
        r = 0; g = 0; b = 0; // Black
    } 
    else {
        r = (iterations % 256);
        g = (iterations * 5 % 256);
        b = (iterations * 10 % 256);
    }

    int pixel_idx = (i * width + j) * 3;

    imageData[pixel_idx + 0] = b;
    imageData[pixel_idx + 1] = g;
    imageData[pixel_idx + 2] = r;
}


MandelbrotGPU::MandelbrotGPU() : hostImage(nullptr), deviceImage(nullptr) {
    config.img_width = 1920;
    config.img_height = 1080;
    config.max_iterations = 1000;
    config.output_filename = "mandelbrot_gpu.png";
}

MandelbrotGPU::~MandelbrotGPU() {
    delete hostImage;
    if (deviceImage) {
        cudaFree(deviceImage);
    }
}

void MandelbrotGPU::run() {
    getUserConfiguration();
    generate();
    saveImage(config.output_filename);
}

void MandelbrotGPU::getUserConfiguration() {
std::cout << "\n--- Mandelbrot Generator Config ---\n";
    std::cout << "Width [default: 1920]: ";
    std::cin >> config.img_width;
    if (config.img_width <= 0) config.img_width = 1920;
    if (config.img_width > 19200) {
        std::cout << "Warning: Width too large, setting to 19200." << std::endl;
        config.img_width = 19200;
    }
    std::cout << "Height [default: 1080]: ";
    std::cin >> config.img_height;
    if (config.img_height <= 0) config.img_height = 1080;
    if (config.img_height > 10800) {
        std::cout << "Warning: Height too large, setting to 10800." << std::endl;
        config.img_height = 10800;
    }
    std::cout << "Max Iterations [default: 1000]: ";
    std::cin >> config.max_iterations;
    if (config.max_iterations < 1) config.max_iterations = 1000;
    if (config.max_iterations > 10000) {
        std::cout << "Warning: Max Iterations too large, setting to 10000." << std::endl;
        config.max_iterations = 10000;
    }
    std::cout << "Output Filename [default: mandelbrot.png]: ";
    std::cin >> config.output_filename;
    if (config.output_filename.empty()) {
        config.output_filename = "mandelbrot.png";
    }
}

void MandelbrotGPU::generate() {
    hostImage = new cv::Mat(config.img_height, config.img_width, CV_8UC3);
    size_t data_size = (size_t)config.img_width * config.img_height * 3 * sizeof(unsigned char);

    cudaError_t err = cudaMalloc(&deviceImage, data_size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Malloc failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    std::cout << "\nGenerating Mandelbrot set (" << config.img_width << "x" << config.img_height << ") on the GPU..." << std::endl;
    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (config.img_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (config.img_height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );
    
    auto start = std::chrono::high_resolution_clock::now();

    mandelbrot_kernel<<<blocksPerGrid, threadsPerBlock>>>(deviceImage, config.img_width, config.img_height, config.max_iterations);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    cudaDeviceSynchronize();

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    cudaMemcpy(hostImage->data, deviceImage, data_size, cudaMemcpyDeviceToHost);

    std::cout << "Generation complete." << std::endl;
    std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;
}

void MandelbrotGPU::saveImage(const std::string& filename) {
    if (!hostImage || hostImage->empty()) {
        std::cerr << "Error: Image has not been generated yet. Cannot save." << std::endl;
        return;
    }

    config.output_filename = "gpu_" + config.output_filename;
    cv::imwrite(filename, *hostImage);
    std::cout << "Image saved as " << filename << std::endl;
}
