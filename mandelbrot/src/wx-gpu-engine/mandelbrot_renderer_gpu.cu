#include "mandelbrot_renderer_gpu.hpp"
#include <cuda_runtime.h>
#include <iostream>

// Helper macro for robust CUDA error checking
#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        std::cerr << "CUDA Error in " << __FILE__ << " at line " << __LINE__ \
                  << ": " << cudaGetErrorString(err_) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// GPU-only function to map iteration count to a color
__device__ Color mapColor(int iterations, int maxIterations) {
    if (iterations == maxIterations) return {0, 0, 0};
    double t = static_cast<double>(iterations) / maxIterations;
    uint8_t r = static_cast<uint8_t>(9 * (1 - t) * t * t * t * 255);
    uint8_t g = static_cast<uint8_t>(15 * (1 - t) * (1 - t) * t * t * 255);
    uint8_t b = static_cast<uint8_t>(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
    return {r, g, b};
}

// The CUDA kernel that calculates the Mandelbrot set
__global__ void mandelbrot_kernel(
    Color* pixels, int width, int height,
    double minReal, double maxReal, double minImag, double maxImag,
    int maxIterations
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= width || idy >= height) return;

    double c_real = minReal + idx * (maxReal - minReal) / (width - 1);
    double c_imag = minImag + idy * (maxImag - minImag) / (height - 1);

    double z_real = 0.0, z_imag = 0.0;
    int iterations = 0;
    while (z_real * z_real + z_imag * z_imag < 4.0 && iterations < maxIterations) {
        double temp_real = z_real * z_real - z_imag * z_imag + c_real;
        z_imag = 2.0 * z_real * z_imag + c_imag;
        z_real = temp_real;
        iterations++;
    }
    
    pixels[idy * width + idx] = mapColor(iterations, maxIterations);
}

// The public-facing C++ wrapper function
void renderWithCuda(
    std::vector<Color>& pixels, int width, int height,
    double minReal, double maxReal, double minImag, double maxImag,
    int maxIterations
) {
    Color* d_pixels = nullptr;
    size_t bufferSize = width * height * sizeof(Color);
    CUDA_CHECK(cudaMalloc(&d_pixels, bufferSize));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    mandelbrot_kernel<<<numBlocks, threadsPerBlock>>>(
        d_pixels, width, height, minReal, maxReal, minImag, maxImag, maxIterations
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(pixels.data(), d_pixels, bufferSize, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_pixels));
}