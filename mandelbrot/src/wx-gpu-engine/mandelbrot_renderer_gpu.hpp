#ifndef MANDELBROT_RENDERER_GPU_H
#define MANDELBROT_RENDERER_GPU_H

#include <vector>
#include <cstdint>

#include "../common/common_types.h"

/**
 * @brief Generates the Mandelbrot set on the GPU using CUDA.
 * * This function orchestrates the memory allocation on the GPU, kernel launch,
 * and copying the results back to the host's pixel buffer.
 *
 * @param pixels A reference to the host-side pixel buffer (std::vector).
 * @param width The width of the image.
 * @param height The height of the image.
 * @param minReal The minimum value on the real axis.
 * @param maxReal The maximum value on the real axis.
 * @param minImag The minimum value on the imaginary axis.
 * @param maxImag The maximum value on the imaginary axis.
 * @param maxIterations The escape-time iteration limit.
 */
void renderWithCuda(
    std::vector<Color>& pixels, int width, int height,
    double minReal, double maxReal, double minImag, double maxImag,
    int maxIterations
);

#endif // MANDELBROT_RENDERER_GPU_H