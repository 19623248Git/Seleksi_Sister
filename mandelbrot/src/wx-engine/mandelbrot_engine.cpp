#include "mandelbrot_engine.hpp"
#include <algorithm> // For std::max
#include <iostream>  // <-- ADDED: Optional, for logging thread count

MandelbrotEngine::MandelbrotEngine() : maxIterations(255) {}

void MandelbrotEngine::setMaxIterations(int iterations) {
    maxIterations = std::max(1, iterations);
}

// --- ADDED: The missing getter function ---
int MandelbrotEngine::getMaxIterations() const { return maxIterations; }

// =========================================================================
// NEW MULTITHREADED IMPLEMENTATION
// =========================================================================
void MandelbrotEngine::generateMandelbrot(
    std::vector<Color>& pixels, int width, int height,
    double minReal, double maxReal, double minImag, double maxImag
) const {
    // 1. Determine how many threads to use (usually the number of cores)
    const unsigned int num_threads = std::thread::hardware_concurrency();
    std::cout << "INFO: Using " << num_threads << " threads for generation." << std::endl;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    const int rows_per_thread = height / num_threads;

    // 2. Define the "worker" function that each thread will run.
    // A lambda function is perfect for this.
    auto worker = [=, &pixels](int startY, int endY) {
        double real_factor = (maxReal - minReal) / (width - 1);
        double imag_factor = (maxImag - minImag) / (height - 1);

        // Process the assigned rows
        for (int y = startY; y < endY; ++y) {
            double c_imag = minImag + y * imag_factor;
            for (int x = 0; x < width; ++x) {
                double c_real = minReal + x * real_factor;
                // Each thread writes its result directly into the shared pixel vector
                pixels[y * width + x] = calculatePixelColor(c_real, c_imag);
            }
        }
    };

    // 3. Launch all the threads
    for (unsigned int i = 0; i < num_threads; ++i) {
        int startY = i * rows_per_thread;
        // The last thread handles any remaining rows if height isn't perfectly divisible
        int endY = (i == num_threads - 1) ? height : startY + rows_per_thread;
        
        threads.emplace_back(worker, startY, endY);
    }

    // 4. Wait for all threads to finish their work (synchronization point)
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
}

// =========================================================================
// ORIGINAL PIXEL CALCULATION (UNCHANGED)
// =========================================================================
Color MandelbrotEngine::calculatePixelColor(double real, double imag) const {
    double z_real = 0.0;
    double z_imag = 0.0;
    int iterations = 0;

    while (z_real * z_real + z_imag * z_imag < 4.0 && iterations < maxIterations) {
        double temp_real = z_real * z_real - z_imag * z_imag + real;
        z_imag = 2.0 * z_real * z_imag + imag;
        z_real = temp_real;
        iterations++;
    }

    if (iterations == maxIterations) {
        return {0, 0, 0};
    }

    double t = static_cast<double>(iterations) / maxIterations;
    uint8_t r = static_cast<uint8_t>(9 * (1 - t) * t * t * t * 255);
    uint8_t g = static_cast<uint8_t>(15 * (1 - t) * (1 - t) * t * t * 255);
    uint8_t b = static_cast<uint8_t>(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);

    return {r, g, b};
}