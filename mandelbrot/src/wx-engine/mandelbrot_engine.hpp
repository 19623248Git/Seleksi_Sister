#ifndef MANDELBROT_ENGINE_H
#define MANDELBROT_ENGINE_H

#include <cstdint>
#include <vector>   // <-- ADDED: For std::vector
#include <thread>   // <-- ADDED: For std::thread

#include "../common/common_types.h"

class MandelbrotEngine {
public:
    MandelbrotEngine();

    void setMaxIterations(int iterations);

    int getMaxIterations() const;

    // This function remains for single-pixel calculation if needed
    Color calculatePixelColor(double real, double imag) const;

    /**
     * @brief Generates the full Mandelbrot set using multiple threads.
     * @param pixels A reference to the output pixel buffer.
     * @param width The width of the image.
     * @param height The height of the image.
     * @param minReal The minimum value on the real axis.
     * @param maxReal The maximum value on the real axis.
     * @param minImag The minimum value on the imaginary axis.
     * @param maxImag The maximum value on the imaginary axis.
    */
    void generateMandelbrot(
        std::vector<Color>& pixels, int width, int height,
        double minReal, double maxReal, double minImag, double maxImag
    ) const; // <-- ADDED: The new multithreaded method

private:
    int maxIterations;
};

#endif // MANDELBROT_ENGINE_H