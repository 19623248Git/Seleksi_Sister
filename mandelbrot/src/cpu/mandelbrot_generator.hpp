#ifndef MANDELBROT_GENERATOR_H
#define MANDELBROT_GENERATOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class MandelbrotGenerator {
public:

    struct Configuration {
        int img_width, img_height, max_iterations;
        double real_min, real_max, imag_min, imag_max;
        bool use_multithreading;
        std::string output_filename;
    } config;

    MandelbrotGenerator();

    void run();

    double time_execution = 0;

    cv::Mat mandelbrotImage;

    void getUserConfiguration();

    void generate();

    cv::Vec3b calculatePixel(int i, int j);

    void saveImage(const std::string& filename);
};

#endif // MANDELBROT_GENERATOR_H
