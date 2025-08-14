#ifndef MANDELBROT_GPU_H
#define MANDELBROT_GPU_H

#include <string>


namespace cv {
    class Mat;
}

class MandelbrotGPU {
public:

    struct Configuration {
        int img_width, img_height, max_iterations;
        std::string output_filename;
    } config;

    MandelbrotGPU();

    ~MandelbrotGPU();

    void run();

    cv::Mat* hostImage;
    
    unsigned char* deviceImage;

    void getUserConfiguration();

    void generate();

    void saveImage(const std::string& filename);
};

#endif // MANDELBROT_GPU_H
