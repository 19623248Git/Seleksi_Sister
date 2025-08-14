#include "mandelbrot_generator.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <omp.h>


MandelbrotGenerator::MandelbrotGenerator() {
    config.img_width = 1920;
    config.img_height = 1080;
    config.max_iterations = 1000;
    config.real_min = -2.0;
    config.real_max = 1.0;
    config.imag_min = -1.5;
    config.imag_max = 1.5;
    config.output_filename = "mandelbrot.png";
    config.use_multithreading = true;
}


void MandelbrotGenerator::getUserConfiguration() {
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


cv::Vec3b MandelbrotGenerator::calculatePixel(int x, int y) {
    double real_c = config.real_min + (double)x / (config.img_width - 1) * (config.real_max - config.real_min);
    double imag_c = config.imag_min + (double)y / (config.img_height - 1) * (config.imag_max - config.imag_min);

    double real_z = 0.0, imag_z = 0.0;
    int n = 0;
    while (real_z * real_z + imag_z * imag_z <= 4.0 && n < config.max_iterations) {
        double temp_real_z = real_z * real_z - imag_z * imag_z + real_c;
        imag_z = 2.0 * real_z * imag_z + imag_c;
        real_z = temp_real_z;
        n++;
    }

    if (n == config.max_iterations) {
        return cv::Vec3b(0, 0, 0);
    } else {
        uchar r = (n % 256);
        uchar g = (n * 5 % 256);
        uchar b = (n * 10 % 256);
        return cv::Vec3b(b, g, r);
    }
}


void MandelbrotGenerator::generate() {
    mandelbrotImage = cv::Mat(config.img_height, config.img_width, CV_8UC3);
    
    #pragma omp parallel for schedule(dynamic) if(config.use_multithreading)
    for (int y = 0; y < config.img_height; ++y) {
        for (int x = 0; x < config.img_width; ++x) {
            mandelbrotImage.at<cv::Vec3b>(y, x) = calculatePixel(x, y);
        }
    }
}


void MandelbrotGenerator::saveImage(const std::string& filename) {
    if (!cv::imwrite(filename, mandelbrotImage)) {
        std::cerr << "Error: Failed to save image " << filename << std::endl;
    } else {
        std::cout << "Image successfully saved as " << filename << std::endl;
    }
}


void MandelbrotGenerator::run() {
    int choice;
    std::cout << "Choose the options below:\n";
    std::cout << "1. Serial (Single-Thread)\n";
    std::cout << "2. Parallel (Multi-Thread)\n";
    std::cout << "3. Benchmark (Serial vs Parallel)\n";
    std::cout << "Your Choice: ";
    std::cin >> choice;

    if (choice < 1 || choice > 3) {
        std::cerr << "Invalid choice, exiting..." << std::endl;
        return;
    }

    getUserConfiguration();

    switch (choice) {
        case 1: {
            std::cout << "\n--- Running Serial Mode ---" << std::endl;
            config.use_multithreading = false;
            auto start = std::chrono::high_resolution_clock::now();
            generate();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            std::cout << "Serial Execution Done" << std::endl;
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "Time Execution: " << static_cast<double>(duration_ms.count()) << " ms" << std::endl;
            this->time_execution = static_cast<double>(duration_ms.count());
            saveImage("serial_" + config.output_filename);
            break;
        }
        case 2: {
            std::cout << "\n--- Running Parallel Mode ---" << std::endl;
            config.use_multithreading = true;
            auto start = std::chrono::high_resolution_clock::now();
            generate();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            std::cout << "Parallel Execution Done" << std::endl;
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "Time Execution: " << static_cast<double>(duration_ms.count()) << " ms" << std::endl;
            this->time_execution = static_cast<double>(duration_ms.count());
            saveImage("parallel_" + config.output_filename);
            break;
        }
        case 3: {
            std::cout << "\n--- Starting Benchmark ---" << std::endl;

            std::cout << "\n[1] Running Serial Version (single-thread)..." << std::endl;
            config.use_multithreading = false;
            auto start_serial = std::chrono::high_resolution_clock::now();
            generate();
            auto end_serial = std::chrono::high_resolution_clock::now();
            auto duration_ms_serial = std::chrono::duration_cast<std::chrono::milliseconds>(end_serial - start_serial);
            std::cout << "Serial Version Done" << std::endl;
            saveImage("serial_" + config.output_filename);

            std::cout << "\n[2] Running Parallel Version (multi-thread)..." << std::endl;
            config.use_multithreading = true;
            auto start_parallel = std::chrono::high_resolution_clock::now();
            generate();
            auto end_parallel = std::chrono::high_resolution_clock::now();
            auto duration_ms_parallel = std::chrono::duration_cast<std::chrono::milliseconds>(end_parallel - start_parallel);
            std::cout << "Parallel Version Done" << std::endl;
            saveImage("parallel_" + config.output_filename);

            double speedup = (duration_ms_serial.count() > 1.0E-6 && duration_ms_parallel.count() > 1.0E-6)
                             ? static_cast<double>(duration_ms_serial.count()) / static_cast<double>(duration_ms_parallel.count())
                             : 0.0;

            std::cout << "\n\n--- Benchmark Results ---" << std::endl;
            std::cout << "-------------------------------------------------" << std::endl;
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "Serial Time Execution:   " << duration_ms_serial.count() << " ms" << std::endl;
            std::cout << "Paralel Time Execution:  " << duration_ms_parallel.count() << " ms" << std::endl;
            std::cout << "-------------------------------------------------" << std::endl;
            std::cout << "Speedup Ratio: " << speedup << "x" << std::endl;
            std::cout << "-------------------------------------------------" << std::endl;
            break;
        }
    }
}
