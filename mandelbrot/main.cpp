#include <iostream>
#include <chrono>
#include <iomanip>
#include <memory>
#include <limits>

// Include the CUDA runtime header to recognize CUDA types and functions
#include <cuda_runtime.h>

// Include the headers for the CPU and GPU Mandelbrot implementations.
#include "src/cpu/mandelbrot_generator.hpp"
#include "src/gpu/mandelbrot_gpu.hpp"

// Forward declaration for the GPU kernel launch logic.
// This is defined in mandelbrot_gpu.cu
__global__ void mandelbrot_kernel(unsigned char* imageData, int width, int height, int max_iter);

/**
 * @brief A wrapper function to launch the CUDA kernel from this file.
 */
cudaError_t launch_mandelbrot_kernel_wrapper(unsigned char* deviceImage, int width, int height, int max_iter) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    mandelbrot_kernel<<<blocksPerGrid, threadsPerBlock>>>(deviceImage, width, height, max_iter);
    
    // Return the last error to check for launch issues
    return cudaGetLastError();
}


/**
 * @brief Prints the results of a two-way benchmark comparison with aligned columns.
 */
void print_benchmark_results(double time1_ms, double time2_ms, const std::string& label1, const std::string& label2) {
    // Calculate speedup, handling potential division by zero.
    double speedup = (time1_ms > 1.0E-9 && time2_ms > 1.0E-9)
                     ? time1_ms / time2_ms
                     : 0.0;
    
    const int label_width = 28; // Define a fixed width for the label column

    std::cout << "\n\n--- Benchmark Results: " << label1 << " vs. " << label2 << " ---" << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    
    // Use std::left and std::setw for proper alignment
    std::cout << std::left << std::setw(label_width) << (label1 + " Time Execution:") << time1_ms << " ms" << std::endl;
    std::cout << std::left << std::setw(label_width) << (label2 + " Time Execution:") << time2_ms << " ms" << std::endl;

    std::cout << "-------------------------------------------------" << std::endl;
    std::cout << "Speedup Ratio (" << label1 << "/" << label2 << "): " << speedup << "x" << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;
}

/**
 * @brief Prints the results of a full three-way benchmark comparison.
 */
void print_full_benchmark_results(double serial_ms, double parallel_ms, double gpu_ms) {
    double speedup_parallel_vs_serial = (serial_ms > 1.0E-9 && parallel_ms > 1.0E-9) ? serial_ms / parallel_ms : 0.0;
    double speedup_gpu_vs_serial = (serial_ms > 1.0E-9 && gpu_ms > 1.0E-9) ? serial_ms / gpu_ms : 0.0;
    // FIX: Added GPU vs Parallel speedup calculation
    double speedup_gpu_vs_parallel = (parallel_ms > 1.0E-9 && gpu_ms > 1.0E-9) ? parallel_ms / gpu_ms : 0.0;

    const int label_width = 28;

    std::cout << "\n\n--- Full Benchmark Results ---" << std::endl;
    std::cout << "=================================================" << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    std::cout << std::left << std::setw(label_width) << "Serial CPU Time:" << serial_ms << " ms" << std::endl;
    std::cout << std::left << std::setw(label_width) << "Parallel CPU Time:" << parallel_ms << " ms" << std::endl;
    std::cout << std::left << std::setw(label_width) << "GPU Time:" << gpu_ms << " ms" << std::endl;
    
    std::cout << "-------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(label_width) << "Speedup (Parallel vs Serial):" << speedup_parallel_vs_serial << "x" << std::endl;
    std::cout << std::left << std::setw(label_width) << "Speedup (GPU vs Serial):" << speedup_gpu_vs_serial << "x" << std::endl;
    std::cout << std::left << std::setw(label_width) << "Speedup (GPU vs Parallel):" << speedup_gpu_vs_parallel << "x" << std::endl;
    std::cout << "=================================================" << std::endl;
}


/**
 * @brief Displays the main menu of options to the user.
 */
void display_menu() {
    std::cout << "\n=============================================" << std::endl;
    std::cout << "   Mandelbrot Generator (CPU vs. GPU)    " << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "1. Generate (Serial CPU)" << std::endl;
    std::cout << "2. Generate (Parallel CPU)" << std::endl;
    std::cout << "3. Generate (GPU)" << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "4. Benchmark: Serial CPU vs. Parallel CPU" << std::endl;
    std::cout << "5. Benchmark: Serial CPU vs. GPU" << std::endl;
    std::cout << "6. Benchmark: Parallel CPU vs. GPU" << std::endl;
    std::cout << "7. Benchmark: All (Serial vs Parallel vs GPU)" << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "0. Exit" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "Enter your choice: ";
}

/**
 * @brief Pauses execution and waits for the user to press Enter.
 */
void pause_for_user() {
    std::cout << "\nPress Enter to continue...";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get();
}


int main() {
    int choice;
    do {
        display_menu();
        std::cin >> choice;

        if (std::cin.fail()) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            choice = -1;
        }

        switch (choice) {
            case 1: { // Serial CPU
                std::cout << "\n--- Running Serial Mode ---" << std::endl;
                MandelbrotGenerator cpu_gen;
                cpu_gen.getUserConfiguration();
                cpu_gen.config.use_multithreading = false;

                auto start = std::chrono::high_resolution_clock::now();
                cpu_gen.generate();
                auto end = std::chrono::high_resolution_clock::now();
                auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                double duration_ms = duration_us.count() / 1000.0;

                std::cout << "Serial Execution Done." << std::endl;
                std::cout << std::fixed << std::setprecision(4);
                std::cout << "Time Execution: " << duration_ms << " ms" << std::endl;
                cpu_gen.saveImage("serial_" + cpu_gen.config.output_filename);
                pause_for_user();
                break;
            }
            case 2: { // Parallel CPU
                std::cout << "\n--- Running Parallel Mode ---" << std::endl;
                MandelbrotGenerator cpu_gen;
                cpu_gen.getUserConfiguration();
                cpu_gen.config.use_multithreading = true;

                auto start = std::chrono::high_resolution_clock::now();
                cpu_gen.generate();
                auto end = std::chrono::high_resolution_clock::now();
                auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                double duration_ms = duration_us.count() / 1000.0;
                
                std::cout << "Parallel Execution Done." << std::endl;
                std::cout << std::fixed << std::setprecision(4);
                std::cout << "Time Execution: " << duration_ms << " ms" << std::endl;
                cpu_gen.saveImage("parallel_" + cpu_gen.config.output_filename);
                pause_for_user();
                break;
            }
            case 3: { // GPU
                std::cout << "\n--- Running GPU Mode ---" << std::endl;
                MandelbrotGPU gpu_gen;
                gpu_gen.run();
                pause_for_user();
                break;
            }
            case 4: { // Benchmark: Serial vs Parallel
                std::cout << "\n--- Benchmark: Serial vs. Parallel ---" << std::endl;
                MandelbrotGenerator cpu_gen;
                cpu_gen.getUserConfiguration();

                // Run Serial
                std::cout << "\n[1] Running Serial Version..." << std::endl;
                cpu_gen.config.use_multithreading = false;
                auto start_serial = std::chrono::high_resolution_clock::now();
                cpu_gen.generate();
                auto end_serial = std::chrono::high_resolution_clock::now();
                auto duration_us_serial = std::chrono::duration_cast<std::chrono::microseconds>(end_serial - start_serial);
                double duration_ms_serial = duration_us_serial.count() / 1000.0;
                cpu_gen.saveImage("serial_" + cpu_gen.config.output_filename);

                // Run Parallel
                std::cout << "\n[2] Running Parallel Version..." << std::endl;
                cpu_gen.config.use_multithreading = true;
                auto start_parallel = std::chrono::high_resolution_clock::now();
                cpu_gen.generate();
                auto end_parallel = std::chrono::high_resolution_clock::now();
                auto duration_us_parallel = std::chrono::duration_cast<std::chrono::microseconds>(end_parallel - start_parallel);
                double duration_ms_parallel = duration_us_parallel.count() / 1000.0;
                cpu_gen.saveImage("parallel_" + cpu_gen.config.output_filename);

                print_benchmark_results(duration_ms_serial, duration_ms_parallel, "Serial", "Parallel");
                pause_for_user();
                break;
            }
            case 5:
            case 6: {
                std::string cpu_mode_label = (choice == 5) ? "Serial" : "Parallel";
                bool use_multithreading = (choice == 6);

                std::cout << "\n--- Benchmark: " << cpu_mode_label << " vs. GPU ---" << std::endl;
                
                MandelbrotGenerator base_config_gen;
                base_config_gen.getUserConfiguration();
                auto config = base_config_gen.config;

                // --- Run CPU Version ---
                std::cout << "\n[1] Running " << cpu_mode_label << " Version..." << std::endl;
                MandelbrotGenerator cpu_gen;
                cpu_gen.config = config;
                cpu_gen.config.use_multithreading = use_multithreading;
                
                auto start_cpu = std::chrono::high_resolution_clock::now();
                cpu_gen.generate();
                auto end_cpu = std::chrono::high_resolution_clock::now();
                auto duration_us_cpu = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);
                double duration_ms_cpu = duration_us_cpu.count() / 1000.0;
                cpu_gen.saveImage(cpu_mode_label + "_" + config.output_filename);

                // --- Run GPU Version ---
                std::cout << "\n[2] Running GPU Version..." << std::endl;
                double duration_ms_gpu = 0;
                
                cv::Mat host_image(config.img_height, config.img_width, CV_8UC3);
                unsigned char* device_image = nullptr;
                size_t data_size = (size_t)config.img_width * config.img_height * 3 * sizeof(unsigned char);

                cudaError_t err = cudaMalloc(&device_image, data_size);
                if (err != cudaSuccess) {
                    std::cerr << "CUDA Malloc failed: " << cudaGetErrorString(err) << std::endl;
                } else {
                    auto start_gpu = std::chrono::high_resolution_clock::now();
                    launch_mandelbrot_kernel_wrapper(device_image, config.img_width, config.img_height, config.max_iterations);
                    cudaDeviceSynchronize();
                    auto end_gpu = std::chrono::high_resolution_clock::now();
                    auto duration_us_gpu = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu);
                    duration_ms_gpu = duration_us_gpu.count() / 1000.0;
                    
                    cudaMemcpy(host_image.data, device_image, data_size, cudaMemcpyDeviceToHost);
                    cv::imwrite("gpu_" + config.output_filename, host_image);
                    std::cout << "Image saved as gpu_" + config.output_filename << std::endl;
                    cudaFree(device_image);
                }

                print_benchmark_results(duration_ms_cpu, duration_ms_gpu, cpu_mode_label, "GPU");
                pause_for_user();
                break;
            }
            case 7: { // Benchmark: All
                std::cout << "\n--- Full Benchmark: Serial vs. Parallel vs. GPU ---" << std::endl;
                MandelbrotGenerator base_config_gen;
                base_config_gen.getUserConfiguration();
                auto config = base_config_gen.config;

                // --- [1] Run Serial Version ---
                std::cout << "\n[1/3] Running Serial Version..." << std::endl;
                MandelbrotGenerator serial_gen;
                serial_gen.config = config;
                serial_gen.config.use_multithreading = false;
                auto start_serial = std::chrono::high_resolution_clock::now();
                serial_gen.generate();
                auto end_serial = std::chrono::high_resolution_clock::now();
                double duration_ms_serial = std::chrono::duration_cast<std::chrono::microseconds>(end_serial - start_serial).count() / 1000.0;
                serial_gen.saveImage("serial_" + config.output_filename);

                // --- [2] Run Parallel Version ---
                std::cout << "\n[2/3] Running Parallel Version..." << std::endl;
                MandelbrotGenerator parallel_gen;
                parallel_gen.config = config;
                parallel_gen.config.use_multithreading = true;
                auto start_parallel = std::chrono::high_resolution_clock::now();
                parallel_gen.generate();
                auto end_parallel = std::chrono::high_resolution_clock::now();
                double duration_ms_parallel = std::chrono::duration_cast<std::chrono::microseconds>(end_parallel - start_parallel).count() / 1000.0;
                parallel_gen.saveImage("parallel_" + config.output_filename);

                // --- [3] Run GPU Version ---
                std::cout << "\n[3/3] Running GPU Version..." << std::endl;
                double duration_ms_gpu = 0;
                cv::Mat host_image(config.img_height, config.img_width, CV_8UC3);
                unsigned char* device_image = nullptr;
                size_t data_size = (size_t)config.img_width * config.img_height * 3 * sizeof(unsigned char);
                cudaError_t err = cudaMalloc(&device_image, data_size);
                if (err != cudaSuccess) {
                    std::cerr << "CUDA Malloc failed: " << cudaGetErrorString(err) << std::endl;
                } else {
                    auto start_gpu = std::chrono::high_resolution_clock::now();
                    launch_mandelbrot_kernel_wrapper(device_image, config.img_width, config.img_height, config.max_iterations);
                    cudaDeviceSynchronize();
                    auto end_gpu = std::chrono::high_resolution_clock::now();
                    duration_ms_gpu = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count() / 1000.0;
                    cudaMemcpy(host_image.data, device_image, data_size, cudaMemcpyDeviceToHost);
                    cv::imwrite("gpu_" + config.output_filename, host_image);
                    std::cout << "Image saved as gpu_" + config.output_filename << std::endl;
                    cudaFree(device_image);
                }

                print_full_benchmark_results(duration_ms_serial, duration_ms_parallel, duration_ms_gpu);
                pause_for_user();
                break;
            }
            case 0:
                std::cout << "Exiting program." << std::endl;
                break;
            default:
                std::cerr << "Invalid choice. Please try again." << std::endl;
                break;
        }
    } while (choice != 0);

    return 0;
}
