# Introduction

This repository consists of CLI and GUI app to generate the mandelbrot set. The purpose of this project is to learn parallelism using multithreading or GPU

## Parallelization Implementation

The concept of parallelization utilizes *embarassingly paralel*, whereas each thread's process doesn't have to wait for other threads to complete its task. Since the mandelbrot set image generation relies on `(a,b)` from a complex number, we can substitute `(a,b) -> (x,y)` of the output image's pixel coordinate, which can be used to generate the set, hence it is *embarassingly paralel*. 

### The CPU Parallel implementation utilizes OpenMP as follows:
```cpp
#pragma omp parallel for schedule(dynamic) if(config.use_multithreading)
    for (int y = 0; y < config.img_height; ++y) {
        for (int x = 0; x < config.img_width; ++x) {
            mandelbrotImage.at<cv::Vec3b>(y, x) = calculatePixel(x, y);
        }
    }
```
OpenMP first creates worker threads, one for each CPU cores. OpenMP then treats all the iterations of the for loop as a task, enqueued into a central queue of tasks in our case (`parallel for`). OpenMP then instructs that if any thread is available, it will grab the next task from the queue and execute it (`schedule(dynamic)`), which ensures all threads fairly.  

### The GPU Parallel implementation utilizes Cuda as follows:
```cpp
config.img_width = 1920
config.img_height = 1080

__global__  // This indicates global function or """kernel""" that can be accessed by your device
void mandelbrot_kernel(unsigned char* imageData, int width, int height, int max_iter);

dim3 threadsPerBlock(16, 16);

dim3 blocksPerGrid(

        (config.img_width + threadsPerBlock.x - 1) / threadsPerBlock.x,

        (config.img_height + threadsPerBlock.y - 1) / threadsPerBlock.y

);

 mandelbrot_kernel<<<blocksPerGrid, threadsPerBlock>>>(deviceImage, config.img_width, config.img_height, config.max_iterations); 
```

Each Grid or Kernel contains `N x M` Blocks, and within each Blocks contains `n x m` threads. The following sets `16 x 16` threads in each block or equivalent to 256 threads. With this information, there consists of `120 * 68` blocks in this kernel.

## Video Demonstration of GUI

#### The video can be found within this google drive <a href="https://drive.google.com/drive/folders/1CQa1M1413l83ldlgb3ZS5YKh6Pj1wi6C?usp=sharing">link</a>!

## Running the Program Locally

Apt Installs:
```bash
sudo apt update && sudo apt install build-essential libopencv-dev pkg-config
sudo apt install libwxgtk3.2-dev
# CUDA toolkit install but honestly I forgot how
```

Compile instructions:
```bash
make build_main         # main program
make build_wx           # GUI with multithreading
make build_wxgpu        # GUI with CUDA
```

run instructions:
```bash
make run_main           # main program
make run_wx             # GUI with multithreading
make run_wxgpu          # GUI with CUDA
```
## Benchmark Results

### Configuration: 

- Width: 3840
- Height: :2160
- Iterations: 5000

| Metric | Time (ms) |
| :--- | :--- |
| Serial CPU Time | 29901.0300 |
| Parallel CPU Time | 2181.0130 |
| GPU Time | 890.5130 |



| Comparison | Speedup |
| :--- | :--- |
| Parallel vs Serial | 13.7097x |
| GPU vs Serial | 33.5773x |
| GPU vs Parallel | 2.4492x |

### Serial Result:
![serial.png](/mandelbrot/img/serial.png)

### Parallel Result:
![parallel.png](/mandelbrot/img/parallel.png)

### GPU Result:
![gpu.png](/mandelbrot/img/gpu.png)
