# ===================================================================
# COMPILERS & TOOLCHAINS
# ===================================================================
CXX = g++
NVCC = nvcc

# ===================================================================
# GLOBAL FLAGS & LIBRARIES
# ===================================================================
CXXFLAGS_BASE = -std=c++11 -Wall -O3
NVCCFLAGS_BASE = -std=c++11 -O3 -arch=sm_61 # Adjust -arch for your GPU if needed

# --- CUDA Library Path (User may need to adjust this) ---
CUDA_LIB_DIR ?= /usr/local/cuda/lib64

# --- Library Configurations (via helper scripts) ---
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS = $(shell pkg-config --libs opencv4)
WX_CXXFLAGS = $(shell wx-config --cxxflags)
WX_LIBS = $(shell wx-config --libs)

# Create a version of wx-config flags without -pthread for NVCC compilation
WX_CXXFLAGS_FOR_NVCC = $(filter-out -pthread,$(WX_CXXFLAGS))
# Create a version of linker flags that replaces -pthread with -lpthread for the linker
WX_LIBS_FOR_NVCC = $(filter-out -pthread,$(WX_LIBS)) -lpthread

# ===================================================================
# DIRECTORIES & TARGETS
# ===================================================================
BIN_DIR = bin
CPU_SRC_DIR = src/cpu
GPU_SRC_DIR = src/gpu

# --- Main Benchmark CLI Target ---
MAIN_TARGET = $(BIN_DIR)/mandelbrot_benchmark
MAIN_SRCS_CPP = main.cpp $(CPU_SRC_DIR)/mandelbrot_generator.cpp
MAIN_SRCS_CU = $(GPU_SRC_DIR)/mandelbrot_gpu.cu
MAIN_OBJS = $(addprefix $(BIN_DIR)/, $(notdir $(MAIN_SRCS_CPP:.cpp=.o))) $(addprefix $(BIN_DIR)/, $(notdir $(MAIN_SRCS_CU:.cu=.o)))
MAIN_CXXFLAGS = $(CXXFLAGS_BASE) -fopenmp -I$(CPU_SRC_DIR) -I$(GPU_SRC_DIR) $(OPENCV_CFLAGS)
MAIN_NVCCFLAGS = $(NVCCFLAGS_BASE) -x cu -I$(CPU_SRC_DIR) -I$(GPU_SRC_DIR) $(OPENCV_CFLAGS) -Xcompiler "-fopenmp"
MAIN_LDFLAGS = -fopenmp $(OPENCV_LIBS) -L$(CUDA_LIB_DIR) -lcudart

# --- wxWidgets GUI Target ---
WX_TARGET = $(BIN_DIR)/mandelbrot_wx
WX_ENGINE_DIR = src/wx-engine
WX_SRCS = app_wx.cpp $(WX_ENGINE_DIR)/mandelbrot_engine.cpp
WX_OBJS = $(addprefix $(BIN_DIR)/, $(notdir $(WX_SRCS:.cpp=.o)))
WX_CXXFLAGS_FULL = $(CXXFLAGS_BASE) -fopenmp -Isrc/common -I$(WX_ENGINE_DIR) $(WX_CXXFLAGS)
WX_LDFLAGS_FULL = -fopenmp $(WX_LIBS)

# --- wxWidgets + GPU GUI Target ---
WX_GPU_TARGET = $(BIN_DIR)/mandelbrot_wx_gpu
WX_GPU_ENGINE_DIR = src/wx-gpu-engine
WX_GPU_SRCS_CPP = app_wx_gpu.cpp
WX_GPU_SRCS_CU = $(WX_GPU_ENGINE_DIR)/mandelbrot_renderer_gpu.cu
WX_GPU_OBJS = $(addprefix $(BIN_DIR)/, $(notdir $(WX_GPU_SRCS_CPP:.cpp=.o) $(WX_GPU_SRCS_CU:.cu=.o)))
WX_GPU_CXXFLAGS_FULL = $(CXXFLAGS_BASE) -Isrc/common -I$(WX_GPU_ENGINE_DIR) -Isrc/wx-engine $(WX_CXXFLAGS)
WX_GPU_NVCCFLAGS_FULL = $(NVCCFLAGS_BASE) -Isrc/common -I$(WX_GPU_ENGINE_DIR) -Isrc/wx-engine $(WX_CXXFLAGS_FOR_NVCC)

# ===================================================================
# BUILD RULES
# ===================================================================
all: main wx wxgpu

main: $(MAIN_TARGET)
wx: $(WX_TARGET)
wxgpu: $(WX_GPU_TARGET)

# --- Linking Rules ---
$(MAIN_TARGET): $(MAIN_OBJS)
	@echo "Linking Main Benchmark target with g++..."
	$(CXX) $^ -o $@ $(MAIN_LDFLAGS)
	@echo "Main Benchmark build successful: $@"

$(WX_TARGET): $(filter $(BIN_DIR)/app_wx.o $(BIN_DIR)/mandelbrot_engine.o, $(WX_OBJS))
	@echo "Linking wxWidgets target..."
	$(CXX) $^ -o $@ $(WX_LDFLAGS_FULL)
	@echo "wxWidgets build successful: $@"

$(WX_GPU_TARGET): $(WX_GPU_OBJS) $(BIN_DIR)/mandelbrot_engine.o
	@echo "Linking wxWidgets+GPU target..."
	$(NVCC) $^ -o $@ $(WX_LIBS_FOR_NVCC)
	@echo "wxWidgets+GPU build successful: $@"

# --- Compilation Rules ---
$(BIN_DIR)/main.o: main.cpp
	@mkdir -p $(BIN_DIR)
	@echo "Compiling main benchmark driver with NVCC: $<"
	$(NVCC) $(MAIN_NVCCFLAGS) -c $< -o $@

$(BIN_DIR)/mandelbrot_generator.o: $(CPU_SRC_DIR)/mandelbrot_generator.cpp
	@mkdir -p $(BIN_DIR)
	@echo "Compiling CPU source: $<"
	$(CXX) $(MAIN_CXXFLAGS) -c $< -o $@

$(BIN_DIR)/mandelbrot_gpu.o: $(GPU_SRC_DIR)/mandelbrot_gpu.cu
	@mkdir -p $(BIN_DIR)
	@echo "Compiling CUDA source: $<"
	$(NVCC) $(MAIN_NVCCFLAGS) -c $< -o $@

$(BIN_DIR)/mandelbrot_engine.o: $(WX_ENGINE_DIR)/mandelbrot_engine.cpp
	@mkdir -p $(BIN_DIR)
	@echo "Compiling wxWidgets engine: $<"
	$(CXX) $(WX_CXXFLAGS_FULL) -c $< -o $@

$(BIN_DIR)/mandelbrot_renderer_gpu.o: $(WX_GPU_ENGINE_DIR)/mandelbrot_renderer_gpu.cu
	@mkdir -p $(BIN_DIR)
	@echo "Compiling CUDA wx engine: $<"
	$(NVCC) $(WX_GPU_NVCCFLAGS_FULL) -c $< -o $@

# Rules for wxWidgets driver files
$(BIN_DIR)/app_wx.o: app_wx.cpp
	@mkdir -p $(BIN_DIR)
	@echo "Compiling wxWidgets driver: $<"
	$(CXX) $(WX_CXXFLAGS_FULL) -c $< -o $@

$(BIN_DIR)/app_wx_gpu.o: app_wx_gpu.cpp
	@mkdir -p $(BIN_DIR)
	@echo "Compiling wxWidgets+GPU driver: $<"
	$(CXX) $(WX_GPU_CXXFLAGS_FULL) -c $< -o $@


# ===================================================================
# PHONY TARGETS (Commands that are not files)
# ===================================================================
.PHONY: all main wx wxgpu build_main build_wx build_wxgpu run_main run_wx run_wxgpu clean

# Build Aliases
build_main: main
build_wx: wx
build_wxgpu: wxgpu

# Run Targets
run_main: main
	./$(MAIN_TARGET)

run_wx: wx
	./$(WX_TARGET)

run_wxgpu: wxgpu
	./$(WX_GPU_TARGET)

# Cleanup
clean:
	@echo "Cleaning up all build artifacts..."
	rm -rf $(BIN_DIR)
	@echo "Cleanup complete."
