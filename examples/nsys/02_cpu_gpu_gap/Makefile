NVCC ?= nvcc
ARCH ?= sm_120
CXXFLAGS ?= -O3 -lineinfo -std=c++17
INCLUDES := -I../../nsight_common

all: cpu_gpu_gap

cpu_gpu_gap: cpu_gpu_gap.cu
	$(NVCC) $(CXXFLAGS) -arch=$(ARCH) $(INCLUDES) $< -o $@

clean:
	rm -f cpu_gpu_gap *.nsys-rep *.sqlite *.qdrep

.PHONY: all clean
