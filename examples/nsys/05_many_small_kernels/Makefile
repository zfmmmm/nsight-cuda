NVCC ?= nvcc
ARCH ?= sm_120
CXXFLAGS ?= -O3 -lineinfo -std=c++17
INCLUDES := -I../../nsight_common

all: many_small_kernels

many_small_kernels: many_small_kernels.cu
	$(NVCC) $(CXXFLAGS) -arch=$(ARCH) $(INCLUDES) $< -o $@

clean:
	rm -f many_small_kernels *.nsys-rep *.sqlite *.qdrep

.PHONY: all clean
