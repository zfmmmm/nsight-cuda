NVCC ?= nvcc
ARCH ?= sm_120
CXXFLAGS ?= -O3 -lineinfo -std=c++17
INCLUDES := -I../../nsight_common

all: memcpy_bottleneck

memcpy_bottleneck: memcpy_bottleneck.cu
	$(NVCC) $(CXXFLAGS) -arch=$(ARCH) $(INCLUDES) $< -o $@

clean:
	rm -f memcpy_bottleneck *.nsys-rep *.sqlite *.qdrep

.PHONY: all clean
