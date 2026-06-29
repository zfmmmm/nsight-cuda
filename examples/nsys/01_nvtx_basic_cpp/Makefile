NVCC ?= nvcc
ARCH ?= sm_120
CXXFLAGS ?= -O3 -lineinfo -std=c++17
INCLUDES := -I../../nsight_common

all: nvtx_basic_cpp

nvtx_basic_cpp: nvtx_basic_cpp.cu
	$(NVCC) $(CXXFLAGS) -arch=$(ARCH) $(INCLUDES) $< -o $@

clean:
	rm -f nvtx_basic_cpp *.nsys-rep *.sqlite *.qdrep

.PHONY: all clean
