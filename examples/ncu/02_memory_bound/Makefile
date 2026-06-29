NVCC ?= nvcc
ARCH ?= sm_120
CXXFLAGS ?= -O3 -lineinfo -std=c++17
INCLUDES := -I../../nsight_common

all: memory_bound

memory_bound: memory_bound.cu
	$(NVCC) $(CXXFLAGS) -arch=$(ARCH) $(INCLUDES) $< -o $@

clean:
	rm -f memory_bound *.ncu-rep

.PHONY: all clean
