NVCC ?= nvcc
ARCH ?= sm_120
CXXFLAGS ?= -O3 -lineinfo -std=c++17
INCLUDES := -I../../nsight_common

all: shared_memory_occupancy_limit

shared_memory_occupancy_limit: shared_memory_occupancy_limit.cu
	$(NVCC) $(CXXFLAGS) -arch=$(ARCH) $(INCLUDES) $< -o $@

clean:
	rm -f shared_memory_occupancy_limit *.ncu-rep

.PHONY: all clean
