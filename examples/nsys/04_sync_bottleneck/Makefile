NVCC ?= nvcc
ARCH ?= sm_120
CXXFLAGS ?= -O3 -lineinfo -std=c++17
INCLUDES := -I../../nsight_common

all: sync_bottleneck

sync_bottleneck: sync_bottleneck.cu
	$(NVCC) $(CXXFLAGS) -arch=$(ARCH) $(INCLUDES) $< -o $@

clean:
	rm -f sync_bottleneck *.nsys-rep *.sqlite *.qdrep

.PHONY: all clean
