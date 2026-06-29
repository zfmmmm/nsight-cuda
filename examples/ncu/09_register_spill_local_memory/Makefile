NVCC ?= nvcc
ARCH ?= sm_120
CXXFLAGS ?= -O3 -lineinfo -std=c++17
INCLUDES := -I../../nsight_common

all: register_spill_local_memory

register_spill_local_memory: register_spill_local_memory.cu
	$(NVCC) $(CXXFLAGS) -Xptxas --maxrregcount=32 -arch=$(ARCH) $(INCLUDES) $< -o $@

clean:
	rm -f register_spill_local_memory *.ncu-rep

.PHONY: all clean
