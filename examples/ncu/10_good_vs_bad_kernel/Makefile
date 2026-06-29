NVCC ?= nvcc
ARCH ?= sm_120
CXXFLAGS ?= -O3 -lineinfo -std=c++17
INCLUDES := -I../../nsight_common

all: good_vs_bad_kernel

good_vs_bad_kernel: good_vs_bad_kernel.cu
	$(NVCC) $(CXXFLAGS) -arch=$(ARCH) $(INCLUDES) $< -o $@

clean:
	rm -f good_vs_bad_kernel *.ncu-rep

.PHONY: all clean
