NVCC ?= nvcc
ARCH ?= sm_120
CXXFLAGS ?= -O3 -lineinfo -std=c++17
INCLUDES := -I../../nsight_common

all: register_pressure

register_pressure: register_pressure.cu
	$(NVCC) $(CXXFLAGS) -arch=$(ARCH) $(INCLUDES) $< -o $@

clean:
	rm -f register_pressure *.ncu-rep

.PHONY: all clean
