NVCC ?= nvcc
ARCH ?= sm_120
CXXFLAGS ?= -O3 -lineinfo -std=c++17
INCLUDES := -I../../nsight_common

all: shared_bank_conflict

shared_bank_conflict: shared_bank_conflict.cu
	$(NVCC) $(CXXFLAGS) -arch=$(ARCH) $(INCLUDES) $< -o $@

clean:
	rm -f shared_bank_conflict *.ncu-rep

.PHONY: all clean
