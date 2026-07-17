#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <string>

#define CHECK_CUDA(call)                                                                         \
    do {                                                                                         \
        cudaError_t status = (call);                                                             \
        if (status != cudaSuccess) {                                                             \
            std::fprintf(                                                                        \
                stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status) \
            );                                                                                   \
            std::exit(EXIT_FAILURE);                                                             \
        }                                                                                        \
    } while (0)

inline void print_device_info() {
    int device = 0;
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    std::printf("device=%d name=%s sm=%d%d\n", device, prop.name, prop.major, prop.minor);
}

inline int parse_mode(int argc, char** argv, const char* good_name = "good") {
    if (argc > 1 && std::string(argv[1]) == good_name) {
        return 1;
    }
    return 0;
}
