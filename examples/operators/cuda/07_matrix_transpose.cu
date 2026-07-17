// 算子: Matrix Transpose
// 面试考点: naive vs tiled shared memory transpose, tile[32][33] 避免 bank conflict
// 编译: nvcc -O3 -lineinfo -std=c++17 -I../include 07_matrix_transpose.cu -o 07_matrix_transpose
// 运行: ./07_matrix_transpose
#include "common.hpp"

__global__ void transpose_naive(const float* in, float* out, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y < rows && x < cols) out[x * rows + y] = in[y * cols + x];
}
__global__ void transpose_tiled(const float* in, float* out, int rows, int cols) {
    __shared__ float tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x, y = blockIdx.y * 32 + threadIdx.y;
    if (y < rows && x < cols) tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
    __syncthreads();
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    if (y < cols && x < rows) out[y * rows + x] = tile[threadIdx.x][threadIdx.y];
}
int main() {
    const int rows = 2048, cols = 2048, n = rows * cols;
    thrust::host_vector<float> h(n), ref(n);
    fill_random(h);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) ref[c * rows + r] = h[r * cols + c];
    thrust::device_vector<float> d = h, out1(n), out2(n);
    dim3 block(32, 32), grid((cols + 31) / 32, (rows + 31) / 32);
    float ms1 = time_cuda_ms([&] {
        transpose_naive<<<grid, block>>>(
            thrust::raw_pointer_cast(d.data()), thrust::raw_pointer_cast(out1.data()), rows, cols
        );
    });
    float ms2 = time_cuda_ms([&] {
        transpose_tiled<<<grid, block>>>(
            thrust::raw_pointer_cast(d.data()), thrust::raw_pointer_cast(out2.data()), rows, cols
        );
    });
    thrust::host_vector<float> got = out2;
    double err = 0;
    bool pass = check_close(ref, got, 1e-6f, &err);
    print_result(
        "matrix_transpose_tiled",
        "2048x2048",
        pass,
        err,
        ms2,
        2.0 * n * sizeof(float) / ms2 / 1e6,
        "GB/s"
    );
    std::printf("naive time: %.4f ms, tiled time: %.4f ms\n", ms1, ms2);
    return pass ? 0 : 1;
}
