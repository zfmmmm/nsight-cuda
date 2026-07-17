// 算子: Embedding Gather
// 面试考点: token id -> embedding row，典型访存型 kernel
// 编译: nvcc -O3 -lineinfo -std=c++17 -I../include 19_embedding_gather.cu -o 19_embedding_gather
// 运行: ./19_embedding_gather
#include "common.hpp"
__global__ void embedding_kernel(
    const float* table, const int* ids, float* out, int tokens, int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x, total = tokens * dim;
    if (idx < total) {
        int t = idx / dim, d = idx % dim;
        out[idx] = table[ids[t] * dim + d];
    }
}
int main() {
    const int vocab = 32768, tokens = 8192, dim = 256;
    thrust::host_vector<float> tab(vocab * dim), ref(tokens * dim);
    thrust::host_vector<int> ids(tokens);
    fill_random(tab);
    fill_random_int(ids, 0, vocab - 1);
    for (int t = 0; t < tokens; ++t)
        for (int d = 0; d < dim; ++d) ref[t * dim + d] = tab[ids[t] * dim + d];
    thrust::device_vector<float> dt = tab, out(tokens * dim);
    thrust::device_vector<int> di = ids;
    int th = 256, bl = (tokens * dim + th - 1) / th;
    float ms = time_cuda_ms([&] {
        embedding_kernel<<<bl, th>>>(
            thrust::raw_pointer_cast(dt.data()),
            thrust::raw_pointer_cast(di.data()),
            thrust::raw_pointer_cast(out.data()),
            tokens,
            dim
        );
    });
    thrust::host_vector<float> got = out;
    double err = 0;
    bool pass = check_close(ref, got, 1e-6f, &err);
    print_result(
        "embedding_gather",
        "tokens=8192,dim=256",
        pass,
        err,
        ms,
        tokens * dim * sizeof(float) * 2.0 / ms / 1e6,
        "GB/s"
    );
    return pass ? 0 : 1;
}
