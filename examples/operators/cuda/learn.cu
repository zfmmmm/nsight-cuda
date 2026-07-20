#include "common.hpp"
__inline__ __device__ float warp_sum(float v){
    //使用shuffle进行warp内原子规约
    for(int off=16;off>0;off>>=1){
        v+=__shfl_down_sync(0xffffffff,v,off);
    }
    return v;
}
//相当于一个block内的规约，partial存的是所有的block内规约的最终答案
__global__ void reduce_sum_kernel(float* x,float* partial,int n){
    float sum=0;
    //正常和
    for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
        sum+=x[i];
    }
    //以warp为单位进行规约
    sum=warp_sum(sum);
    __shared__ float warp_part[32];
    //这个线程在一个warp内的编号
    int in_warp=threadIdx.x&31;
    //这个线程属于block内的哪个warp
    int in_block=threadIdx.x>>5;
    // 如果是warp内的第一个线程，将规约的答案写入shared mem
    if(in_warp==0){
        warp_part[in_block]=sum;
    }
    __syncthreads();
    //warp规约完成，现在将32个warp的规约答案进行规约，实现block内的完整规约
    sum=(threadIdx.x<blockDim.x/32)?warp_part[in_warp]:0;//只选取第一个warp参与运算
    // 如果是第一个warp，进行规约
    if(in_block==0){
        sum=warp_sum(sum);
    }
    //规约后第一个线程是答案，负责写回
    if(threadIdx.x==0){
        partial[blockIdx.x]=sum;
    }
}

int main(){
    int n=1<<24;
    int thread=256,blocks=1024;
    thrust::host_vector<float> h(n);
    thrust::device_vector<float> d=h,partial(blocks),out(1);
    auto launch=[&]{
        reduce_sum_kernel<<<blocks,thread>>>(
            thrust::raw_pointer_cast(d.data()),
            thrust::raw_pointer_cast(partial.data()),
            n
        );
        reduce_sum_kernel<<<1,thread>>>(
            thrust::raw_pointer_cast(partial.data()),
            thrust::raw_pointer_cast(out.data()),
            blocks
        );
    };
    launch();
    thrust::host_vector<float> output=out;
}