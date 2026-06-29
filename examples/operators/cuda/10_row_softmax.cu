// 算子: Row-wise Softmax
// 面试考点: 每 block 一行，先 reduce max，再 reduce sum，数值稳定
// 编译: nvcc -O3 -lineinfo -std=c++17 -I../include 10_row_softmax.cu -o 10_row_softmax
// 运行: ./10_row_softmax
#include "common.hpp"

__inline__ __device__ float warp_reduce_sum(float v){for(int o=16;o>0;o>>=1)v+=__shfl_down_sync(0xffffffff,v,o);return v;}
__inline__ __device__ float warp_reduce_max(float v){for(int o=16;o>0;o>>=1)v=fmaxf(v,__shfl_down_sync(0xffffffff,v,o));return v;}
__global__ void row_softmax_kernel(const float *x, float *y, int rows, int cols) {
  int row=blockIdx.x, tid=threadIdx.x; extern __shared__ float sm[];
  float m=-3.402823466e38f;
  for(int c=tid;c<cols;c+=blockDim.x)m=fmaxf(m,x[row*cols+c]);
  m=warp_reduce_max(m); if((tid&31)==0) sm[tid>>5]=m; __syncthreads();
  m=(tid<blockDim.x/32)?sm[tid]:-3.402823466e38f; if(tid<32)m=warp_reduce_max(m); if(tid==0)sm[0]=m; __syncthreads(); m=sm[0];
  float s=0; for(int c=tid;c<cols;c+=blockDim.x){float e=expf(x[row*cols+c]-m); y[row*cols+c]=e; s+=e;}
  s=warp_reduce_sum(s); if((tid&31)==0) sm[tid>>5]=s; __syncthreads();
  s=(tid<blockDim.x/32)?sm[tid]:0; if(tid<32)s=warp_reduce_sum(s); if(tid==0)sm[0]=s; __syncthreads(); s=sm[0];
  for(int c=tid;c<cols;c+=blockDim.x)y[row*cols+c]/=s;
}
int main(){
  const int rows=4096,cols=1024,n=rows*cols; thrust::host_vector<float> h(n),ref(n); fill_random(h,-4,4);
  for(int r=0;r<rows;++r){float m=-1e30f;for(int c=0;c<cols;++c)m=std::max(m,h[r*cols+c]);float s=0;for(int c=0;c<cols;++c){ref[r*cols+c]=std::exp(h[r*cols+c]-m);s+=ref[r*cols+c];}for(int c=0;c<cols;++c)ref[r*cols+c]/=s;}
  thrust::device_vector<float>d=h,out(n); float ms=time_cuda_ms([&]{row_softmax_kernel<<<rows,256,32*sizeof(float)>>>(raw(d),raw(out),rows,cols);},3,20);
  thrust::host_vector<float>got=out;double err=0;bool pass=check_close(ref,got,1e-5f,&err);
  print_result("row_softmax","rows=4096,cols=1024",pass,err,ms, n*sizeof(float)*2.0/ms/1e6,"GB/s"); return pass?0:1;
}
