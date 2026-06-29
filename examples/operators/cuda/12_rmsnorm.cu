// 算子: Row-wise RMSNorm
// 面试考点: LLM 高频 norm，reduce sum(x^2)，不减 mean
// 编译: nvcc -O3 -lineinfo -std=c++17 -I../include 12_rmsnorm.cu -o 12_rmsnorm
// 运行: ./12_rmsnorm
#include "common.hpp"
__inline__ __device__ float wrsum(float v){for(int o=16;o>0;o>>=1)v+=__shfl_down_sync(0xffffffff,v,o);return v;}
__global__ void rmsnorm_kernel(const float*x,const float*w,float*y,int rows,int cols,float eps){int r=blockIdx.x,t=threadIdx.x;extern __shared__ float sm[];float ss=0;for(int c=t;c<cols;c+=blockDim.x){float v=x[r*cols+c];ss+=v*v;}ss=wrsum(ss);if((t&31)==0)sm[t>>5]=ss;__syncthreads();ss=(t<blockDim.x/32)?sm[t]:0;if(t<32)ss=wrsum(ss);if(t==0)sm[0]=rsqrtf(ss/cols+eps);__syncthreads();float inv=sm[0];for(int c=t;c<cols;c+=blockDim.x)y[r*cols+c]=w[c]*x[r*cols+c]*inv;}
int main(){const int rows=4096,cols=1024,n=rows*cols;thrust::host_vector<float>h(n),w(cols),ref(n);fill_random(h);fill_random(w,0.5,1.5);for(int r=0;r<rows;++r){double ss=0;for(int c=0;c<cols;++c){float v=h[r*cols+c];ss+=v*v;}float inv=1/std::sqrt(ss/cols+1e-6f);for(int c=0;c<cols;++c)ref[r*cols+c]=w[c]*h[r*cols+c]*inv;}thrust::device_vector<float>d=h,dw=w,out(n);float ms=time_cuda_ms([&]{rmsnorm_kernel<<<rows,256,32*sizeof(float)>>>(raw(d),raw(dw),raw(out),rows,cols,1e-6f);},3,20);thrust::host_vector<float>got=out;double err=0;bool pass=check_close(ref,got,1e-4f,&err);print_result("rmsnorm","rows=4096,cols=1024",pass,err,ms,n*sizeof(float)*2.0/ms/1e6,"GB/s");return pass?0:1;}
