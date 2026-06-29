// 算子: FP32 <-> INT8 Quant/Dequant
// 面试考点: per-tensor scale、round/clamp、量化误差
// 编译: nvcc -O3 -lineinfo -std=c++17 -I../include 18_quant_dequant_int8.cu -o 18_quant_dequant_int8
// 运行: ./18_quant_dequant_int8
#include "common.hpp"
__global__ void quant_kernel(const float*x,signed char*q,float scale,int n){for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){int v=lrintf(x[i]/scale);v=max(-128,min(127,v));q[i]=(signed char)v;}}
__global__ void dequant_kernel(const signed char*q,float*y,float scale,int n){for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x)y[i]=float(q[i])*scale;}
int main(){const int n=1<<24;const float scale=0.02f;thrust::host_vector<float>h(n),ref(n);fill_random(h,-2,2);for(int i=0;i<n;++i){int v=std::max(-128,std::min(127,(int)std::lrint(h[i]/scale)));ref[i]=v*scale;}thrust::device_vector<float>d=h,out(n);thrust::device_vector<signed char>q(n);auto qp=thrust::raw_pointer_cast(q.data());int th=256,bl=4096;float ms=time_cuda_ms([&]{quant_kernel<<<bl,th>>>(raw(d),qp,scale,n);dequant_kernel<<<bl,th>>>(qp,raw(out),scale,n);});thrust::host_vector<float>got=out;double err=0;bool pass=check_close(ref,got,1e-6f,&err);print_result("quant_dequant_int8","n=16777216,scale=0.02",pass,err,ms,(n*(sizeof(float)+1+sizeof(float)))/ms/1e6,"GB/s");return pass?0:1;}
