// 算子: ReLU / GELU / SiLU
// 面试考点: 常见 activation、tanh GELU 近似、grid-stride loop
// 编译: nvcc -O3 -lineinfo -std=c++17 -I../include 14_activation_relu_gelu_silu.cu -o 14_activation_relu_gelu_silu
// 运行: ./14_activation_relu_gelu_silu
#include "common.hpp"
__device__ float dgelu(float x){return 0.5f*x*(1+tanhf(0.79788456f*(x+0.044715f*x*x*x)));}
__global__ void act_kernel(const float*x,float*relu,float*gelu,float*silu,int n){for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){float v=x[i];relu[i]=fmaxf(v,0);gelu[i]=dgelu(v);silu[i]=v/(1+expf(-v));}}
int main(){const int n=1<<24;thrust::host_vector<float>h(n),r(n),g(n),s(n);fill_random(h,-5,5);for(int i=0;i<n;++i){float v=h[i];r[i]=std::max(v,0.0f);g[i]=0.5f*v*(1+std::tanh(0.79788456f*(v+0.044715f*v*v*v)));s[i]=v/(1+std::exp(-v));}thrust::device_vector<float>d=h,dr(n),dg(n),ds(n);int th=256,bl=4096;float ms=time_cuda_ms([&]{act_kernel<<<bl,th>>>(raw(d),raw(dr),raw(dg),raw(ds),n);});thrust::host_vector<float>gr=dr,gg=dg,gs=ds;double e1=max_abs_diff(r,gr),e2=max_abs_diff(g,gg),e3=max_abs_diff(s,gs);bool pass=e1<1e-6&&e2<1e-5&&e3<1e-6;print_result("relu_gelu_silu","n=16777216",pass,std::max(e1,std::max(e2,e3)),ms,4.0*n*sizeof(float)/ms/1e6,"GB/s");return pass?0:1;}
