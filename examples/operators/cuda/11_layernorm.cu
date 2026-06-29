// 算子: Row-wise LayerNorm
// 面试考点: 每 block 一行，reduce mean/variance，gamma/beta
// 编译: nvcc -O3 -lineinfo -std=c++17 -I../include 11_layernorm.cu -o 11_layernorm
// 运行: ./11_layernorm
#include "common.hpp"
__inline__ __device__ float wsum(float v){for(int o=16;o>0;o>>=1)v+=__shfl_down_sync(0xffffffff,v,o);return v;}
__global__ void layernorm_kernel(const float*x,const float*g,const float*b,float*y,int rows,int cols,float eps){
  int r=blockIdx.x,t=threadIdx.x;extern __shared__ float sm[];float s=0,ss=0;
  for(int c=t;c<cols;c+=blockDim.x){float v=x[r*cols+c];s+=v;ss+=v*v;}
  s=wsum(s);ss=wsum(ss);if((t&31)==0){sm[t>>5]=s;sm[32+(t>>5)]=ss;}__syncthreads();
  s=(t<blockDim.x/32)?sm[t]:0;ss=(t<blockDim.x/32)?sm[32+t]:0;if(t<32){s=wsum(s);ss=wsum(ss);}if(t==0){sm[0]=s/cols;sm[1]=ss/cols-sm[0]*sm[0];}__syncthreads();
  float mean=sm[0], inv=rsqrtf(sm[1]+eps);for(int c=t;c<cols;c+=blockDim.x)y[r*cols+c]=g[c]*(x[r*cols+c]-mean)*inv+b[c];
}
int main(){const int rows=2048,cols=1024,n=rows*cols;thrust::host_vector<float>h(n),g(cols),b(cols),ref(n);fill_random(h);fill_random(g,0.5,1.5);fill_random(b,-0.1,0.1);
for(int r=0;r<rows;++r){double s=0,ss=0;for(int c=0;c<cols;++c){float v=h[r*cols+c];s+=v;ss+=v*v;}float m=s/cols,var=ss/cols-m*m,inv=1/std::sqrt(var+1e-5f);for(int c=0;c<cols;++c)ref[r*cols+c]=g[c]*(h[r*cols+c]-m)*inv+b[c];}
thrust::device_vector<float>d=h,dg=g,db=b,out(n);float ms=time_cuda_ms([&]{layernorm_kernel<<<rows,256,64*sizeof(float)>>>(raw(d),raw(dg),raw(db),raw(out),rows,cols,1e-5f);},3,20);
thrust::host_vector<float>got=out;double err=0;bool pass=check_close(ref,got,1e-4f,&err);print_result("layernorm","rows=2048,cols=1024",pass,err,ms,n*sizeof(float)*2.0/ms/1e6,"GB/s");return pass?0:1;}
