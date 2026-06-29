// 算子: cuBLAS SGEMM, row-major C=A*B
// 面试考点: cuBLAS column-major 适配。row-major A*B 等价 column-major B^T * A^T
// 编译: nvcc -O3 -lineinfo -std=c++17 -I../include 02_cublas_sgemm.cu -lcublas -o 02_cublas_sgemm
// 运行: ./02_cublas_sgemm
#include "common.hpp"
int main(){const int M=512,N=512,K=512;thrust::host_vector<float>A(M*K),B(K*N),ref(M*N);fill_random(A);fill_random(B,-1,1);for(int m=0;m<M;++m)for(int n=0;n<N;++n){float s=0;for(int k=0;k<K;++k)s+=A[m*K+k]*B[k*N+n];ref[m*N+n]=s;}thrust::device_vector<float>dA=A,dB=B,dC(M*N);cublasHandle_t h;CUBLAS_CHECK(cublasCreate(&h));float alpha=1,beta=0;auto launch=[&]{CUBLAS_CHECK(cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,raw(dB),N,raw(dA),K,&beta,raw(dC),N));};launch();CUDA_CHECK(cudaDeviceSynchronize());thrust::host_vector<float>got=dC;double err=0;bool pass=check_close(ref,got,1e-2f,&err);float ms=time_cuda_ms(launch,3,20);CUBLAS_CHECK(cublasDestroy(h));print_result("cublas_sgemm","M=N=K=512 row-major API side",pass,err,ms,2.0*M*N*K/ms/1e6,"GFLOP/s");return pass?0:1;}
