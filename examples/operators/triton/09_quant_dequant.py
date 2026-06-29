"""Triton fp32/int8 quant/dequant. Run: python 09_quant_dequant.py"""
import torch, triton, triton.language as tl
@triton.jit
def qdq_kernel(x,y,n:tl.constexpr,scale:tl.constexpr,BLOCK:tl.constexpr):
    off=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); m=off<n
    q=tl.minimum(127,tl.maximum(-128,tl.inline_asm_elementwise("cvt.rni.s32.f32 $0;", "=r,f", [tl.load(x+off,mask=m)/scale], dtype=tl.int32, is_pure=True, pack=1)))
    tl.store(y+off,q.to(tl.float32)*scale,mask=m)
def main():
    n=1<<24; scale=0.02; x=torch.randn(n,device="cuda"); y=torch.empty_like(x)
    qdq_kernel[(triton.cdiv(n,1024),)](x,y,n,scale,BLOCK=1024)
    ref=torch.clamp(torch.round(x/scale),-128,127)*scale; err=(y-ref).abs().max().item()
    print(f"operator: triton_quant_dequant\nGPU kernel: {'PASS' if err < 1e-6 else 'FAIL'}\nmax error: {err}\n{'PASS' if err < 1e-6 else 'FAIL'}")
if __name__=="__main__": main()
