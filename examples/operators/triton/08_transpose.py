"""Triton block transpose. Run: python 08_transpose.py"""
import torch, triton, triton.language as tl
@triton.jit
def trans_kernel(x,y,R:tl.constexpr,C:tl.constexpr,BLOCK:tl.constexpr):
    pr,pc=tl.program_id(0),tl.program_id(1); rr=pr*BLOCK+tl.arange(0,BLOCK); cc=pc*BLOCK+tl.arange(0,BLOCK)
    v=tl.load(x+rr[:,None]*C+cc[None,:],mask=(rr[:,None]<R)&(cc[None,:]<C))
    tl.store(y+cc[:,None]*R+rr[None,:],tl.trans(v),mask=(cc[:,None]<C)&(rr[None,:]<R))
def main():
    R=C=2048; x=torch.randn((R,C),device="cuda"); y=torch.empty((C,R),device="cuda")
    trans_kernel[(triton.cdiv(R,32),triton.cdiv(C,32))](x,y,R,C,BLOCK=32); err=(y-x.t()).abs().max().item()
    print(f"operator: triton_transpose\nGPU kernel: {'PASS' if err < 1e-6 else 'FAIL'}\nmax error: {err}\n{'PASS' if err < 1e-6 else 'FAIL'}")
if __name__=="__main__": main()
