"""Triton row-wise stable softmax. Run: python 05_row_softmax.py"""
import torch, triton, triton.language as tl
@triton.jit
def softmax_kernel(x,y,cols:tl.constexpr,BLOCK:tl.constexpr):
    r=tl.program_id(0); offs=tl.arange(0,BLOCK); m=offs<cols
    v=tl.load(x+r*cols+offs,mask=m,other=-float("inf")); v=v-tl.max(v,axis=0); e=tl.exp(v); o=e/tl.sum(e,axis=0)
    tl.store(y+r*cols+offs,o,mask=m)
def main():
    rows,cols=4096,1024; x=torch.randn((rows,cols),device="cuda"); y=torch.empty_like(x)
    softmax_kernel[(rows,)](x,y,cols,BLOCK=1024)
    err=(y-torch.softmax(x,dim=1)).abs().max().item(); print(f"operator: triton_row_softmax\nGPU kernel: {'PASS' if err < 1e-5 else 'FAIL'}\nmax error: {err}\n{'PASS' if err < 1e-5 else 'FAIL'}")
if __name__=="__main__": main()
