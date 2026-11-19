"""Experimental two-pass backward: query-owned dQ/gate/RPB; key-owned dK/dV.
Reverse adjacency built per call; combined query and key passes compute all gradients.
"""
import torch,triton
import triton.language as tl
from .blocksize_original import _forward
from .blocksize_query_pass import _backward as _query_backward
@triton.jit
def _key_backward(Q,K,V,C,I,L,Bias,O,LSE,DO,ORDER,START,DK,DV,
 G:tl.constexpr,H:tl.constexpr,NR:tl.constexpr,ALL:tl.constexpr,W:tl.constexpr,T:tl.constexpr,BT:tl.constexpr):
 dest=tl.program_id(0)//triton.cdiv(T,64);tile=tl.program_id(0)%triton.cdiv(T,64);bh=tl.program_id(1)
 u=tile*64+tl.arange(0,BT);d=tl.arange(0,64);qt=tl.arange(0,32)
 kv=((bh//H*G+dest)*T+u[:,None])*H*64+bh%H*64+d[None,:]
 kval=tl.load(K+kv,u[:,None]<T,other=0.).to(tl.float32)
 residual=tl.load(V+kv,u[:,None]<T,other=0.)-tl.load(C+(bh*G+dest)*64+d)[None,:]
 dk=tl.full((BT,64),0.,tl.float32);dv=tl.full((BT,64),0.,tl.float32)
 start=tl.load(START+bh*G+dest);stop=tl.load(START+bh*G+dest+1)
 for edge in range(start,stop):
  edgeid=tl.load(ORDER+edge);route=edgeid;src=(edgeid//NR)%G;gate=tl.load(L+route)
  for sub in range(triton.cdiv(T,32)):
   t=sub*32+qt
   off=((bh//H*G+src)*T+t[:,None])*H*64+bh%H*64+d[None,:]
   q=tl.load(Q+off,t[:,None]<T,other=0.).to(tl.float32);do=tl.load(DO+off,t[:,None]<T,other=0.)
   delta=tl.sum(do*tl.load(O+off,t[:,None]<T,other=0.),1);lse=tl.load(LSE+(bh*G+src)*T+t,t<T,other=0.)
   ri=(t[:,None]//W-u[None,:]//W+W-1)*(2*W-1)+t[:,None]%W-u[None,:]%W+W-1
   bias=tl.load(Bias+bh%H*((2*W-1)**2)+ri,(t[:,None]<T)&(u[None,:]<T),other=0.)
   score=tl.dot(q,tl.trans(kval),input_precision='tf32x3')*.125
   if ALL:score+=bias
   else:score+=tl.where(src==dest,bias,0.)
   score+=gate;score=tl.where(u[None,:]<T,score,-float('inf'))
   p=tl.exp(score-lse[:,None]);p=tl.where(t[:,None]<T,p,0.)
   dp=tl.dot(do,tl.trans(residual),input_precision='tf32x3');ds=p*(dp-delta[:,None])
   dk+=tl.dot(tl.trans(ds),q,input_precision='tf32x3')*.125
   dv+=tl.dot(tl.trans(p),do,input_precision='tf32x3')
 tl.store(DK+kv,dk,u[:,None]<T);tl.store(DV+kv,dv,u[:,None]<T)
class Fine(torch.autograd.Function):
 @staticmethod
 def forward(ctx,q,k,v,c,ids,gates,bias,w,all_rpb=True):
  b,n,h,d=q.shape;t=w*w;g=n//t
  out=torch.empty_like(v);lse=torch.empty((b,h,n),device=q.device)
  _forward[(g*triton.cdiv(t,16),b*h)](q,k,v,c,ids,gates,bias,out,lse,g,h,ids.shape[-1],all_rpb,w,t,max(16,min(t,64)),num_warps=4,num_stages=1)
  ctx.save_for_backward(q,k,v,c,ids,gates,bias,out,lse);ctx.w=w;ctx.all=all_rpb
  return out
 @staticmethod
 @torch.autograd.function.once_differentiable
 def backward(ctx,grad):
  q,k,v,c,ids,gates,bias,out,lse=ctx.saved_tensors;b,n,h,d=q.shape;w=ctx.w;t=w*w;g=n//t;nr=ids.shape[-1];grad=grad.contiguous()
  key=(ids.long()+torch.arange(b*h,device=q.device).reshape(b,h,1,1)*g).flatten()
  order=torch.argsort(key,stable=True);counts=torch.bincount(key,minlength=b*h*g);starts=torch.cat([torch.zeros(1,device=q.device,dtype=torch.int64),counts.cumsum(0)])
  dq=torch.empty_like(v);dk=torch.empty_like(v);dv=torch.empty_like(v);dc=torch.empty_like(c);db=torch.zeros_like(bias);dl=torch.zeros_like(gates)
  _query_backward[(g*triton.cdiv(t,16),b*h)](q,k,v,c,ids,gates,bias,out,lse,grad,dq,dk,dv,dc,dl,db,g,h,nr,ctx.all,w,t,max(16,min(t,64)),num_warps=4,num_stages=1)
  _key_backward[(g*triton.cdiv(t,64),b*h)](q,k,v,c,ids,gates,bias,out,lse,grad,order,starts,dk,dv,g,h,nr,ctx.all,w,t,max(16,min(t,64)),num_warps=8,num_stages=1)
  dc=-dv.reshape(b,g,t,h,d).sum(2).permute(0,2,1,3).contiguous()
  return dq.to(q.dtype),dk.to(k.dtype),dv,dc,None,dl,db,None,None
