"""Trainable BRHA b4/b8/b16, query16/key64 tiling, atomic first-order backward."""
import torch
import triton
import triton.language as tl
@triton.jit
def _forward(Q, K, V, C, I, L, Bias, O, LSE,
             G: tl.constexpr, H: tl.constexpr, NR: tl.constexpr, ALL: tl.constexpr, W: tl.constexpr, T: tl.constexpr, BT: tl.constexpr):
    g, bh = tl.program_id(0) // triton.cdiv(T,16), tl.program_id(1)
    t = (tl.program_id(0) % triton.cdiv(T,16))*16 + tl.arange(0,16)
    u, d = tl.arange(0,BT), tl.arange(0,64)
    off = ((bh // H * G + g) * T + t[:, None]) * H * 64 + bh % H * 64 + d[None, :]
    q = tl.load(Q + off, t[:, None] < T, other=0.)
    ri = (t[:, None] // W - u[None, :] // W + W-1) * (2*W-1) + t[:, None] % W - u[None, :] % W + W-1
    bias = tl.load(Bias + bh % H * ((2*W-1)**2) + ri, (t[:,None]<T)&(u[None,:]<T), other=0.)
    maximum = tl.full((16,), -float('inf'), tl.float32)
    denom = tl.full((16,), 0, tl.float32)
    acc = tl.full((16, 64), 0, tl.float32)
    for sub in range(NR * triton.cdiv(T,64)):
        j = sub // triton.cdiv(T,64)
        u = (sub % triton.cdiv(T,64))*64 + tl.arange(0,BT)
        ri = (t[:,None]//W-u[None,:]//W+W-1)*(2*W-1)+t[:,None]%W-u[None,:]%W+W-1
        bias = tl.load(Bias+bh%H*((2*W-1)**2)+ri,(t[:,None]<T)&(u[None,:]<T),other=0.)
        route = (bh * G + g) * NR + j
        dest = tl.load(I + route)
        kv = ((bh // H * G + dest) * T + u[:, None]) * H * 64 + bh % H * 64 + d[None, :]
        k = tl.load(K + kv, u[:, None] < T, other=0.)
        residual = tl.load(V + kv, u[:, None] < T, other=0.) - tl.load(C + (bh * G + dest) * 64 + d)[None, :]
        score = tl.dot(q, tl.trans(k), input_precision='tf32x3') * 0.125
        if ALL:
            score += bias
        else:
            score += tl.where(dest == g, bias, 0.)
        score += tl.load(L + route)
        score = tl.where(u[None,:] < T, score, -float('inf'))
        newmax = tl.maximum(maximum, tl.max(score, 1))
        alpha = tl.exp(maximum - newmax)
        p = tl.exp(score - newmax[:, None])
        acc = acc * alpha[:, None] + tl.dot(p, residual, input_precision='tf32x3')
        denom = denom * alpha + tl.sum(p, 1)
        maximum = newmax
    tl.store(O + off, acc / denom[:, None], t[:,None]<T)
    tl.store(LSE + (bh * G + g) * T + t, maximum + tl.log(denom), t<T)



@triton.jit
def _backward(Q, K, V, C, I, L, Bias, O, LSE, DO, DQ, DK, DV, DC, DL, DB,
              G: tl.constexpr, H: tl.constexpr, NR: tl.constexpr, ALL: tl.constexpr, W: tl.constexpr, T: tl.constexpr, BT: tl.constexpr):
    g, bh = tl.program_id(0) // triton.cdiv(T,16), tl.program_id(1)
    t = (tl.program_id(0) % triton.cdiv(T,16))*16 + tl.arange(0,16)
    u, d = tl.arange(0,BT), tl.arange(0,64)
    off = ((bh // H * G + g) * T + t[:, None]) * H * 64 + bh % H * 64 + d[None, :]
    q = tl.load(Q + off, t[:, None] < T, other=0.).to(tl.float32)
    do = tl.load(DO + off, t[:, None] < T, other=0.)
    delta = tl.sum(do * tl.load(O + off, t[:, None] < T, other=0.), 1)
    lse = tl.load(LSE + (bh * G + g) * T + t, t < T, other=0.)
    ri = (t[:, None] // W - u[None, :] // W + W-1) * (2*W-1) + t[:, None] % W - u[None, :] % W + W-1
    bias = tl.load(Bias + bh % H * ((2*W-1)**2) + ri, (t[:,None]<T)&(u[None,:]<T), other=0.)
    dq = tl.full((16, 64), 0, tl.float32)
    for sub in range(NR * triton.cdiv(T,64)):
        j = sub // triton.cdiv(T,64)
        u = (sub % triton.cdiv(T,64))*64 + tl.arange(0,BT)
        ri = (t[:,None]//W-u[None,:]//W+W-1)*(2*W-1)+t[:,None]%W-u[None,:]%W+W-1
        bias = tl.load(Bias+bh%H*((2*W-1)**2)+ri,(t[:,None]<T)&(u[None,:]<T),other=0.)
        route = (bh * G + g) * NR + j
        dest = tl.load(I + route)
        kv = ((bh // H * G + dest) * T + u[:, None]) * H * 64 + bh % H * 64 + d[None, :]
        k = tl.load(K + kv, u[:, None] < T, other=0.).to(tl.float32)
        residual = tl.load(V + kv, u[:, None] < T, other=0.) - tl.load(C + (bh * G + dest) * 64 + d)[None, :]
        score = tl.dot(q, tl.trans(k), input_precision='tf32x3') * 0.125
        if ALL:
            score += bias
        else:
            score += tl.where(dest == g, bias, 0.)
        score += tl.load(L + route)
        score = tl.where(u[None,:] < T, score, -float('inf'))
        p = tl.exp(score - lse[:, None])
        dp = tl.dot(do, tl.trans(residual), input_precision='tf32x3')
        p = tl.where(t[:,None]<T, p, 0.)
        ds = p * (dp - delta[:, None])
        dr = tl.dot(tl.trans(p), do, input_precision='tf32x3')
        dq += tl.dot(ds, k, input_precision='tf32x3') * 0.125
        dk = tl.dot(tl.trans(ds), q, input_precision='tf32x3') * 0.125
        tl.atomic_add(DK + kv, dk, u[:,None]<T, sem='relaxed')
        tl.atomic_add(DV + kv, dr, u[:,None]<T, sem='relaxed')
        tl.atomic_add(DC + (bh * G + dest) * 64 + d, -tl.sum(dr, 0), sem='relaxed')
        # A single route's gate is a constant softmax shift: exactly zero.
        if NR == 1:
            tl.store(DL + route, 0.)
        else:
            tl.atomic_add(DL + route, tl.sum(tl.sum(ds, 0), 0), sem='relaxed')
        if ALL:
            db = ds
        else:
            db = tl.where(dest == g, ds, 0.)
        tl.atomic_add(DB + bh % H * ((2*W-1)**2) + ri, db, (t[:,None]<T)&(u[None,:]<T), sem='relaxed')
    tl.store(DQ + off, dq, t[:,None]<T)




class Fine(torch.autograd.Function):
 @staticmethod
 def forward(ctx,q,k,v,c,ids,gates,bias,w,all_rpb=True):
  b,n,h,d=q.shape;t=w*w;g=n//t
  assert d==64 and n%t==0 and bias.shape==(h,(2*w-1)**2)
  assert all(z.is_contiguous() for z in (q,k,v,c,ids,gates,bias))
  out=torch.empty_like(v);lse=torch.empty((b,h,n),device=q.device)
  _forward[(g*triton.cdiv(t,16),b*h)](q,k,v,c,ids,gates,bias,out,lse,g,h,ids.shape[-1],all_rpb,w,t,max(16,min(t,64)),num_warps=4,num_stages=1)
  ctx.save_for_backward(q,k,v,c,ids,gates,bias,out,lse);ctx.w=w;ctx.all=all_rpb
  return out
 @staticmethod
 @torch.autograd.function.once_differentiable
 def backward(ctx,grad):
  q,k,v,c,ids,gates,bias,out,lse=ctx.saved_tensors;b,n,h,d=q.shape;w=ctx.w;t=w*w;g=n//t
  dq=torch.empty_like(v);dk,dv,dc,db=[torch.zeros_like(z,dtype=torch.float32) for z in (k,v,c,bias)];dl=torch.zeros_like(gates)
  _backward[(g*triton.cdiv(t,16),b*h)](q,k,v,c,ids,gates,bias,out,lse,grad.contiguous(),dq,dk,dv,dc,dl,db,g,h,ids.shape[-1],ctx.all,w,t,max(16,min(t,64)),num_warps=4,num_stages=1)
  return dq.to(q.dtype),dk.to(k.dtype),dv,dc,None,dl,db,None,None

def reference(q,k,v,c,ids,gates,bias,w,all_rpb=True):
 b,n,h,d=q.shape;t=w*w;g=n//t;r=ids.shape[-1]
 bi=torch.arange(b,device=q.device)[:,None,None,None];hi=torch.arange(h,device=q.device)[None,:,None,None]
 kw=k.float().reshape(b,g,t,h,d).permute(0,3,1,2,4)
 rw=v.reshape(b,g,t,h,d).permute(0,3,1,2,4)-c[:,:,:,None,:]
 ks=kw[bi,hi,ids.long()].reshape(b,h,g,r*t,d);rs=rw[bi,hi,ids.long()].reshape(b,h,g,r*t,d)
 qw=q.float().reshape(b,g,t,h,d).permute(0,3,1,2,4);p=torch.arange(t,device=q.device)
 ri=(p[:,None]//w-p[None,:]//w+w-1)*(2*w-1)+p[:,None]%w-p[None,:]%w+w-1
 rb=bias[:,ri][None,:,None,:,None,:]
 if not all_rpb:rb=rb*(ids==torch.arange(g,device=q.device)[None,None,:,None])[...,None,:,None]
 logits=(qw@ks.transpose(-1,-2))*.125
 logits=logits.reshape(b,h,g,t,r,t)+rb+gates[...,None,:,None]
 return (logits.flatten(-2).softmax(-1)@rs).permute(0,2,3,1,4).reshape(b,n,h,d).contiguous()
