"""B8 complete backward, with deterministic inverse routing entirely on device.

Unchanged pinned forward; DP/gradient products retain TF32x3. Optional native QK uses the already-BF16 operands with FP32 accumulation; FP32 input checks retain TF32x3. Smaller key tiles limit compiler resources.
Dense int32 inverse-routing workspace is transient (not an attention score matrix). Prefix/compaction preserve ascending-query edge order without argsort/bincount or CPU scalar synchronization. No gradients are dropped. Non-B8 inputs are delegated to the pinned baseline.
"""
import torch
import triton
import triton.language as tl
from .blocksize_original import _forward
from .blocksize_keyowned import Fine as BaselineFine, _key_backward as original_key

CONFIG=dict(chunk=8,query_partial=True,key_split=True,key_warps=4,key_tile=32,native_qk=True)
METADATA={}
def record(k,name):
    if name not in METADATA:
        METADATA[name]=dict(regs=getattr(k,'n_regs',None),spills=getattr(k,'n_spills',None),
                            shared=getattr(k.metadata,'shared',None),hash=getattr(k,'hash',None))

@triton.jit
def _query(Q,K,V,C,I,L,Bias,O,LSE,DO,DQ,DL,DB,
           G:tl.constexpr,H:tl.constexpr,NR:tl.constexpr,ALL:tl.constexpr,NATIVE:tl.constexpr):
    g=tl.program_id(0)//4;tile=tl.program_id(0)%4;bh=tl.program_id(1)
    t=tile*16+tl.arange(0,16);u=tl.arange(0,64);d=tl.arange(0,64)
    off=((bh//H*G+g)*64+t[:,None])*H*64+bh%H*64+d[None,:]
    qraw=tl.load(Q+off);q=qraw.to(tl.float32);do=tl.load(DO+off)
    delta=tl.sum(do*tl.load(O+off),1);lse=tl.load(LSE+(bh*G+g)*64+t)
    ri=(t[:,None]//8-u[None,:]//8+7)*15+t[:,None]%8-u[None,:]%8+7
    bias=tl.load(Bias+bh%H*225+ri)
    dq=tl.full((16,64),0.,tl.float32);db=tl.full((16,64),0.,tl.float32)
    for j in range(NR):
        route=(bh*G+g)*NR+j;dest=tl.load(I+route)
        kv=((bh//H*G+dest)*64+u[:,None])*H*64+bh%H*64+d[None,:]
        kraw=tl.load(K+kv);k=kraw.to(tl.float32)
        residual=tl.load(V+kv)-tl.load(C+(bh*G+dest)*64+d)[None,:]
        if NATIVE:score=tl.dot(qraw,tl.trans(kraw))*.125
        else:score=tl.dot(q,tl.trans(k),input_precision='tf32x3')*.125
        if ALL:score+=bias
        else:score+=tl.where(dest==g,bias,0.)
        score+=tl.load(L+route)
        p=tl.exp(score-lse[:,None])
        dp=tl.dot(do,tl.trans(residual),input_precision='tf32x3')
        ds=p*(dp-delta[:,None])
        dq+=tl.dot(ds,k,input_precision='tf32x3')*.125
        dl=tl.sum(tl.sum(ds,0),0)
        if NR==1:dl=0.
        tl.store(DL+route*4+tile,dl)
        if ALL:db+=ds
        else:db+=tl.where(dest==g,ds,0.)
    tl.store(DQ+off,dq)
    tl.store(DB+((bh*G+g)*64+t[:,None])*64+u[None,:],db)

@triton.jit
def _reduce_bias(X,Y):
    h=tl.program_id(0);r=tl.program_id(1);q=tl.arange(0,64)
    ky=q//8-(r//15-7);kx=q%8-(r%15-7);valid=(ky>=0)&(ky<8)&(kx>=0)&(kx<8)
    x=tl.load(X+h*4096+q*64+ky*8+kx,valid,other=0.)
    tl.store(Y+h*225+r,tl.sum(x,0))

@triton.jit
def _make_tasks(Starts,TaskStarts,TaskKey,TaskEdge,G:tl.constexpr,CHUNK:tl.constexpr,MAX_TASKS:tl.constexpr):
    key=tl.program_id(0);j=tl.arange(0,MAX_TASKS)
    a=tl.load(TaskStarts+key);b=tl.load(TaskStarts+key+1);edge=tl.load(Starts+key)
    tl.store(TaskKey+a+j,key,a+j<b);tl.store(TaskEdge+a+j,edge+j*CHUNK,a+j<b)

@triton.jit
def _key_chunks(Q,K,V,C,I,L,Bias,O,LSE,DO,Order,Starts,TaskStarts,TaskKey,TaskEdge,DK,DV,
                G:tl.constexpr,H:tl.constexpr,NR:tl.constexpr,BHG:tl.constexpr,CHUNK:tl.constexpr,ALL:tl.constexpr,BT:tl.constexpr,NATIVE:tl.constexpr):
    task=tl.program_id(0);total=tl.load(TaskStarts+BHG)
    if task<total:
        key=tl.load(TaskKey+task);bh=key//G;dest=key%G
        first=tl.load(TaskEdge+task);stop=tl.minimum(first+CHUNK,tl.load(Starts+key+1))
        u=tl.program_id(1)*BT+tl.arange(0,BT);d=tl.arange(0,64);qt=tl.arange(0,32)
        kv=((bh//H*G+dest)*64+u[:,None])*H*64+bh%H*64+d[None,:]
        kraw=tl.load(K+kv);kval=kraw.to(tl.float32)
        residual=tl.load(V+kv)-tl.load(C+(bh*G+dest)*64+d)[None,:]
        dk=tl.full((BT,64),0.,tl.float32);dv=tl.full((BT,64),0.,tl.float32)
        for edge in range(first,stop):
            route=tl.load(Order+edge);src=(route//NR)%G;gate=tl.load(L+route)
            for sub in range(2):
                t=sub*32+qt;off=((bh//H*G+src)*64+t[:,None])*H*64+bh%H*64+d[None,:]
                qraw=tl.load(Q+off);q=qraw.to(tl.float32);do=tl.load(DO+off)
                delta=tl.sum(do*tl.load(O+off),1);lse=tl.load(LSE+(bh*G+src)*64+t)
                ri=(t[:,None]//8-u[None,:]//8+7)*15+t[:,None]%8-u[None,:]%8+7
                bias=tl.load(Bias+bh%H*225+ri)
                if NATIVE:score=tl.dot(qraw,tl.trans(kraw))*.125
                else:score=tl.dot(q,tl.trans(kval),input_precision='tf32x3')*.125
                if ALL:score+=bias
                else:score+=tl.where(src==dest,bias,0.)
                p=tl.exp(score+gate-lse[:,None])
                dp=tl.dot(do,tl.trans(residual),input_precision='tf32x3');ds=p*(dp-delta[:,None])
                dk+=tl.dot(tl.trans(ds),q,input_precision='tf32x3')*.125
                dv+=tl.dot(tl.trans(p),do,input_precision='tf32x3')
        flat=task*4096+u[:,None]*64+d[None,:]
        tl.store(DK+flat,dk);tl.store(DV+flat,dv)

@triton.jit
def _reduce_keys(PK,PV,TaskStarts,DK,DV,G:tl.constexpr,H:tl.constexpr,BLOCK:tl.constexpr):
    key=tl.program_id(0);bh=key//G;dest=key%G;ix=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    first=tl.load(TaskStarts+key);stop=tl.load(TaskStarts+key+1)
    dk=tl.full((BLOCK,),0.,tl.float32);dv=tl.full((BLOCK,),0.,tl.float32)
    for task in range(first,stop):
        dk+=tl.load(PK+task*4096+ix);dv+=tl.load(PV+task*4096+ix)
    off=((bh//H*G+dest)*64+ix//64)*H*64+bh%H*64+ix%64
    tl.store(DK+off,dk);tl.store(DV+off,dv)

@triton.jit
def _inverse_scatter(I,Dense,G:tl.constexpr,NR:tl.constexpr,E:tl.constexpr,BLOCK:tl.constexpr):
    e=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);dest=tl.load(I+e,e<E,other=0)
    bh=e//(G*NR);src=e//NR%G
    # Every query selects unique keys, so no write collisions.
    tl.store(Dense+(bh*G+dest)*G+src,e,e<E)

@triton.jit
def _inverse_counts(Dense,Counts,G:tl.constexpr,BG:tl.constexpr):
    key=tl.program_id(0);src=tl.arange(0,BG);edge=tl.load(Dense+key*G+src,src<G,other=-1)
    tl.store(Counts+key,tl.sum((edge>=0).to(tl.int32),0))

@triton.jit
def _inverse_prefix(Counts,Starts,Tasks,N:tl.constexpr,CHUNK:tl.constexpr,BLOCK:tl.constexpr):
    ix=tl.arange(0,BLOCK);counts=tl.load(Counts+ix,ix<N,other=0)
    sums=tl.cumsum(counts,0);tasks=tl.cumsum(tl.cdiv(counts,CHUNK),0)
    tl.store(Starts,0);tl.store(Tasks,0)
    tl.store(Starts+ix+1,sums,ix<N);tl.store(Tasks+ix+1,tasks,ix<N)

@triton.jit
def _inverse_compact_tasks(Dense,Starts,Tasks,Order,TaskKey,TaskEdge,G:tl.constexpr,CHUNK:tl.constexpr,BG:tl.constexpr,BT:tl.constexpr):
    key=tl.program_id(0);src=tl.arange(0,BG);edge=tl.load(Dense+key*G+src,src<G,other=-1)
    valid=edge>=0;rank=tl.cumsum(valid.to(tl.int32),0);start=tl.load(Starts+key)
    tl.store(Order+start+rank-1,edge,valid)
    task=tl.arange(0,BT);first=tl.load(Tasks+key);last=tl.load(Tasks+key+1)
    tl.store(TaskKey+first+task,key,first+task<last)
    tl.store(TaskEdge+first+task,start+task*CHUNK,first+task<last)

def backward(ctx,grad):
    q,k,v,c,ids,gates,bias,out,lse=ctx.saved_tensors
    b,n,h,d=q.shape;g=n//64;nr=ids.shape[-1];grad=grad.contiguous();cfg=ctx.cfg;native=cfg['native_qk'] and q.dtype==torch.bfloat16
    dq=torch.empty_like(v);dk=torch.empty_like(v);dv=torch.empty_like(v)
    if cfg['query_partial']:
        dbpart=torch.empty((b,h,g,64,64),device=q.device,dtype=torch.float32)
        dlpart=torch.empty((*gates.shape,4),device=q.device,dtype=torch.float32)
        krn=_query[(g*4,b*h)](q,k,v,c,ids,gates,bias,out,lse,grad,dq,dlpart,dbpart,g,h,nr,ctx.all,native,num_warps=4,num_stages=1)
        record(krn,f'query_{g}_{nr}_n{native}')
        dbsum=dbpart.sum((0,2));db=torch.empty_like(bias)
        _reduce_bias[(h,225)](dbsum,db,num_warps=4);dl=dlpart.sum(-1)
        del dbpart,dbsum,dlpart
    else:
        from .blocksize_query_pass import _backward as query_old
        dl=torch.zeros_like(gates);db=torch.zeros_like(bias);dc=torch.empty_like(c)
        query_old[(g*4,b*h)](q,k,v,c,ids,gates,bias,out,lse,grad,dq,dk,dv,dc,dl,db,g,h,nr,ctx.all,8,64,64,num_warps=4,num_stages=1)
    if cfg['key_split']:
        chunk=cfg['chunk'];bhg=b*h*g;edges=ids.numel();capacity=edges//chunk+bhg
        dense=torch.full((bhg,g),-1,device=q.device,dtype=torch.int32)
        counts=torch.empty(bhg,device=q.device,dtype=torch.int32)
        starts=torch.empty(bhg+1,device=q.device,dtype=torch.int64);task_starts=torch.empty_like(starts)
        order=torch.empty(edges,device=q.device,dtype=torch.int64)
        task_key=torch.empty(capacity,device=q.device,dtype=torch.int32);task_edge=torch.empty_like(task_key,dtype=torch.int64)
        _inverse_scatter[(triton.cdiv(edges,256),)](ids,dense,g,nr,edges,256,num_warps=4)
        _inverse_counts[(bhg,)](dense,counts,g,triton.next_power_of_2(g),num_warps=4)
        _inverse_prefix[(1,)](counts,starts,task_starts,bhg,chunk,triton.next_power_of_2(bhg),num_warps=4)
        _inverse_compact_tasks[(bhg,)](dense,starts,task_starts,order,task_key,task_edge,g,chunk,triton.next_power_of_2(g),triton.next_power_of_2(triton.cdiv(g,chunk)),num_warps=4)
        pk=torch.empty((capacity,64,64),device=q.device,dtype=torch.float32);pv=torch.empty_like(pk)
        krn=_key_chunks[(capacity,64//cfg['key_tile'])](q,k,v,c,ids,gates,bias,out,lse,grad,order,starts,task_starts,task_key,task_edge,pk,pv,g,h,nr,b*h*g,chunk,ctx.all,cfg['key_tile'],native,num_warps=cfg['key_warps'],num_stages=1)
        record(krn,f'key_{g}_{nr}_c{chunk}_w{cfg["key_warps"]}_t{cfg["key_tile"]}_n{native}')
        _reduce_keys[(b*h*g,8)](pk,pv,task_starts,dk,dv,g,h,512,num_warps=4)
    else:
        key=(ids.long()+torch.arange(b*h,device=q.device).reshape(b,h,1,1)*g).flatten()
        order=torch.argsort(key,stable=True);counts=torch.bincount(key,minlength=b*h*g)
        starts=torch.cat([torch.zeros(1,device=q.device,dtype=torch.int64),counts.cumsum(0)])
        original_key[(g,b*h)](q,k,v,c,ids,gates,bias,out,lse,grad,order,starts,dk,dv,g,h,nr,ctx.all,8,64,64,num_warps=8,num_stages=1)
    dc=-dv.reshape(b,g,64,h,d).sum(2).permute(0,2,1,3).contiguous()
    return dq.to(q.dtype),dk.to(k.dtype),dv,dc,None,dl,db,None,None

class _Fine(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,c,ids,gates,bias,w,all_rpb=True):
        assert w==8 and q.shape[-1]==64
        b,n,h,d=q.shape;g=n//64
        out=torch.empty_like(v);lse=torch.empty((b,h,n),device=q.device)
        _forward[(g*4,b*h)](q,k,v,c,ids,gates,bias,out,lse,g,h,ids.shape[-1],all_rpb,8,64,64,num_warps=4,num_stages=1)
        ctx.save_for_backward(q,k,v,c,ids,gates,bias,out,lse)
        ctx.all=all_rpb;ctx.cfg=CONFIG.copy()
        return out
    backward=staticmethod(torch.autograd.function.once_differentiable(backward))

class Fine:
    @staticmethod
    def apply(q,k,v,c,ids,gates,bias,w,all_rpb=True):
        return (_Fine if w==8 else BaselineFine).apply(q,k,v,c,ids,gates,bias,w,all_rpb)
