import json, torch
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open
import pyarrow as pa

root = '/mnt/llm_models/models/GLM-5.3-Flash'
croot = root + '/logestix'
idx = json.load(open(root + '/model.safetensors.index.json'))['weight_map']
def T(name):
    with safe_open(root + '/' + idx[name], framework='pt') as f:
        return f.get_tensor(name).float()

# fp8 e4m3 dequant for source experts (reference uses source weights)
def dequant_fp8(b, scale_inv, rows, cols):
    # fp8 e4m3 decode table via bits
    a = np.frombuffer(b, dtype=np.uint8).astype(np.uint16)
    sign = (a >> 7) & 1
    exp = (a >> 3) & 0xF
    man = a & 0x7
    val = np.where(exp == 0, man * 2.0**-9, np.ldexp((man | 8).astype(np.float64), exp - 10))
    val = np.where(a == 0x80, 0.0, val)  # -0
    val = np.where((a & 0x7F) == 0x7F, np.nan, val)
    val = np.where(sign == 1, -val, val)
    v = torch.from_numpy(val.astype(np.float32)).view(rows, cols)
    sc = torch.from_numpy(np.frombuffer(scale_inv, dtype=np.float32).copy())
    SC = (cols + 127) // 128; SR = (rows + 127) // 128
    sc = sc.view(SR, SC).repeat_interleave(128, 0).repeat_interleave(128, 1)[:rows, :cols]
    return v * sc

def read_raw(path, wanted):
    with pa.memory_map(path) as f:
        t = pa.ipc.open_file(f).read_all()
        names = t.column('name').to_pylist()
        out = {}
        for i, nm in enumerate(names):
            if any(nm.endswith(w) for w in wanted):
                out[nm] = dict(data=t.column('data')[i].as_buffer().to_pybytes(),
                               dtype=t.column('dtype')[i].as_py(),
                               shape=t.column('shape')[i].as_py())
        return out

def bf16_bytes_to_f32(b):
    a = np.frombuffer(b, dtype=np.uint16).astype(np.uint32) << 16
    return torch.frombuffer(a.tobytes(), dtype=torch.float32).clone()

H=4096; heads=64; hd=128; KD=8192; DI=12288; EI=2048; E=288; TOPK=8; hc=4
KDA = set([0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,36,37,38,40,41,42,44])

print('globals...', flush=True)
glob_raw = read_raw(croot+'/pinned/global.arrow', ['embed_tokens.weight','lm_head.weight','norm.weight'])
emb = bf16_bytes_to_f32(glob_raw['model.language_model.embed_tokens.weight']['data']).view(154880,4096)
lm_head = bf16_bytes_to_f32(glob_raw['lm_head.weight']['data']).view(154880,4096)
final_norm_w = bf16_bytes_to_f32(glob_raw['model.language_model.norm.weight']['data'])

def rms(x, w=None, eps=1e-6):
    v = x / ((x*x).mean(-1, keepdim=True) + eps) ** 0.5
    return v * w if w is not None else v

state = emb[785].clone().repeat(4,1)
for l in range(45):
    pref = f'model.language_model.layers.{l}.'
    hc_fn=T(pref+'hc_attn_fn'); hc_base=T(pref+'hc_attn_base'); hc_scale=T(pref+'hc_attn_scale')
    flat=state.reshape(-1); flat_n=flat/((flat*flat).mean()+1e-6)**0.5
    mixes=hc_fn@flat_n
    pre=torch.sigmoid(mixes[:4]*hc_scale[0]+hc_base[:4])+1e-6
    post=2*torch.sigmoid(mixes[4:8]*hc_scale[1]+hc_base[4:8])
    comb=torch.softmax((mixes[8:].view(4,4))*hc_scale[2]+hc_base[8:].view(4,4),dim=-1)+1e-6
    comb=comb/(comb.sum(-1,keepdim=True)+1e-6)
    for _ in range(19):
        comb=comb/(comb.sum(-1,keepdim=True)+1e-6)
        comb=comb/(comb.sum(-2,keepdim=True)+1e-6)
    collapsed=(pre.unsqueeze(1)*state).sum(0)
    xn=rms(collapsed, T(pref+'input_layernorm.weight'))
    if l in KDA:
        cq=T(pref+'self_attn.q_conv1d.weight').squeeze(1); ck=T(pref+'self_attn.k_conv1d.weight').squeeze(1); cv=T(pref+'self_attn.v_conv1d.weight').squeeze(1)
        q_raw=T(pref+'self_attn.q_proj.weight')@xn; k_raw=T(pref+'self_attn.k_proj.weight')@xn; v_raw=T(pref+'self_attn.v_proj.weight')@xn
        qc=F.silu(cq[:,3]*q_raw); kc=F.silu(ck[:,3]*k_raw); vc=F.silu(cv[:,3]*v_raw)
        def l2(v):
            m=v.view(heads,hd); return m/(((m*m).sum(-1,keepdim=True)+1e-6)**0.5)
        qn=l2(qc)*(1.0/(hd**0.5)); kn=l2(kc)
        fa=T(pref+'self_attn.f_a_proj.weight'); fb=T(pref+'self_attn.f_b_proj.weight'); dtb=T(pref+'self_attn.dt_bias'); Alog=T(pref+'self_attn.A_log')
        gd=(-5.0*torch.sigmoid(torch.exp(Alog).unsqueeze(1)*(fb@(fa@xn)+dtb).view(heads,hd))).exp()
        beta=torch.sigmoid(T(pref+'self_attn.b_proj.weight')@xn)
        attn=torch.zeros(heads,hd); vcv=vc.view(heads,hd)
        for h in range(heads):
            attn[h]=(torch.outer(kn[h], vcv[h]*beta[h]))@qn[h]
        gate=torch.sigmoid((T(pref+'self_attn.g_b_proj.weight')@(T(pref+'self_attn.g_a_proj.weight')@xn)).view(heads,hd))
        on=T(pref+'self_attn.o_norm.weight')
        nrm=attn/((attn*attn).mean(-1,keepdim=True)+1e-6)**0.5
        attn_out=(nrm*gate*on).reshape(-1)@T(pref+'self_attn.o_proj.weight').T
    else:
        # DSA single-token: attn out = o_proj(v) where v per head from kv_b@rms(kv_a@x)
        kv_a = dequant_fp8(*[read_raw(croot+f'/pinned/layer_{l:03d}_nvfp4.arrow', ['kv_a_proj_with_mqa.weight'])[f'...layer.{l}....kv_a_proj_with_mqa.weight' if False else 'x']['data']] if False else (None,None), 0,0) if False else None
        # DSA projections in converted are NVFP4; for the CPU reference use SOURCE fp8:
        src_map = json.load(open(root+'/model.safetensors.index.json'))['weight_map']
        def S(name, rows, cols):
            key=f'model.language_model.layers.{l}.'+name
            with safe_open(root+'/'+src_map[key], framework='pt') as f:
                w=f.get_tensor(key)
                sc=f.get_tensor(key.replace('.weight','.weight_scale_inv')) if key.replace('.weight','.weight_scale_inv') in src_map else None
            if w.dtype == torch.float8_e4m3fn:
                return dequant_fp8(w.view(torch.uint8).numpy().tobytes(), sc.numpy().tobytes(), rows, cols)
            return w.float()
        q_a=S('self_attn.q_a_proj.weight', 1536, H)
        qr=rms(q_a@xn, T(pref+'self_attn.q_a_layernorm.weight'))
        q_b=S('self_attn.q_b_proj.weight', heads*256, 1536)
        q=q_b@qr
        kv_a=S('self_attn.kv_a_proj_with_mqa.weight', 512, H)
        kvp=rms(kv_a@xn, T(pref+'self_attn.kv_a_layernorm.weight'))
        kv_b=T(pref+'self_attn.kv_b_proj.weight').float()  # BF16 raw [32768, 512]
        kvb=kv_b@kvp  # [32768]
        # per head k[256], v[256]; single token → softmax=1 → out=v
        v_all=kvb.view(heads, 512)[:, 256:]  # [64,256]
        attn_out=(v_all.reshape(-1))
        o_big=S('self_attn.o_proj.weight', H, heads*256)
        attn_out=o_big= o_big if False else o_big
        attn_out = dequant_o = None
        # o_proj is NVFP4 converted; use source fp8:
        o_src=S('self_attn.o_proj.weight', H, heads*256)
        attn_out=o_src@v_all.reshape(-1)
    # FFN
    state2 = post.unsqueeze(1)*attn_out.unsqueeze(0).expand(4,-1) + comb.T@state
    xn2=rms(state2[0] if False else collapsed, T(pref+'post_attention_layernorm.weight')) if False else None
    # NOTE: ffn hc_pre uses state2 (updated state) — redo collapse for ffn:
    flat2=state2.reshape(-1); flat2_n=flat2/((flat2*flat2).mean()+1e-6)**0.5
    mixes2=hc_fn@flat2_n if False else T(pref+'hc_ffn_fn')@flat2_n
    hc_ffn_fn=T(pref+'hc_ffn_fn')
    mixes2=hc_ffn_fn@flat2_n
    pre2=torch.sigmoid(mixes2[:4]*hc_scale[0]+hc_base[:4])+1e-6
    collapsed2=(pre2.unsqueeze(1)*state2).sum(0)
    xn2=rms(collapsed2, T(pref+'post_attention_layernorm.weight'))
    if l < 3:
        g=T(pref+'mlp.gate_proj.weight')@xn2; u=T(pref+'mlp.up_proj.weight')@xn2
        g=g.clamp(max=10.0); u=u.clamp(-10.0,10.0)
        mlp=(F.silu(g)*u)@T(pref+'mlp.down_proj.weight').T
    else:
        gw=T(pref+'mlp.gate.weight').float(); gbias=T(pref+'mlp.gate.e_score_correction_bias')
        scores=torch.sigmoid(gw@xn2)+gbias
        top=scores.topk(TOPK).indices
        tw=torch.sigmoid((gw@xn2))[top]
        tw=tw/(tw.sum()+1e-20)*2.5
        acc=torch.zeros(H)
        for kk,e in enumerate(top.tolist()):
            path=croot+f'/experts/layer_{l:03d}/exp_{e:03d}.arrow'
            with pa.memory_map(path) as f:
                et=pa.ipc.open_file(f).read_all()
            w1p=et.column(0)[0].as_buffer().to_pybytes(); w2p=et.column(1)[0].as_buffer().to_pybytes(); w3p=et.column(2)[0].as_buffer().to_pybytes()
            s1p=et.column(3)[0].as_buffer().to_pybytes(); s2p=et.column(4)[0].as_buffer().to_pybytes(); s3p=et.column(5)[0].as_buffer().to_pybytes()
            gs1=et.column(6)[0].as_py(); gs2=et.column(7)[0].as_py(); gs3=et.column(8)[0].as_py()
            def dq(packed, scales, gs, rows, cols):
                e2m1=torch.tensor([0,.5,1,1.5,2,3,4,6,-0,-.5,-1,-1.5,-2,-3,-4,-6])
                p=torch.frombuffer(bytearray(packed), dtype=torch.uint8)
                n=rows*cols
                nib=torch.zeros(n, dtype=torch.int64)
                lo=(p&0xF).to(torch.int64); hi=((p>>4)&0xF).to(torch.int64)
                nib[0::2]=lo; nib[1::2]=hi[:len(hi)]
                vals=e2m1[nib[:n]].view(rows,cols)
                sb=torch.frombuffer(bytearray(scales), dtype=torch.uint8).to(torch.int64)
                sv=torch.where(sb==0, torch.zeros_like(sb, dtype=torch.float32), torch.exp2(((sb>>3)).to(torch.float32)-7))
                blocks=sv.repeat_interleave(16)[:n].view(rows,cols)
                return vals*blocks*gs
            w1=dq(w1p,s1p,gs1,EI,H); w2=dq(w2p,s2p,gs2,H,EI); w3=dq(w3p,s3p,gs3,EI,H)
            g_=w1@xn2; u_=w3@xn2
            g_=g_.clamp(max=10.0); u_=u_.clamp(-10.0,10.0)
            acc += tw[kk]*((F.silu(g_)*u_)@w2)
        mlp=acc
    state = post.unsqueeze(1)*mlp.unsqueeze(0).expand(4,-1) + comb.T@state2
    print('layer', l, 'state_mean', float(state.mean()), flush=True)

hn=state.mean(0)
hn=rms(hn, final_norm_w)
logits=lm_head@hn
top=torch.topk(logits, 5)
print('CPU top-5:', top.indices.tolist())
print('CPU top-5 vals:', [round(v,3) for v in top.values.tolist()])
