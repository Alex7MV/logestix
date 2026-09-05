import json, torch
import torch.nn.functional as F
from safetensors import safe_open
root = '/mnt/llm_models/models/GLM-5.3-Flash'
idx = json.load(open(root + '/model.safetensors.index.json'))['weight_map']
def T(name):
    with safe_open(root + '/' + idx[name], framework='pt') as f:
        return f.get_tensor(name).float()
pref = 'model.language_model.layers.0.'
emb = T('model.language_model.embed_tokens.weight')
hc_fn = T(pref+'hc_attn_fn'); hc_base = T(pref+'hc_attn_base'); hc_scale = T(pref+'hc_attn_scale')
x = emb[785]; streams = x.repeat(4,1)
flat = streams.reshape(-1); flat_n = flat / ((flat*flat).mean()+1e-6)**0.5
mixes = hc_fn @ flat_n
pre = torch.sigmoid(mixes[:4]*hc_scale[0] + hc_base[:4]) + 1e-6
collapsed = (pre.unsqueeze(1)*streams).sum(0)
ln = T(pref+'input_layernorm.weight')
xn = collapsed/((collapsed*collapsed).mean()+1e-6)**0.5*ln
Wq=T(pref+'self_attn.q_proj.weight'); Wk=T(pref+'self_attn.k_proj.weight'); Wv=T(pref+'self_attn.v_proj.weight')
cq=T(pref+'self_attn.q_conv1d.weight').squeeze(1); ck=T(pref+'self_attn.k_conv1d.weight').squeeze(1); cv=T(pref+'self_attn.v_conv1d.weight').squeeze(1)
bp=T(pref+'self_attn.b_proj.weight')
q_raw=Wq@xn; k_raw=Wk@xn; v_raw=Wv@xn
qc=F.silu(cq[:,3]*q_raw); kc=F.silu(ck[:,3]*k_raw); vc=F.silu(cv[:,3]*v_raw)
def l2(v):
    m=v.view(64,128); return m/(((m*m).sum(-1,keepdim=True)+1e-6)**0.5)
qn=l2(qc)*(1.0/(128**0.5)); kn=l2(kc)
beta=torch.sigmoid(bp@xn)
print('CPU qc[0:4]:', [round(v,8) for v in qc[:4].tolist()])
print('CPU vc[0:4]:', [round(v,8) for v in vc[:4].tolist()])
print('CPU kn[0][0:4]:', [round(v,8) for v in kn[0][:4].tolist()])
print('CPU qn[0][0:4]:', [round(v,8) for v in qn[0][:4].tolist()])
print('CPU beta[0]:', round(beta[0].item(),6))
print('CPU v_raw[0:4]:', [round(v,8) for v in v_raw[:4].tolist()])
