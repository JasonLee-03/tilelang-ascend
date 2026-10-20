"""Verify dispatch selection for in-gate and out-of-gate shapes."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import torch, torch_npu  # noqa
import gqa as G

probes = [
    ("R2_in_gate",    dict(query=(2,128,32,128), key=(2,128,8,128), causal=True,  dt=torch.float16)),
    ("R3_in_gate",    dict(query=(2,128,128,128), key=(2,128,8,128), causal=True, dt=torch.bfloat16)),
    ("c2_in_gate",    dict(query=(2,128,48,128), key=(2,128,4,128), causal=False, dt=torch.float16)),
    ("R4_out_kv2",    dict(query=(4,256,64,128), key=(4,256,8,128), causal=False, dt=torch.bfloat16)),
    ("B1_out_small",  dict(query=(1,128,8,128), key=(1,128,2,128), causal=True,  dt=torch.float16)),
]
for name, p in probes:
    B,S,Nq,D = p['query']; _,Skv,Nkv,_ = p['key']
    G._KERNEL_BUILD_CACHE.clear(); G._COMPILED_CACHE.clear()
    torch.manual_seed(0)
    q=(torch.rand(B,S,Nq,D)*2-1).to(p['dt']).npu(); k=(torch.rand(B,Skv,Nkv,D)*2-1).to(p['dt']).npu(); v=(torch.rand(B,Skv,Nkv,D)*2-1).to(p['dt']).npu()
    o = G.gqa(query=q,key=k,value=v,scaleValue=-1.0,is_causal=p['causal'])
    torch.npu.synchronize()
    chosen = [v_[2] for v_ in G._KERNEL_BUILD_CACHE.values()]
    print(f"{name}: chosen={chosen} out={tuple(o.shape)}")
