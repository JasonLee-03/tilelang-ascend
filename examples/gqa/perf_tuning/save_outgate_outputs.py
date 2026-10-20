"""B3 evidence: save gqa() outputs for OUT-OF-GATE shapes (dispatch must stay bit-identical)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import torch, torch_npu  # noqa
from gqa import gqa

tag = sys.argv[1]  # 'pre' or 'post'
shapes = [
    ("default_B1", 1, 128, 128, 8, 2, 128, True, torch.float16),      # per-head (16 inst < 20)
    ("c4like_kv2", 4, 256, 256, 64, 8, 128, False, torch.bfloat16),   # g2stack ns=2 (kv=2, out of flat gate)
    ("c5like_d256", 2, 128, 128, 32, 8, 256, True, torch.float16),    # per-head ladder D=256
    ("decode_c11", 16, 1, 2048, 32, 8, 128, False, torch.float16),    # dense decode path
    ("l1_skv124", 2, 128, 124, 32, 8, 128, True, torch.float16),      # kv=1 but Skv tail → padded kv=1... stacked 64 ≥ 40 → IN flat gate! (excluded from out-gate set)
]
out = {}
for name, B, S, Skv, Nq, Nkv, D, causal, dt in shapes:
    if name == "l1_skv124":
        continue  # in-gate shape (kv_padded=1): handled by B2 suite, not an out-gate probe
    torch.manual_seed(1234 + B + S)
    q = (torch.rand(B, S, Nq, D) * 2 - 1).to(dt).npu()
    k = (torch.rand(B, Skv, Nkv, D) * 2 - 1).to(dt).npu()
    v = (torch.rand(B, Skv, Nkv, D) * 2 - 1).to(dt).npu()
    o = gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
    torch.npu.synchronize()
    out[name] = o.cpu()
    print(f"saved {name} {tuple(o.shape)}")
torch.save(out, os.path.join(HERE, f"outgate_outputs_{tag}.pt"))
print("WROTE", os.path.join(HERE, f"outgate_outputs_{tag}.pt"))
