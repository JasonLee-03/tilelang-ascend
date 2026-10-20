"""Debug: diff pattern of flat variant vs shipped kernel (mc2 edge case)."""
import os, sys
import torch, torch_npu  # noqa
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from gqa import _BIG_NEG, _dummy_mask, gqa, transpose_out, transpose_qkv  # noqa: E402
from gqa_flat_variant import build_flat_kernel  # noqa: E402
from run_flat_variant import _padded_causal_mask  # noqa: E402

B, S, S_kv, N_q, N_kv, D = 5, 128, 128, 8, 2, 128
causal = "--noncausal" not in sys.argv
dtype = torch.float16
torch.manual_seed(42 + B + S + S_kv)
q = (torch.rand(B, S, N_q, D) * 2 - 1).to(dtype).npu()
k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
out_ref = gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
torch.npu.synchronize()
qb, kb, vb = transpose_qkv(q, k, v, S_out=128, S_kv_out=128)
mask = _padded_causal_mask(B, S, S_kv, 128, 128, q.device) if causal else _dummy_mask(q.device)
fn, _, _, um = build_flat_kernel(B, S, S_kv, 1.0 / (D**0.5), causal, N_q, N_kv, D, "float16")
out_var = transpose_out(fn(qb, kb, vb, mask))
torch.npu.synchronize()

d = (out_var.float() - out_ref.float()).abs()  # [B,S,Nq,D]
print("overall max", float(d.max()))
# per (bz, hq)
ph = d.amax(dim=(1, 3))  # [B, Nq]
print("per (bz,hq) max diff:")
for bz in range(B):
    print("  bz", bz, [f"{float(ph[bz, h]):.3e}" for h in range(N_q)])
# per s-block (64) x vid strip (32)
print("per (bz, s32, hq) — s strips of 32 rows (vid granularity within block):")
ds = d.reshape(B, 4, 32, N_q, D).amax(dim=(2, 4))  # [B, 4strips, Nq]
for bz in range(B):
    for h in range(N_q):
        row = [f"{float(ds[bz, s, h]):.2e}" for s in range(4)]
        bad = any(float(ds[bz, s, h]) > 2e-3 for s in range(4))
        if bad:
            print(f"  bz{bz} hq{h}: {row}")
# task mapping: tid = bz*(Nb*Gb*qblocks)... order bx fastest, then gb, kg, bz
# cores: cid -> tasks [cid*2, cid*2+1]
G = N_q // N_kv; Gb = G // 2; qb_ = 2
print("task->core map (tid: bz,kg,gb,bx -> hq0):")
for tid in range(40):
    bx = tid % qb_; gb = (tid // qb_) % Gb; kg = (tid // (qb_ * Gb)) % N_kv; bz = tid // (qb_ * Gb * N_kv)
    hq0 = kg * G + gb * 2
    print(f"  tid{tid:02d} core{tid//2} pos{tid%2} bz{bz} kg{kg} gb{gb} bx{bx} hq0={hq0}")
