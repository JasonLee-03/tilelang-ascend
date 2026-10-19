import sys, re, collections
sys.path.insert(0,'/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4')
sys.path.insert(0,'/tmp/opencode')
import tilelang
from tilelang import language as T
from r4_fused_sampling import R4_PASS_CONFIGS, VEC_NUM, _aligned_topk
from r4_splitv_ub_probe import ub_usage, UB_LIMIT

@tilelang.jit(pass_configs=R4_PASS_CONFIGS)
def k1a_v2(B: int, V: int, K_MAX: int, BV: int, S: int, NBS: int):
    """Single-loop variant: uniform block space [0, n_blk_total); index n_blk_full IS the tail."""
    n_blk_full = V // BV
    TAIL = V - n_blk_full * BV
    TAIL_PAD = ((TAIL + 31) // 32) * 32
    n_blk_total = n_blk_full + (1 if TAIL > 0 else 0)
    b_num = T.ceildiv(B, VEC_NUM); AT = _aligned_topk(K_MAX); grid = b_num * S
    @T.prim_func
    def main(logits: T.Tensor([B, V], "float32"), pv: T.Tensor([B * S, K_MAX], "float32")):
        with T.Kernel(grid, is_npu=True) as (cid, vid):
            task = cid * VEC_NUM + vid
            row = task // S
            sid = task % S
            if row < B:
                l_dst = T.alloc_shared([AT], "float32"); l_val = T.alloc_shared([K_MAX], "float32")
                merged = T.alloc_shared([2 * K_MAX], "float32"); m_dst = T.alloc_shared([AT], "float32")
                run_v = T.alloc_shared([K_MAX], "float32")
                T.tile.fill(run_v, -T.infinity("float32"))
                if n_blk_total > 0:
                    z_ub = T.alloc_shared([BV], "float32")
                    if TAIL > 0:
                        ztp = T.alloc_shared([1, TAIL_PAD], "float32")
                        ztt = T.alloc_shared([TAIL_PAD], "float32")
                    for lb in T.serial(NBS):
                        b = sid * NBS + lb
                        if b < n_blk_full:
                            T.copy(logits[row, b * BV], z_ub)
                            T.tile.topk(l_dst, z_ub, K_MAX, BV)
                            T.tile.gather_mask(l_val, l_dst, "P0101")
                            T.copy(run_v, merged[:K_MAX]); T.copy(l_val, merged[K_MAX:])
                            T.tile.topk(m_dst, merged, K_MAX, 2 * K_MAX)
                            T.tile.gather_mask(run_v, m_dst, "P0101")
                        if TAIL > 0:
                            if b == n_blk_full:
                                T.copy(logits[row, n_blk_full * BV], ztp, pad_value=-T.infinity("float32"))
                                T.copy(ztp[0, :TAIL_PAD], ztt)
                                T.tile.topk(l_dst, ztt, K_MAX, TAIL_PAD)
                                T.tile.gather_mask(l_val, l_dst, "P0101")
                                T.copy(run_v, merged[:K_MAX]); T.copy(l_val, merged[K_MAX:])
                                T.tile.topk(m_dst, merged, K_MAX, 2 * K_MAX)
                                T.tile.gather_mask(run_v, m_dst, "P0101")
                T.copy(run_v, pv[row * S + sid, :])
    return main

def geom(V, BV, B, cube=20):
    nbf = V // BV; TAIL = V - nbf*BV; nbt = nbf + (1 if TAIL>0 else 0)
    bn = -(-B//VEC_NUM)
    S = max(1, min(cube//bn, nbt))
    NBS = -(-nbt//S); S = -(-nbt//NBS); NBS = -(-nbt//S)
    per = [max(0, min(nbt,(s+1)*NBS) - s*NBS) for s in range(S)]
    return bn, nbf, TAIL, nbt, S, NBS, per

CASES=[(1,2,256,256,4096),(2,1,4096,256,4096),(3,8,16384,256,4096),(4,4,16384,256,4096),
 (5,7,4409,256,4096),(6,8,10240,256,4096),(7,4,152064,256,4096),(8,4,16384,128,2048),
 (9,64,4096,256,4096),(10,4,4096,256,4096),(11,8,8192,256,4096),(12,16,32768,256,4096),
 (13,6,2304,128,2048),(14,32,1024,256,4096),(15,64,152064,256,4096),(16,128,32768,256,4096),
 (17,16,152377,256,4096),(18,64,65536,256,4096)]
print("FULL 18-CASE SPLIT TABLE (cube_core_num=20, VEC_NUM=2)")
print(f"{'c':>3}{'B':>5}{'V':>8}{'K':>5}{'BV':>6}{'bn':>4}{'nbf':>4}{'TAIL':>6}{'nbt':>4}{'S':>3}{'NBS':>4}{'grid':>5}{'wav':>4}  per-split-blocks  dispatch")
worst=0; worstc=None
for (cid,B,V,K,BV) in CASES:
    bn,nbf,TAIL,nbt,S,NBS,per = geom(V,BV,B)
    grid=bn*S; wav=-(-grid//20)
    disp = "SPLIT-V" if S>=2 else "CURRENT (S=1)"
    print(f"{cid:>3}{B:>5}{V:>8}{K:>5}{BV:>6}{bn:>4}{nbf:>4}{TAIL:>6}{nbt:>4}{S:>3}{NBS:>4}{grid:>5}{wav:>4}  {str(per)[:44]:<44} {disp}")
    if S>=2:
        src=k1a_v2(B,V,K,BV,S,NBS).get_kernel_source()
        tot,nb,reg=ub_usage(src)
        pcs=re.findall(r'TopK<\w+>\([^;]*?,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)',src)
        bad=[(K_,rt,an) for K_,rt,an in pcs if int(rt)*32-int(an)!=0]
        u2u=re.findall(r'copy_ub_to_ub<[^,]+,[^,]+,\s*(\d+)>',src)
        badu=[c for c in u2u if (int(c)*4)%32!=0]
        st = "OK" if (tot<=int(reg or UB_LIMIT) and not bad and not badu) else "FAIL"
        print(f"      -> K1A_v2 UB={tot}/{reg} ({100*tot/int(reg or UB_LIMIT):.2f}%) topk_padCount={'ALL0' if not bad else bad} ub2ub_counts={sorted(set(u2u))} non32B={badu} {st}")
        if tot>worst: worst,worstc=tot,cid
print(f"\nWORST K1A_v2 UB = {worst} (case{worstc}) / {UB_LIMIT} = {100*worst/UB_LIMIT:.2f}%")
