"""rev3 split-V performance prediction, calibrated against MEASURED msprof data.

Calibration sources (all NPU msprof on 910b2, cube=20 vec=40):
  perf_tuning/round2_item2/probe_perf.json  -> p (softmax/block), t (topk+merge/block), F (per-row fixed)
  perf_tuning/round2_item2/ceiling.json     -> k1_us / k2g_us per case per task (18 cases)
"""
import math, json
CUBE, VEC = 20, 2
# ---- calibrated constants (BV=4096,K_MAX=256) ----
p4096   = 1.802     # us/block, softmax side (probe case3 & case7 nosort, 34p=61.26)
t4096   = 8.902     # us/block, topk+merge   (probe case7 329.37/37; case3 35.66/4=8.915)
F256    = 21.63     # us/row, fixed (probe case2/3/7/15/16 -- 5 independent fits, spread 21.4..22.8)
# ---- calibrated constants (BV=2048,K_MAX=128) ----
t2048   = 3.955     # us/block (probe case8 31.64/8)
F128_p8 = 21.10     # = F128 + 8*p2048 (probe case8 nosort); NOT separable -> F128 reported as a range
# ---- derived: cross-split merge topk(2*K_MAX) cost, from the Sort32/MrgSort unit census ----
#  case15: block topk(4096,K=256)=128 Sort32reps+43 MrgSort=171u ; merge topk(512,K=256)=16+5=21u
#  t4096 covers 171+21=192u -> 1u = t4096/192
u256    = t4096/192.0
t_mrg256= 21*u256                       # ~0.97 us
#  case08: block topk(2048,K=128)=64+21=85u ; merge topk(256,K=128)=8+3=11u ; 96u total
u128    = t2048/96.0
t_mrg128= 11*u128                       # ~0.45 us
# K2g per-block (post round1-#2: reduce_max+argmax, NO sort)
k2g_blk = {(4096,256): 8.02, (2048,128): 4.01}   # ceiling.json case7 304.58/38 ; case13 scaling by BV

def waves(g): return max(1, -(-g//CUBE))

def geom(V,BV,B):
    nbf=V//BV; TAIL=V-nbf*BV; nbt=nbf+(1 if TAIL>0 else 0); bn=-(-B//VEC)
    S=max(1,min(CUBE//bn,nbt)); NBS=-(-nbt//S); S=-(-nbt//NBS); NBS=-(-nbt//S)
    per=[max(0,min(nbt,(s+1)*NBS)-s*NBS) for s in range(S)]
    return bn,nbf,TAIL,nbt,S,NBS,per

def tailcost(TP,K):
    # measured points: TP=256->2.53, 320->3.13, 512->1.84, 1024->4.67 (full tail incl topk)
    tab={256:2.53,320:3.13,512:1.84,1024:4.67}
    if TP in tab: return tab[TP]
    ks=sorted(tab); import bisect
    i=min(bisect.bisect_left(ks,TP),len(ks)-1); return tab[ks[max(0,i-1)]]

def k1(B,V,K,BV,S,NBS,per,ALAUNCH):
    nbf=V//BV; TAIL=V-nbf*BV; TP=((TAIL+31)//32)*32 if TAIL>0 else 0
    bn=-(-B//VEC)
    p = p4096 if BV==4096 else (F128_p8/8.0*0.0+1.802*BV/4096 if False else 1.802*BV/4096)
    t = t4096 if BV==4096 else t2048
    F = F256 if K==256 else 6.0   # F128: undetermined; case13 fit => ~4.4, case8 fit => ~13.9. use LOW (6) = optimistic-for-split
    tm= t_mrg256 if K==256 else t_mrg128
    tc= tailcost(TP,K) if TAIL>0 else 0.0
    old = waves(bn)*(F + nbf*(p+t) + tc)
    if S<2: return old, old, 1.0
    mx  = max(per)                       # longest split = critical path
    nbf_s = mx - (1 if (TAIL>0 and mx==per[-1]) else 0)
    # K1A: split-parallel, topk ONLY (no epilogue) -> fixed cost = launch only
    TA  = waves(bn*S)*(ALAUNCH + nbf_s*t + (tc if TAIL>0 else 0.0)*0.55)
    # K1B: serial per row: full softmax scan + (S-1) cross-split merges + full epilogue
    TB  = waves(bn)*(ALAUNCH + F + nbf*p + (tc if TAIL>0 else 0.0)*0.45 + S*tm)
    return old, TA+TB, old/(TA+TB)

def k2g(B,V,K,BV,S,NBS,per,ALAUNCH):
    nbt=-(-V//BV); bn=-(-B//VEC); tb=k2g_blk.get((BV,K), 8.02*BV/4096)
    old = waves(bn)*(6.0 + nbt*tb)       # K2g has no K_MAX epilogue; small fixed
    if S<2: return old, old, 1.0
    mx=max(per)
    TA = waves(bn*S)*(ALAUNCH + mx*tb)
    TB = waves(bn)*(ALAUNCH + 6.0 + S*0.10)
    return old, TA+TB, old/(TA+TB)

CASES=[(1,2,256,256,4096),(2,1,4096,256,4096),(3,8,16384,256,4096),(4,4,16384,256,4096),
 (5,7,4409,256,4096),(6,8,10240,256,4096),(7,4,152064,256,4096),(8,4,16384,128,2048),
 (9,64,4096,256,4096),(10,4,4096,256,4096),(11,8,8192,256,4096),(12,16,32768,256,4096),
 (13,6,2304,128,2048),(14,32,1024,256,4096),(15,64,152064,256,4096),(16,128,32768,256,4096),
 (17,16,152377,256,4096),(18,64,65536,256,4096)]
MEAS={2:32.52,3:64.50,7:419.47,8:52.74,15:835.72,16:426.12}   # probe_perf real_us (K1)
CEIL=json.load(open('/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4/perf_tuning/round2_item2/ceiling.json'))
CM={r['case']:r for r in (CEIL['per_case'] if isinstance(CEIL,dict) else CEIL)}

for AL in (0.0, 3.0, 6.0):
    print(f"\n================ a_launch = {AL} us/kernel (UNMEASURED parameter) ================")
    print(f"{'c':>3}{'B':>5}{'V':>8}{'K':>4}{'BV':>6}{'bn':>4}{'nbt':>4}{'S':>3}{'maxblk':>7} | {'K1old':>8}{'K1new':>8}{'x':>6}{'K1meas':>8}{'err':>7} | {'K2gold':>8}{'K2gnew':>8}{'x':>6}{'K2gmeas':>8}{'err':>7}")
    for (cid,B,V,K,BV) in CASES:
        bn,nbf,TAIL,nbt,S,NBS,per=geom(V,BV,B)
        o,n,sp=k1(B,V,K,BV,S,NBS,per,AL)
        og,ng,spg=k2g(B,V,K,BV,S,NBS,per,AL)
        m=MEAS.get(cid); me=f"{m:8.2f}{100*(o-m)/m:+6.1f}%" if m else f"{'-':>8}{'-':>7}"
        mg=CM.get(cid,{}).get('k1_us_fused'); mge=f"{mg:8.2f}{100*(o-mg)/mg:+6.1f}%" if mg else f"{'-':>8}{'-':>7}"
        k2m=CM.get(cid,{}).get('k2_us_fused'); k2e=f"{k2m:8.2f}{100*(og-k2m)/k2m:+6.1f}%" if k2m else f"{'-':>8}{'-':>7}"
        print(f"{cid:>3}{B:>5}{V:>8}{K:>4}{BV:>6}{bn:>4}{nbt:>4}{S:>3}{max(per):>7} | {o:8.1f}{n:8.1f}{sp:6.2f}{me} | {og:8.1f}{ng:8.1f}{spg:6.2f}{k2e}")

# ===================== FINAL: guarded dispatch + task-level table =====================
A_LAUNCH_GUARD = 8.0   # CONSERVATIVE default until Stage 2 measures it (see DESIGN 10.x)
SAFETY = 1.5
def guarded(B,V,K,BV):
    bn,nbf,TAIL,nbt,S,NBS,per=geom(V,BV,B)
    if S<2 or nbf<2: return S,False,"S<2 or n_blk_full<2"
    # max FULL blocks owned by any single split (tail excluded from the saving estimate)
    mbf=max(min(nbf,max(0,(s+1)*NBS)-s*NBS) for s in range(S))
    t = t4096 if BV==4096 else t2048
    tm= t_mrg256 if K==256 else t_mrg128
    saving=(nbf-mbf)*t; cost=SAFETY*(A_LAUNCH_GUARD+S*tm)
    return S,(saving>cost),f"save={saving:.1f}us vs guard={cost:.1f}us (maxblk_full={mbf})"

print("\n\n################ FINAL GUARDED DISPATCH + TASK-LEVEL PREDICTION ################")
print(f"a_launch_guard={A_LAUNCH_GUARD}us (CONSERVATIVE, unmeasured), SAFETY={SAFETY}")
print(f"{'c':>3}{'B':>5}{'V':>8}{'K':>4}{'BV':>6}{'bn':>4}{'S':>3}{'use':>5} | {'maskOLD':>8}{'maskNEW':>8}{'x':>6} | {'fusedOLD':>9}{'fusedNEW':>9}{'x':>6} | guard reason")
gm_o=[];gm_n=[];gf_o=[];gf_n=[]
for AL in (3.0,):
  for (cid,B,V,K,BV) in CASES:
    bn,nbf,TAIL,nbt,S,NBS,per=geom(V,BV,B)
    use,why=guarded(B,V,K,BV)[1],guarded(B,V,K,BV)[2]
    Se = S if use else 1
    per_e = per if use else [nbt]
    o,n,_=k1(B,V,K,BV,Se,NBS if use else nbt,per_e,AL)
    og,ng,_=k2g(B,V,K,BV,Se,NBS if use else nbt,per_e,AL)
    cm=CM.get(cid,{})
    k2m=cm.get('k2_us_mask',0.0)             # K2m NOT split (user constraint 4)
    k1m=cm.get('k1_us_mask',0.0); k1f=cm.get('k1_us_fused',0.0); k2f=cm.get('k2_us_fused',0.0)
    # anchor the model to MEASURED K1/K2g where available (model err <=1% for K1 on 5/6 probes)
    r1 = n/o                       # K1 model new/old ratio (model anchored to measurement by ratio, not absolute)
    r2 = ng/og                     # K2g model new/old ratio
    mask_old=k1m+k2m; mask_new=k1m*r1+k2m            # K2m NOT split (user constraint 4)
    fused_old=k1f+k2f;  fused_new=k1f*r1+k2f*r2
    print(f"{cid:>3}{B:>5}{V:>8}{K:>4}{BV:>6}{bn:>4}{Se:>3}{('YES' if use else 'no'):>5} | "
          f"{mask_old:8.1f}{mask_new:8.1f}{mask_old/mask_new:6.2f} | {fused_old:9.1f}{fused_new:9.1f}{fused_old/fused_new:6.2f} | {why}")
    gm_o.append(mask_old);gm_n.append(mask_new);gf_o.append(fused_old);gf_n.append(fused_new)
import statistics
def geo(a,b): return math.exp(sum(math.log(y/x) for x,y in zip(a,b))/len(a))
print(f"\ngeomean speedup  MASK  task: {geo(gm_o,gm_n):.4f}x   (18 cases)")
print(f"geomean speedup  FUSED task: {geo(gf_o,gf_n):.4f}x   (18 cases)")
print(f"min per-case speedup: mask {min(y/x for x,y in zip(gm_n,gm_o)):.3f}x(=worst regression {max(x/y for x,y in zip(gm_o,gm_n)):.3f}x slower)  fused {min(y/x for x,y in zip(gf_n,gf_o)):.3f}x")
print(f"max per-case speedup: mask {max(y/x for x,y in zip(gm_o,gm_n)):.2f}x  fused {max(y/x for x,y in zip(gf_o,gf_n)):.2f}x")
print("split-path cases:",[c[0] for c in CASES if guarded(c[1],c[2],c[3],c[4])[1]])
