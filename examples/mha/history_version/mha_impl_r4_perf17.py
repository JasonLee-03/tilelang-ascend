"""MHA (Multi-Head Attention) forward, BSHD layout, Expert mode (DESIGN rev4).

y = softmax(Q @ K^T * s) @ V,   s = scaleValue if scaleValue > 0 else 1/sqrt(D)
is_causal: scores[..., i, j] = -inf for j > i + (S_kv - S)  (bottom-right aligned)
All-masked rows (only reachable out-of-domain) yield 0 via the sumexp + 1e-30 guard.

rev4 (2026-09-22; design = DESIGN.md rev4, S1 = Form B->A mma-ization of the
prefill D=256 path; base = rev3 final bbfcbcaf, cann-bench 71.70):
- THREE-way C-scope dispatch (§5.2/§5.6): Form A (S>16 & D<=128, unchanged) /
  Form B-mma (S>16 & D=256, NEW: manual K/N-split T.mma pipeline replacing
  gemm_v0's per-step PipeBarrier<PIPE_M> drain — case9 scalar_mte1_stall ~78%
  wall, §S4-3-2) / Form B-gemm_v0 (S<=16 decode all D, fallback, source
  verbatim unchanged => decode-domain IR byte-identical by construction).
- Form B-mma: GEMM1 = 2x k-chunk (KC=128) legacy chain mma(init=True->False)
  into l0c_s[side] (probe P3 16/16 BITEXACT vs gemm_v0 kL0Size=128); GEMM2 =
  PLAN B N-split — narrow l0c_o [2,32,NC] pool + per-chunk colseg FIX write
  to w3 (plan-a mma C-region column slice crashed the aicore "CUBE
  instruction parameter is invalid", probe P2; plan b = sparse_flash_mla:
  673-681 precedent, probe P2b/P5 16/16 BITEXACT vs gemm_v0 full-N). l0b_pool
  [2,128,128] 64KB SHARED (GEMM1 K^T transpose / GEMM2 V direct, phase
  time-multiplexed, token stream continues across phases — §4.5-d; probe P4
  real-compiled readout: L0A 32 / L0B 64 exact-top / L0C 64 plan b / L1 152 /
  UB 122 KB, all within limits). Fine-grained tokens SIG_L0KC 0,1 / SIG_L0AO
  2,3 / SIG_L0AS 4,5 (MTE1<->M) + SIG_L0C 0,1 (M<->FIX shared stream), init
  13 set / destroy 12 wait (§7.2-2a accounting table; no dangling tokens, no
  conditional DMA — #R5-3 discipline). Batch protocol = #R3-1/#R5-1 mirror
  (single_kv_block task batch) + Form A generic isomorph (multi-kv);
  emit_v V-scope + all workspaces/SEM protocol ZERO change (§3.4).
- Anchors (§9.2): A = vs Developer final 6505 on L0 #1-9 (Form A domain, IR
  unchanged => 9/9 BITEXACT hard guard); B = vs rev3 final bbfcbcaf on the
  Form B-mma domain (diagnostic: BITEXACT expected via K-ascending chained
  fp32 accumulation + column-independent N-split + emit_v verbatim, NOT
  promised across C++ templates; mismatch != FAIL, §9.2 discipline).

rev4r12 (2026-10-05; perf_session=12 net state; design semantics unchanged):
- #R12-2 (A10, USER-AUTHORIZED framework change): gemm_v0 template opt-in
  per-buffer MTE2->MTE1 sync (common.h 9th default template param kMTE2Sync
  + ascend.py mte2_sync kwarg; default path byte-identical — 17/17 IR guard
  + 90/90 gemm_v0 UT consumer smoke). kv_db decode domain (S<=16, D=256,
  multi-block; case15/18) rebuilt on the #R11-1 K-double-slot skeleton:
  PRE-CALL K(idx+1) prefetch (global guard idx+1<eff — crosses rounds, the
  #R11-1 round-end site is redundant and removed) + per-buffer SetFlag
  (SIG_KDB0/1 parity ids, runtime-id Set per form_a SIG_L0AB+side precedent,
  strict 1:1 Set->Wait alternation) + parity-unrolled G1 call sites binding
  the compile-time mte2_sync + G2 V/P FIFO-prefix flag SIG_VP (Set after
  P copy covers V+P) + V single-slot after-return prefetch (S5) with
  round-head V priming at the G1 batch tail (S3) + Q(t) relocation into
  S0 (t==0) / S4 (task-end, Q BEFORE K' — FIFO prefix must cover both
  G1'(0) operands). Breaks the #R11-1 conservation law (pipe-level prologue
  drain absorbed prefetches): MTE2 streams continuously through gemm calls.
  msprof median (fresh same-session baseline): case15 508.11->463.58us
  (-8.76%) / case18 508.57->465.36 (-8.50%); op duty signature 68.3->73.7%
  (+5.4pp), mte2 busy rate 75.0->75.8 GB/s/core (zero reorder tax vs the
  r11 S2 -2.7%), scalar_t -41.5us = wall delta, cube_t/vec_t/GM bytes
  exactly unchanged; 7 guard cases in-band (case20 +2.8% on IR-identical
  source = drift band, session-8 precedent). Residual (documented): G2-phase
  MTE2 idle during SEM_WS2 protocol waits — v_l1 single slot + L1 budget
  400/511.75KB forbid a second V slot at bN=256 => upgrade arm (bN=128
  k/v dual DB, 268KB) handed to a future session. Full precision battery
  green on the adopted state (probe 22+12 BITEXACT vs perf16, anchors 9/9
  +14/14, stress 4+17 cfg x12 nondet=0, l0/l1/coverage/all).
- ADOPTION CARRIER = FRAMEWORK: this state requires the paired common.h /
  ascend.py opt-in param (wheel repackage + eval-env reinstall for cann-bench
  re-scoring — Orchestrator side; TL_TEMPLATE_PATH resolution: pip package
  carries tilelang/src/tl_templates per setup.py).

rev4r8 (2026-09-29/30; perf_session=8 net state; design semantics unchanged):
- #R8-2/#R8-3: flat (task,kv) continuous token stream replaces the per-task
  nested [task][outer-batch] loops in BOTH multi-block generic domains
  (form_b_mma case9/4 + form_a case7/8/3/2/5; flat_v = (S>16) & !single).
  NUM_STAGES=8 slots cross task boundaries without draining; runtime-
  nonuniform causal eff_kv mapped by a sequential-walk state machine
  (total_flat walk + batch-head snapshot + per-slot re-walk, C/V dual-side
  identical formulas, no ws transmission). Task-switch Q/l0a_s rounds fully
  inside the kv==0 condition (co-conditioned DMA+flags, net-0 per task);
  V task-state machine per §S4-3-5 (phase2 kv0 resets acc_o/sumexp, phase1
  kv0 m_i/row_thresh exception, phase2 kv-last final normalization with the
  +1e-30 guard verbatim + SIG_S_HALF closed-loop writeback). Form A deltas:
  l0a_s reload `i<2 or kv_g1<2` (P2 pool-alias batch-head reload x task
  starting mid-batch) + q_l1 return on the task-LAST slot (MTE1 in-order
  coverage of cross-batch q_l1 re-reads — design-time race fix).
- Per-slot op sequences VERBATIM vs legacy => BITEXACT (probes: fbm 16/16 +
  form_a 15/15 + guards 7/7; anchors A 9/9 + B 14/14; stress x12 nondet=0);
  decode x5 + single-block x5 + fbm-single IR byte-identical vs rev4 base.
  msprof median: case9 -8.1% / case4 -20.7% / case2 -33.1% / case3 -21.6% /
  case5 -18.1% / case7 -7.4% / case8 -6.3%, guards zero-regression;
  mechanism = C wait_id5 round-boundary -95~97% both domains with all
  real-work pipes flat (pure structural gain: rounds ~103->~58/core, batch
  depth 4.5->8). #R8-1 (L0B 4x16KB slot rotation) rolled back — mte1_stall
  is a four-phase-chain symptom, not a lever (evidence-closed, variant
  archived /tmp/opencode/r8_probes/mha_r81_edit_v1.py).

rev3 Expert implementation (2026-09-19; design = DESIGN.md rev3, semantic
baseline = history_version/mha_impl_developer_final_6505.py, cann-bench 65.05):
- Expert frame: `T.Kernel(core_num, is_npu=True) as (cid, vid)` — NO threads
  parameter (§2.2-b/§8.6-1); vid extent 2 is the structural blockIdx.y binding
  (cv_1_2). Four pass_configs all False (§4.7): manual T.Scope("C"/"V"),
  manual set_flag/wait_flag + set_cross_flag/wait_cross_flag, explicit
  alloc_L1/alloc_ub/alloc_L0A/L0B/L0C, explicit GM workspace x3
  (workspace_idx=[4,5,6], [cid, stage] private indexing).
- Dual C-scope forms (host predicate dispatch, §5.2/§6.1), V-scope shared
  via a @T.macro emitter (two self-contained prim_funcs built by a Python-level
  branch — TVMScript parse-time scoping, rev2-A1 lesson; plain-Python emitters
  cannot host T.serial frames, @T.macro can — probed):
    Form A (prefill S>16 & D<=128): T.mma + L0A/L0B/L0C double slots
      (side=i%2 manual ping-pong) + ZN/NZ annotate_layout + fine-grained
      pipe-pair flags (mha_sink_fwd_bhsd GEMM1/GEMM2 pattern).
      P2 (D==bN==128): GEMM2 shares the GEMM1 L0 pool (same shapes,
      mha_sink isomorph); P1 (D=64): separate l0a_o/l0b_o/l0c_o pool.
      single_kv_block: task-batched like Form B (#R5-1 session-5) — see the
      NUM_STAGES bullet below. PLUS (#R5-2 session-6): O direct write —
      GEMM2's FIX copies l0c_o straight into Output[bz, s0:s0+valid_m, by, :]
      (runtime row slice; gate probe 8/8 int-exact BITEXACT), deleting the
      w3 GM round-trip, the V phase-2 acc_o pass and the WS3 SEM pair; the
      fixpipe fp32->dtype RNE cast is bit-identical to the removed V-side
      vcast (#R3-3 probe 11/11). Form B keeps w3 + phase-2 (parse-time
      `if not form_a:` gate -> Form B IR byte-identical).
    Form B (decode S<=16 all D + prefill D=256): T.gemm_v0 (internal L0
      management, kL0Size three-tier inherited: 32/64/128), barrier-free
      segment ordering (#R4-1 session-4: gemm_v0's self-balanced
      prologue/epilogue pipe flags — MTE2_MTE1/FIX_M in, MTE1_MTE2/M_FIX
      out, common.h L1305-L1371 — cover every segment-boundary hazard; the
      rev3 barrier_all coarse sync was the redundant-global-sync
      antipattern: AIC pipe busy-sum ~95% with zero overlap, msprof op
      evidence case9).
- NUM_STAGES=8 batched CV pipeline: outer T.serial(eff_num_outer) + inner
  T.serial(batch_iters) runtime bounds, identical formulas on C/V scopes
  (§7.2-5 flag-pairing hard constraint); CROSS_INTERVAL=1 cross-core
  semaphore batching (gqa_fwd_varlen iter3 values). T.Pipelined is BANNED
  (§2.2-d/R9 guardrail); manual side=i%2 ping-pong expresses the pipeline.
  Batch dimension: kv-block when kv_iters>1; TASK when single_kv_block
  (kv_iters==1 would otherwise collapse batch_iters to 1 and leave the
  io_buf[2]/L0 double slots with nothing to ping-pong — every task ran a
  strictly serial C-QK -> V-softmax -> C-PV -> V-epi chain). #R3-1 added the
  task batch for Form B; #R5-1 (session-5) added the Form A C-scope mirror and
  dropped the `not form_a` half of the emit_v dispatch gate, so BOTH forms now
  batch. Form A slot-level deltas vs its generic body: the Q GM->L1 load moves
  inside the slot loop and q_l1->l0a_s[side] is issued every slot (Q is
  task-dependent), with the SIG_Q_L1 return moved to right after that single
  MTE1 read (per-slot token balance identical to the generic per-task one).
- vid row split: each AIV handles half_M = block_M//2 rows; writeback
  rows_vid = min(half_M, max(0, valid_m - vid*half_M)) with a runtime
  `if rows_vid > 0` guard (§5.4). Deferred-rescale batched softmax
  (r_factors stored in phase 1, exp in phase 2 — bit-identical to the
  per-iteration Developer order, §9.3.3-6).
- Precision red lines inherited verbatim (§3.2.1 mapping, 12/12 items):
  score double-rounding golden grid (w1 fp32 mainline -> V-side r16(qk) ->
  *s fp32 -> r16(scores)), single-KV-block pre-normalization specialization
  (kv_iters==1), sumexp+1e-30 all-masked-row guard, bottom-right-aligned
  pre-exp -inf mask chain (2D compare/select + row_thresh task-level hoist +
  fully-visible runtime skip), causal KV runtime trim (eff_kv_iters), Fixed
  Core (core_num device query + balanced task loop), bM/bN/gemm1_kL0
  three-tier system (Developer session-1/2 measured hard-limit solutions),
  BSHD row-stride DMA (zero host tensor ops, §4.8), valid_m/valid_n runtime
  tail extents with need_clear zeroing.
- Implementation-level buffer adaptations vs the §4.3 table (shape algebra
  forced, semantics unchanged; recorded in debug_log.md rev3-attempt1):
    io_buf3 [2,hM,D] fp32: dedicated w3 staging — ws3 vid-half slices are
      [hM,D] while io_buf slots are [hM,bN]; D!=bN in 5/6 config classes.
      Aliased to io_buf (same flag ids, token-balanced) when D==bN (P2).
    r_2d [hM,D] fp32: phase-2 rescale / final-div broadcast scratch on the
      acc_o column shape (Developer r_2d analog); buf_2d [hM,bN] cannot
      serve when D!=bN. Aliased to buf_2d when D==bN (both pure-V scratch,
      sequential V-pipe windows — §8.6-2 applies to cross-core copy
      destinations only).
    rr_ub [hM,bN] dtype: double-rounding round-trip scratch, kept SEPARATE
      from acc_s_half — acc_s_half is the MTE3 cross-core copy source under
      the SIG_S_HALF ownership loop; using it as round-trip scratch outside
      the loop would race the previous iteration's MTE3 read (§8.6-2 alias
      ban).
- Virtual-channel ban (§8.6-3/R12): no T.copy(L0C->UB) / T.copy(UB->L1)
  anywhere; CV exchange is the explicit 3-workspace GM path only (the sole
  physical route under this framework, §8.2 source evidence).
- Probes (scratch, /tmp/opencode/rev3_probes/): R10 Expert BSHD family
  16/16 bit-exact (Form A annotate ZN/NZ + mma + transpose + fine flags /
  Form B gemm_v0 + barrier_all; S=1/30/64/90/128 + divisible; need_clear
  zeros via host ws readback; vid-half writeback + rows_vid=0 guard via
  padded-sentinel Out; cross-task ws reuse; transpose-sensitive identity K);
  R15 mask-chain V-scope isolation — see debug_log.md.
"""

import tilelang
import torch
from tilelang import language as T
from tilelang.intrinsics import make_zn_layout, make_nz_layout

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,  # manual T.Scope("C"/"V")
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,  # manual SEM 0-5
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,  # manual SIG flag matrix
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,  # explicit alloc_* declarations
}

NUM_STAGES = 8  # CV pipeline depth (even constraint asserted; Stage-3 tunable 4-16)
CROSS_INTERVAL = 1  # cross-core sync interval (gqa iter3 verified; Stage-3 tunable 2-4)


def _cube_core_num() -> int:
    """Physical cube-core count (HOST METADATA query, zero tensor data touched;
    precedent: examples/deepseek_v4/lightning_indexer.py:14-21)."""
    try:
        import torch_npu

        return int(torch_npu.npu.get_device_properties(0).cube_core_num)
    except Exception:  # noqa: BLE001 - conservative 910B-class fallback
        return 20


@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6], pass_configs=pass_configs)
def mha_kernel(
    B,
    S,
    S_kv,
    N,
    D,
    dtype="float16",
    scaleValue=-1.0,
    is_causal=False,
):
    accum_dtype = "float"
    assert NUM_STAGES % 2 == 0, "NUM_STAGES must be even (io_buf/L0 double-slot rotation, §8.6-7)"
    # ---- Host-side pure-Python config derivation (no tensor data touched, §4.8) ----
    sm_scale = scaleValue if scaleValue > 0 else (1.0 / D) ** 0.5
    # Tile three-tier system: verbatim inheritance of the Developer final state
    # (mha.py:120-179 probe-verified hard-limit solutions; see its docstring and
    # perf_tuning/optimization_log.md for the full evidence index).
    block_M = 16 if S <= 16 else (32 if D == 256 else 64)
    if S <= 16:
        # bN=512 (D<=128, S_kv>256): GEMM1 transpose_B B-tile 512*kL0(32)*2B =
        # 32768 == the 4-step ping-pong L0B budget; D=256 excluded by L1 (536KB).
        block_N = 512 if (D <= 128 and S_kv > 256) else (256 if S_kv > 128 else 128)
    else:
        block_N = 128
    gemm1_kL0 = (
        32
        if (S <= 16 and D <= 128 and S_kv > 256)  # bN=512: 512*32*2 = 32768 <= 32KB
        else (64 if (S <= 16 and D == 256 and S_kv > 128) else 128)  # bN=256 D=256 / rest
    )
    # ---- Expert additions (§5.2) ----
    form_a = (S > 16) and (D <= 128)  # Form A: mma full pipeline (rev4 unchanged)
    # ---- rev4 three-way dispatch (§5.2/§5.6): mutually exclusive & exhaustive ----
    form_b_mma = (S > 16) and (D == 256)  # Form B-mma: manual K/N-split mma pipeline (rev4)
    # else (S <= 16): Form B-gemm_v0 decode fallback (all D, source verbatim unchanged)
    KC = 128  # Form B-mma k-chunk width (compile-time constant; D=256 -> 2 chunks)
    NC = 128  # Form B-mma n-chunk width (compile-time constant; D=256 -> 2 chunks)
    if form_b_mma:
        assert D % KC == 0 and D % NC == 0, "K/N-split requires D divisible by chunk width (D=256 closed, §3.5.1)"
    half_M = block_M // 2  # vid half rows (structural cv_1_2: 8/16/32)
    causal_offset = S_kv - S  # bottom-right alignment (>= 0 when causal, asserted)
    need_bound_mask = S_kv % block_N != 0  # tail-column clamp (compile-time)
    need_mask = is_causal or need_bound_mask
    # Unified threshold offset: causal -> bottom-right offset; non-causal -> big
    # constant so T.min(row + offset_eff, S_kv-1) collapses to S_kv-1.
    offset_eff = causal_offset if is_causal else 2**30
    s_blocks = (S + block_M - 1) // block_M
    kv_iters = (S_kv + block_N - 1) // block_N  # compile-time constant
    single_kv_block = kv_iters == 1  # i.e. S_kv <= block_N (predicate-move inherited)
    # ---- #R8-2/#R8-3 (session-8 Stage-3) flat (task,kv) token-stream domain:
    # generic multi-kv prefill of BOTH mma forms (mission domain order: r8-2 =
    # form_b_mma (case9/4, ADOPTED -7.96%/-20.7%), r8-3 = form_a extension
    # (case7/8/3/2/5)). Single-block (task-batched branch) and decode fallback
    # (S<=16) keep their branches verbatim => IR byte-identical for those
    # domains. Parse-time closure flag consumed by emit_v (three-way V
    # dispatch) and by both C-scope generic branches.
    flat_v = (S > 16) and not single_kv_block
    # ---- #R11-1/#R12-2: decode D=256 K-side L1 double-buffer domain ----
    # form_b_gemm_v0 fallback only (S<=16), D==256, multi-kv-block (kv_iters>=2
    # => case15/18 + L0/L1 sentinels). L1: q8+k[2]256+v128+p8 = 400KB <=
    # 511.75KB (DESIGN §4.5-a D2 row + single-side DB; compile-readout gate).
    # #R12-2 (A10, session-12): gemm_v0 opt-in per-buffer MTE2_MTE1 sync
    # (mte2_sync kwarg; framework common.h/ascend.py opt-in param) breaks the
    # #R11-1 conservation law: the prologue waits ONLY the caller's FIFO-prefix
    # flag, so later-enqueued prefetches keep streaming instead of being
    # absorbed by the pipe-level drain. C-side intra-core scheduling ONLY:
    # emit_v / SEM matrix / single-block path / D<=128 domains byte-identical
    # (IR 15/17 guards).
    kv_db = (S <= 16) and (D == 256) and not single_kv_block
    # ---- #R12-2 (A10): opt-in per-buffer MTE2->MTE1 event ids (kv_db only) ----
    # Type MTE2_MTE1; usable ids 0-5 (AscendC forbids 6/7 in static-Tensor
    # scenarios, SetFlag/WaitFlag(ISASI) doc). All kv_db gemm_v0 calls are
    # opt-in => the default-mode prologue never self-sets MTE2_MTE1 id0 in
    # this kernel => ids 0/1/2 are collision-free (other HardEvent types are
    # independent flag spaces; C-core and V-core are physically separate).
    # Discipline: strict Set->Wait 1:1 alternation per id (outstanding <=1)
    # -- safe under both 1-bit and counting flag semantics; runtime-id Sets
    # follow the form_a SIG_L0AB+side production precedent.
    SIG_KDB0 = 0  # K slot 0 resident (G1 operand, even idx)
    SIG_KDB1 = 1  # K slot 1 resident (G1 operand, odd idx)
    SIG_VP = 2  # V+P resident (G2 operand; Set after P copy covers V by FIFO prefix)
    block_num = B * N * s_blocks
    core_num = min(block_num, _cube_core_num())  # Fixed Core (inherited #5)
    q_tasks = block_num // core_num  # base tasks per core
    r_tasks = block_num % core_num  # first r_tasks cores take one extra

    # ---- Cross-core semaphore IDs (AIC <-> 2xAIV group, mode=2; §7.2-1) ----
    SEM_WS1_C2V = 0  # w1 scores ready: Cube -> Vector
    SEM_WS1_V2C = 1  # w1 consumed: Vector -> Cube
    SEM_WS2_V2C = 2  # w2 softmax P ready: Vector -> Cube
    SEM_WS2_C2V = 3  # w2 consumed: Cube -> Vector
    SEM_WS3_C2V = 4  # w3 PV ready: Cube -> Vector
    SEM_WS3_V2C = 5  # w3 consumed: Vector -> Cube

    # ---- C-side intra-core event IDs (per directed pipe pair; Form A; §7.2-2) ----
    # MTE2 <-> MTE1
    SIG_K_L1 = 0
    SIG_P_L1 = 1
    SIG_V_L1 = 2
    SIG_Q_L1 = 3
    # MTE1 <-> M
    SIG_L0AB = 0  # double slots 0/1 (Form A)
    # MTE1 <-> M (Form B-mma fine-grained tokens, §7.2-2a; ids 0-5 <= 8 hw limit;
    # no gemm_v0 in the Form B-mma prim_func => no L0AB_EVENT=0 collision, §3.4)
    SIG_L0KC = 0  # l0b_pool chunk 0/1 — GEMM1 kc / GEMM2 nc SHARED stream (§4.5-d-2)
    SIG_L0AO = 2  # l0a_o side 0/1 (GEMM2 A ping-pong)
    SIG_L0AS = 4  # l0a_s chunk 0/1 (GEMM1 A, task/slot-level preload)
    # M <-> FIX
    SIG_L0C = 0  # double slots 0/1

    # ---- V-side intra-core event IDs (per AIV; §7.2-3) ----
    # MTE2 <-> V
    SIG_IO_UB = 0  # io_buf (w1) double slots 0/1
    SIG_IO3 = 2  # io_buf3 (w3) double slots 2/3 (only when D != block_N)
    # MTE3 <-> V
    SIG_S_HALF = 0  # acc_s_half / out_half staging ownership

    # w3 staging + rescale scratch: shape-forced adaptation (§4.3 note; docstring).
    # D == block_N -> alias io_buf / buf_2d (same shapes; pure sequential reuse,
    # token-balanced with shared flag ids); D != block_N -> dedicated buffers.
    io3_is_alias = block_N == D
    sig_io3 = SIG_IO_UB if io3_is_alias else SIG_IO3

    # ==================== Shared V-scope emitter (@T.macro; §6.1) ====================
    @T.macro
    def emit_v(
        io_buf,
        io3,
        work_ub,
        buf_2d,
        rt_2d,
        r_2d,
        mask_2d,
        col_pos,
        row_thresh,
        rr_ub,
        acc_s_half,
        acc_o,
        out_half,
        m_i,
        m_i_prev,
        sumexp,
        r_factors,
        sumexp_is,
        Output,
        workspace_1,
        workspace_2,
        workspace_3,
        cid,
        vid,
        my_start,
        my_count,
        sig_io3_base,
    ):
        # init (§7.2-4, mha_sink:379-386 isomorph): pretend workspaces released
        T.set_cross_flag("MTE2", SEM_WS1_V2C)
        T.set_cross_flag("MTE2", SEM_WS3_V2C)
        T.set_flag("V", "MTE2", SIG_IO_UB)
        T.set_flag("V", "MTE2", SIG_IO_UB + 1)
        if not io3_is_alias:
            T.set_flag("V", "MTE2", sig_io3_base)
            T.set_flag("V", "MTE2", sig_io3_base + 1)
        T.set_flag("MTE3", "V", SIG_S_HALF)

        if single_kv_block:
            # ---- #R3-1/#R5-1 task-batched single-KV-block V path (Form A + B) ----
            # Slot i <-> task my_start + kb*NUM_STAGES + i; per-slot op sequence
            # identical to the generic per-task flow (bit-exact); num_tb/
            # batch_tasks from the SAME formulas as the C mirror (§7.2-5).
            # #R5-1: the `and not form_a` gate is dropped — [IMPL-#R3-1]④ kept
            # Form A on the generic path ONLY because the C-side batch mirror did
            # not exist yet (protocol mismatch -> deadlock), not because of a
            # measured-benefit verdict (final_report.md:239 was predictive). The
            # Form A C mirror now exists (the `if single_kv_block:` branch of the
            # Form A C-scope task loop), so both forms share this path.
            # `single_kv_block and not form_a` == `single_kv_block` whenever
            # form_a is False => Form B IR is provably unchanged.
            num_tb = T.ceildiv(my_count, NUM_STAGES)
            for kb in T.serial(num_tb):
                _remaining_t = my_count - kb * NUM_STAGES
                batch_tasks = T.if_then_else(_remaining_t < NUM_STAGES, _remaining_t, NUM_STAGES)

                # ---- Phase 1 batch: per-task softmax -> w2 slots ----
                T.wait_cross_flag(SEM_WS2_C2V)
                for i in T.serial(batch_tasks):
                    task = my_start + kb * NUM_STAGES + i
                    bx = task % s_blocks
                    s0 = bx * block_M  # loop-body scope (rev2-A1: cross-block defs banned)
                    io_side = i % 2
                    T.tile.fill(sumexp, 0.0)
                    T.tile.fill(m_i, -(2**30))
                    if need_mask:
                        T.tile.arith_progression(row_thresh, s0 + vid * half_M + offset_eff, 1, half_M)
                        T.tile.min(row_thresh, row_thresh, T.cast(S_kv - 1, accum_dtype))

                    # w1[cid, i, vid half] -> io_buf -> work_ub (fp32 mainline)
                    T.wait_flag("V", "MTE2", SIG_IO_UB + io_side)
                    if i % CROSS_INTERVAL == 0:
                        T.wait_cross_flag(SEM_WS1_C2V)
                    T.copy(
                        workspace_1[cid, i, vid * half_M : vid * half_M + half_M, :],
                        io_buf[io_side, :, :],
                    )
                    T.set_flag("MTE2", "V", SIG_IO_UB + io_side)
                    T.wait_flag("MTE2", "V", SIG_IO_UB + io_side)
                    T.copy(io_buf[io_side, :, :], work_ub)
                    T.set_flag("V", "MTE2", SIG_IO_UB + io_side)

                    T.copy(m_i, m_i_prev)  # history max BEFORE reduce (§10.2 lesson)
                    # golden score-grid alignment (§3.2.1-⑦): identical op order
                    T.copy(work_ub, rr_ub)  # r16(qk)
                    T.copy(rr_ub, work_ub)
                    T.tile.mul(work_ub, work_ub, sm_scale)  # * s in fp32
                    T.copy(work_ub, rr_ub)  # r16(scores) = golden grid
                    T.copy(rr_ub, work_ub)

                    if need_mask:  # noqa: SIM102 - parse-time outer / runtime inner (rev2-A1)
                        # single block => idx == 0; s0-block-uniform -> both vids
                        # take the same branch (flag pairing safe; pure-V segment).
                        if block_N - 1 > T.min(s0 + offset_eff, S_kv - 1):
                            T.tile.arith_progression(col_pos, 0, 1, block_N)
                            T.tile.broadcast(buf_2d, col_pos, axis=0)  # [bN] -> [hM,bN]
                            T.tile.broadcast(rt_2d, row_thresh, axis=1)  # [hM] -> [hM,bN]
                            T.tile.compare(mask_2d, buf_2d, rt_2d, "LE")
                            T.tile.select(
                                work_ub,
                                mask_2d,
                                work_ub,
                                -T.infinity(accum_dtype),
                                "VSEL_TENSOR_SCALAR_MODE",
                            )

                    T.reduce_max(work_ub, m_i, dim=-1)
                    T.tile.max(m_i, m_i, m_i_prev)
                    T.tile.sub(r_factors[i, :, :], m_i_prev, m_i)
                    T.tile.broadcast(buf_2d, m_i, axis=1)
                    T.tile.sub(work_ub, work_ub, buf_2d)
                    T.tile.exp(work_ub, work_ub)  # P = exp(s - m_new) in [0, 1]
                    T.reduce_sum(work_ub, sumexp_is[i, :, :], dim=-1)

                    # pre-normalization specialization (§3.2.1-⑤): P_hat before
                    # the dtype rounding = golden one-shot softmax grid.
                    T.tile.add(sumexp, sumexp_is[i, :, :], 1e-30)
                    T.tile.broadcast(buf_2d, sumexp, axis=1)
                    T.tile.div(work_ub, work_ub, buf_2d)

                    # P -> w2 slot i (SIG_S_HALF ownership closed loop)
                    T.wait_flag("MTE3", "V", SIG_S_HALF)
                    T.copy(work_ub, acc_s_half)  # fp32 -> dtype (P native grid)
                    T.set_flag("V", "MTE3", SIG_S_HALF)
                    T.wait_flag("V", "MTE3", SIG_S_HALF)
                    T.copy(
                        acc_s_half,
                        workspace_2[cid, i, vid * half_M : vid * half_M + half_M, :],
                    )
                    T.set_flag("MTE3", "V", SIG_S_HALF)
                    if (i + 1) % CROSS_INTERVAL == 0 or i == batch_tasks - 1:
                        T.set_cross_flag("MTE3", SEM_WS2_V2C)
                T.set_cross_flag("MTE2", SEM_WS1_V2C)

                # ---- Phase 2 batch: per-task w3 slot -> acc_o -> writeback.
                # Form B ONLY (#R5-2 session-6): Form A single-block C-scope now
                # FIX-writes Output directly from l0c_o (O = P_hat @ V is final
                # under the pre-normalization specialization), so this phase and
                # the WS3 SEM round are deleted there. `not form_a` is a
                # parse-time Python bool (closure) -> Form B IR byte-identical
                # (same mechanism as `if not io3_is_alias:` in init above).
                # V-init's SEM_WS3_V2C pre-set token stays: dangling-but-
                # harmless exactly like the by-design dangling WS1_V2C +1
                # (cross-core semaphore state is per-launch; C never waits).
                if not form_a:
                    for i in T.serial(batch_tasks):
                        task = my_start + kb * NUM_STAGES + i
                        bx = task % s_blocks
                        by = task // s_blocks % N
                        bz = task // s_blocks // N % B
                        s0 = bx * block_M
                        valid_m = T.min(block_M, S - s0)
                        rows_vid = T.min(half_M, T.max(0, valid_m - vid * half_M))
                        io_side = i % 2
                        T.wait_flag("V", "MTE2", sig_io3_base + io_side)
                        if i % CROSS_INTERVAL == 0:
                            T.wait_cross_flag(SEM_WS3_C2V)
                        T.copy(
                            workspace_3[cid, i, vid * half_M : vid * half_M + half_M, :],
                            io3[io_side, :, :],
                        )
                        T.set_flag("MTE2", "V", sig_io3_base + io_side)
                        T.wait_flag("MTE2", "V", sig_io3_base + io_side)
                        T.tile.fill(acc_o, 0.0)
                        T.tile.add(acc_o, acc_o, io3[io_side, :, :])
                        T.set_flag("V", "MTE2", sig_io3_base + io_side)
                        # writeback inside the SIG_S_HALF closed loop + rows_vid guard
                        T.wait_flag("MTE3", "V", SIG_S_HALF)
                        T.copy(acc_o, out_half)  # fp32 -> dtype
                        T.set_flag("V", "MTE3", SIG_S_HALF)
                        T.wait_flag("V", "MTE3", SIG_S_HALF)
                        if rows_vid > 0:
                            T.copy(
                                out_half[0:rows_vid, :],
                                Output[bz, s0 + vid * half_M : s0 + vid * half_M + rows_vid, by, :],
                            )
                        T.set_flag("MTE3", "V", SIG_S_HALF)
                    T.set_cross_flag("MTE2", SEM_WS3_V2C)

        elif flat_v:
            # ---- #R8-2 (session-8 Stage-3) FLAT (task,kv) token stream, V side
            # (form_b_mma generic domain, case9/case4). Blueprint = #R4-3 decode
            # flat (17/17 BITEXACT archived: cross-round task-state carry /
            # mid-round task switch / phase2 reset timing) + §S4-3-5 (task-state
            # resets MUST sit in phase2's kv==0 slot — phase1 runs entirely
            # before phase2 within a round, so a phase1 reset of acc_o/sumexp
            # would destroy the PREVIOUS task's in-flight state still awaiting
            # its phase2 consumption; m_i is the exception: phase1-exclusive,
            # reduce_max overwrite semantics + m_i_prev chain make fill→copy
            # bit-identical to the generic task-head sequence).
            # PREFILL delta vs the decode blueprint: causal eff_kv_iters is
            # RUNTIME-NONUNIFORM per task (case9 mean 4.5) ⇒ flat index f→
            # (task,kv) mapping = sequential walk state machine (T.alloc_var,
            # O(1) amortized/slot + one-time O(my_count) total_flat walk),
            # computed IDENTICALLY on C/V from the same my_start/my_count and
            # the verbatim causal formula (§7.2-5 hard constraint; never via
            # ws). Per-slot op sequence = generic verbatim (BITEXACT expected,
            # #R4-3/#R5-1 precedent); batch membership changes: task boundaries
            # no longer drain the pipeline (batch depth 4.5→8, protocol rounds
            # ~103→58/core case9, V task-tail decoupled from round boundaries).
            total_flat = T.alloc_var("int32", init=0)
            for t in T.serial(my_count):
                _tw = my_start + t
                _s0w = (_tw % s_blocks) * block_M
                _effw = T.floordiv(T.min(_s0w + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                total_flat += _effw
            num_fb = T.ceildiv(total_flat, NUM_STAGES)
            tl_head = T.alloc_var("int32", init=0)  # persistent batch-head walk state
            kv_head = T.alloc_var("int32", init=0)
            for fb in T.serial(num_fb):
                _rem_f = total_flat - fb * NUM_STAGES
                batch_iters = T.if_then_else(_rem_f < NUM_STAGES, _rem_f, NUM_STAGES)

                # ---- Phase 1: batched online softmax (w1 -> P -> w2), flat slots ----
                T.wait_cross_flag(SEM_WS2_C2V)
                tl_p1 = T.alloc_var("int32", init=tl_head)
                kv_p1 = T.alloc_var("int32", init=kv_head)
                for i in T.serial(batch_iters):
                    io_side = i % 2
                    task = my_start + tl_p1
                    bx = task % s_blocks
                    s0 = bx * block_M
                    eff = T.floordiv(T.min(s0 + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                    idx = kv_p1
                    # w1[cid, i, vid half] -> io_buf -> work_ub (fp32 mainline)
                    T.wait_flag("V", "MTE2", SIG_IO_UB + io_side)
                    if i % CROSS_INTERVAL == 0:
                        T.wait_cross_flag(SEM_WS1_C2V)
                    T.copy(
                        workspace_1[cid, i, vid * half_M : vid * half_M + half_M, :],
                        io_buf[io_side, :, :],
                    )
                    T.set_flag("MTE2", "V", SIG_IO_UB + io_side)
                    T.wait_flag("MTE2", "V", SIG_IO_UB + io_side)
                    T.copy(io_buf[io_side, :, :], work_ub)
                    T.set_flag("V", "MTE2", SIG_IO_UB + io_side)

                    if kv_p1 == 0:
                        # task-start resets: phase1-EXCLUSIVE state only (m_i
                        # exception per §S4-3-5; acc_o/sumexp reset lives in
                        # phase2's kv==0 slot below) + row_thresh recompute
                        # (task-invariant thresholds; pure-V no-flag segment)
                        T.tile.fill(m_i, -(2**30))
                        if need_mask:
                            T.tile.arith_progression(row_thresh, s0 + vid * half_M + offset_eff, 1, half_M)
                            T.tile.min(row_thresh, row_thresh, T.cast(S_kv - 1, accum_dtype))

                    T.copy(m_i, m_i_prev)  # history max BEFORE reduce (§10.2 lesson)
                    # golden score-grid alignment (§3.2.1-⑦): identical op order
                    T.copy(work_ub, rr_ub)  # r16(qk)
                    T.copy(rr_ub, work_ub)
                    T.tile.mul(work_ub, work_ub, sm_scale)  # * s in fp32
                    T.copy(work_ub, rr_ub)  # r16(scores) = golden grid
                    T.copy(rr_ub, work_ub)

                    if need_mask:  # noqa: SIM102 — parse-time outer / runtime inner (rev2-A1)
                        # runtime skip for fully-visible blocks (§3.2.1-④): the
                        # condition is s0-block-uniform -> both vids take the same
                        # branch (flag pairing safe; pure-V segment either way).
                        if idx * block_N + block_N - 1 > T.min(s0 + offset_eff, S_kv - 1):
                            j0 = idx * block_N
                            T.tile.arith_progression(col_pos, j0, 1, block_N)
                            T.tile.broadcast(buf_2d, col_pos, axis=0)  # [bN] -> [hM,bN]
                            T.tile.broadcast(rt_2d, row_thresh, axis=1)  # [hM] -> [hM,bN]
                            T.tile.compare(mask_2d, buf_2d, rt_2d, "LE")
                            T.tile.select(
                                work_ub,
                                mask_2d,
                                work_ub,
                                -T.infinity(accum_dtype),
                                "VSEL_TENSOR_SCALAR_MODE",
                            )

                    T.reduce_max(work_ub, m_i, dim=-1)
                    T.tile.max(m_i, m_i, m_i_prev)
                    # deferred rescale: store the difference, exp in phase 2
                    # (bit-identical to per-iteration exp, §9.3.3-6)
                    T.tile.sub(r_factors[i, :, :], m_i_prev, m_i)
                    T.tile.broadcast(buf_2d, m_i, axis=1)
                    T.tile.sub(work_ub, work_ub, buf_2d)
                    T.tile.exp(work_ub, work_ub)  # P = exp(s - m_new) in [0, 1]
                    T.reduce_sum(work_ub, sumexp_is[i, :, :], dim=-1)

                    # P -> w2 (SIG_S_HALF ownership closed loop, mha_sink:517-528)
                    T.wait_flag("MTE3", "V", SIG_S_HALF)
                    T.copy(work_ub, acc_s_half)  # fp32 -> dtype (P native grid)
                    T.set_flag("V", "MTE3", SIG_S_HALF)
                    T.wait_flag("V", "MTE3", SIG_S_HALF)
                    T.copy(
                        acc_s_half,
                        workspace_2[cid, i, vid * half_M : vid * half_M + half_M, :],
                    )
                    T.set_flag("MTE3", "V", SIG_S_HALF)
                    if (i + 1) % CROSS_INTERVAL == 0 or i == batch_iters - 1:
                        T.set_cross_flag("MTE3", SEM_WS2_V2C)
                    # walk advance (pure scalar; eff = THIS slot's task)
                    kv_p1 += 1
                    if kv_p1 == eff:
                        kv_p1 = 0
                        tl_p1 += 1
                T.set_cross_flag("MTE2", SEM_WS1_V2C)

                # ---- Phase 2: O accumulation batch (w3 -> acc_o), flat slots ----
                tl_p2 = T.alloc_var("int32", init=tl_head)
                kv_p2 = T.alloc_var("int32", init=kv_head)
                for i in T.serial(batch_iters):
                    io_side = i % 2
                    task = my_start + tl_p2
                    bx = task % s_blocks
                    by = task // s_blocks % N
                    bz = task // s_blocks // N % B
                    s0 = bx * block_M
                    valid_m = T.min(block_M, S - s0)
                    rows_vid = T.min(half_M, T.max(0, valid_m - vid * half_M))
                    eff = T.floordiv(T.min(s0 + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                    if kv_p2 == 0:
                        # task-start resets (§S4-3-5): equivalent migration of
                        # the generic task-head fills — same first-consumption
                        # point, same fill values; MUST be phase2 (a phase1
                        # reset would destroy the previous task's in-flight
                        # acc_o/sumexp awaiting their tail below)
                        T.tile.fill(acc_o, 0.0)
                        T.tile.fill(sumexp, 0.0)
                    T.tile.exp(r_factors[i, :, :], r_factors[i, :, :])  # r_i
                    T.tile.mul(sumexp, sumexp, r_factors[i, :, :])
                    T.tile.add(sumexp, sumexp, sumexp_is[i, :, :])
                    T.tile.broadcast(r_2d, r_factors[i, :, :], axis=1)  # [hM,1]->[hM,D]
                    T.tile.mul(acc_o, acc_o, r_2d)
                    T.wait_flag("V", "MTE2", sig_io3_base + io_side)
                    if i % CROSS_INTERVAL == 0:
                        T.wait_cross_flag(SEM_WS3_C2V)
                    T.copy(
                        workspace_3[cid, i, vid * half_M : vid * half_M + half_M, :],
                        io3[io_side, :, :],
                    )
                    T.set_flag("MTE2", "V", sig_io3_base + io_side)
                    T.wait_flag("MTE2", "V", sig_io3_base + io_side)
                    T.tile.add(acc_o, acc_o, io3[io_side, :, :])
                    T.set_flag("V", "MTE2", sig_io3_base + io_side)
                    if kv_p2 == eff - 1:
                        # ---- task tail: final normalization + writeback rides
                        # the task's LAST flat slot (SIG_S_HALF closed loop
                        # entirely inside the runtime if — wait/set enter-exit
                        # together, #R4-3 blueprint; conditions vid-independent:
                        # tl/kv/eff walk identically on both vids, §7.2-5) ----
                        T.tile.add(sumexp, sumexp, 1e-30)  # all-masked-row guard
                        T.tile.broadcast(r_2d, sumexp, axis=1)
                        T.tile.div(acc_o, acc_o, r_2d)
                        T.wait_flag("MTE3", "V", SIG_S_HALF)
                        T.copy(acc_o, out_half)  # fp32 -> dtype
                        T.set_flag("V", "MTE3", SIG_S_HALF)
                        T.wait_flag("V", "MTE3", SIG_S_HALF)
                        if rows_vid > 0:
                            T.copy(
                                out_half[0:rows_vid, :],
                                Output[bz, s0 + vid * half_M : s0 + vid * half_M + rows_vid, by, :],
                            )
                        T.set_flag("MTE3", "V", SIG_S_HALF)
                    # walk advance
                    kv_p2 += 1
                    if kv_p2 == eff:
                        kv_p2 = 0
                        tl_p2 += 1
                T.set_cross_flag("MTE2", SEM_WS3_V2C)
                # batch-head state advance (phase2 walk ended at the next head)
                tl_head = tl_p2
                kv_head = kv_p2

        else:
            for t in T.serial(my_count):
                task = my_start + t
                bx = task % s_blocks
                by = task // s_blocks % N
                bz = task // s_blocks // N % B
                s0 = bx * block_M
                valid_m = T.min(block_M, S - s0)
                # Causal KV runtime trim — MUST match the C-scope formula verbatim
                # (§7.2-5/R11 flag-pairing hard constraint; mha_sink:394-400).
                eff_kv_iters = T.floordiv(T.min(s0 + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                eff_num_outer = T.ceildiv(eff_kv_iters, NUM_STAGES)
                rows_vid = T.min(half_M, T.max(0, valid_m - vid * half_M))

                T.tile.fill(acc_o, 0.0)
                T.tile.fill(sumexp, 0.0)
                T.tile.fill(m_i, -(2**30))
                if need_mask:
                    # task-invariant row thresholds (hoisted; vid half-region start):
                    # thresh_r = min(s0 + vid*hM + r + offset_eff, S_kv-1)
                    T.tile.arith_progression(row_thresh, s0 + vid * half_M + offset_eff, 1, half_M)
                    T.tile.min(row_thresh, row_thresh, T.cast(S_kv - 1, accum_dtype))

                for k in T.serial(eff_num_outer):
                    _remaining = eff_kv_iters - k * NUM_STAGES
                    batch_iters = T.if_then_else(_remaining < NUM_STAGES, _remaining, NUM_STAGES)

                    # ---- Phase 1: batched online softmax (w1 -> P -> w2) ----
                    T.wait_cross_flag(SEM_WS2_C2V)
                    for i in T.serial(batch_iters):
                        io_side = i % 2
                        idx = k * NUM_STAGES + i
                        # w1[cid, i, vid half] -> io_buf -> work_ub (fp32 mainline)
                        T.wait_flag("V", "MTE2", SIG_IO_UB + io_side)
                        if i % CROSS_INTERVAL == 0:
                            T.wait_cross_flag(SEM_WS1_C2V)
                        T.copy(
                            workspace_1[cid, i, vid * half_M : vid * half_M + half_M, :],
                            io_buf[io_side, :, :],
                        )
                        T.set_flag("MTE2", "V", SIG_IO_UB + io_side)
                        T.wait_flag("MTE2", "V", SIG_IO_UB + io_side)
                        T.copy(io_buf[io_side, :, :], work_ub)
                        T.set_flag("V", "MTE2", SIG_IO_UB + io_side)

                        T.copy(m_i, m_i_prev)  # history max BEFORE reduce (§10.2 lesson)
                        # golden score-grid alignment (§3.2.1-⑦): round16(qk) -> *s fp32
                        # -> round16(scores); chain order = Developer mha.py:304-308.
                        # rr_ub = dedicated round-trip scratch (NEVER acc_s_half —
                        # MTE3 cross-core source, §8.6-2 alias ban).
                        T.copy(work_ub, rr_ub)  # r16(qk)
                        T.copy(rr_ub, work_ub)
                        T.tile.mul(work_ub, work_ub, sm_scale)  # * s in fp32
                        T.copy(work_ub, rr_ub)  # r16(scores) = golden grid
                        T.copy(rr_ub, work_ub)

                        if need_mask:  # noqa: SIM102 — outer if is PARSE-TIME (Python
                            # bool variant switch, §5.6); inner if is a RUNTIME TIR
                            # condition. Combining them into `need_mask and <TIR expr>`
                            # would hand a BoolOp AST to the TVMScript parser (unverified)
                            # — the two-level structure is intentional (rev2-A1 inherited).
                            # runtime skip for fully-visible blocks (§3.2.1-④): the
                            # condition is s0-block-uniform -> both vids take the same
                            # branch (flag pairing safe; pure-V segment either way).
                            if idx * block_N + block_N - 1 > T.min(s0 + offset_eff, S_kv - 1):
                                j0 = idx * block_N
                                T.tile.arith_progression(col_pos, j0, 1, block_N)
                                T.tile.broadcast(buf_2d, col_pos, axis=0)  # [bN] -> [hM,bN]
                                T.tile.broadcast(rt_2d, row_thresh, axis=1)  # [hM] -> [hM,bN]
                                T.tile.compare(mask_2d, buf_2d, rt_2d, "LE")
                                T.tile.select(
                                    work_ub,
                                    mask_2d,
                                    work_ub,
                                    -T.infinity(accum_dtype),
                                    "VSEL_TENSOR_SCALAR_MODE",
                                )

                        T.reduce_max(work_ub, m_i, dim=-1)
                        T.tile.max(m_i, m_i, m_i_prev)
                        # deferred rescale: store the difference, exp in phase 2
                        # (bit-identical to per-iteration exp, §9.3.3-6)
                        T.tile.sub(r_factors[i, :, :], m_i_prev, m_i)
                        T.tile.broadcast(buf_2d, m_i, axis=1)
                        T.tile.sub(work_ub, work_ub, buf_2d)
                        T.tile.exp(work_ub, work_ub)  # P = exp(s - m_new) in [0, 1]
                        T.reduce_sum(work_ub, sumexp_is[i, :, :], dim=-1)

                        if single_kv_block:
                            # pre-normalization specialization (§3.2.1-⑤): P_hat =
                            # P/(sumexp+1e-30) BEFORE the dtype rounding = golden
                            # one-shot softmax grid; phase 2 skips rescale/final div.
                            T.tile.add(sumexp, sumexp_is[0, :, :], 1e-30)
                            T.tile.broadcast(buf_2d, sumexp, axis=1)
                            T.tile.div(work_ub, work_ub, buf_2d)

                        # P -> w2 (SIG_S_HALF ownership closed loop, mha_sink:517-528)
                        T.wait_flag("MTE3", "V", SIG_S_HALF)
                        T.copy(work_ub, acc_s_half)  # fp32 -> dtype (P native grid)
                        T.set_flag("V", "MTE3", SIG_S_HALF)
                        T.wait_flag("V", "MTE3", SIG_S_HALF)
                        T.copy(
                            acc_s_half,
                            workspace_2[cid, i, vid * half_M : vid * half_M + half_M, :],
                        )
                        T.set_flag("MTE3", "V", SIG_S_HALF)
                        if (i + 1) % CROSS_INTERVAL == 0 or i == batch_iters - 1:
                            T.set_cross_flag("MTE3", SEM_WS2_V2C)
                    T.set_cross_flag("MTE2", SEM_WS1_V2C)

                    # ---- Phase 2: O accumulation batch (w3 -> acc_o) ----
                    for i in T.serial(batch_iters):
                        if not single_kv_block:
                            T.tile.exp(r_factors[i, :, :], r_factors[i, :, :])  # r_i
                            T.tile.mul(sumexp, sumexp, r_factors[i, :, :])
                            T.tile.add(sumexp, sumexp, sumexp_is[i, :, :])
                            T.tile.broadcast(r_2d, r_factors[i, :, :], axis=1)  # [hM,1]->[hM,D]
                            T.tile.mul(acc_o, acc_o, r_2d)
                        io_side = i % 2
                        T.wait_flag("V", "MTE2", sig_io3_base + io_side)
                        if i % CROSS_INTERVAL == 0:
                            T.wait_cross_flag(SEM_WS3_C2V)
                        T.copy(
                            workspace_3[cid, i, vid * half_M : vid * half_M + half_M, :],
                            io3[io_side, :, :],
                        )
                        T.set_flag("MTE2", "V", sig_io3_base + io_side)
                        T.wait_flag("MTE2", "V", sig_io3_base + io_side)
                        T.tile.add(acc_o, acc_o, io3[io_side, :, :])
                        T.set_flag("V", "MTE2", sig_io3_base + io_side)
                    T.set_cross_flag("MTE2", SEM_WS3_V2C)

                # ---- task tail: final normalization (generic path) ----
                if not single_kv_block:
                    T.tile.add(sumexp, sumexp, 1e-30)  # all-masked-row guard
                    T.tile.broadcast(r_2d, sumexp, axis=1)
                    T.tile.div(acc_o, acc_o, r_2d)
                # writeback inside the SIG_S_HALF closed loop + rows_vid runtime
                # guard (§5.4 vid row-split rule; best-practices §6: final store
                # must be inside the ownership loop).
                T.wait_flag("MTE3", "V", SIG_S_HALF)
                T.copy(acc_o, out_half)  # fp32 -> dtype
                T.set_flag("V", "MTE3", SIG_S_HALF)
                T.wait_flag("V", "MTE3", SIG_S_HALF)
                if rows_vid > 0:
                    T.copy(
                        out_half[0:rows_vid, :],
                        Output[bz, s0 + vid * half_M : s0 + vid * half_M + rows_vid, by, :],
                    )
                T.set_flag("MTE3", "V", SIG_S_HALF)

        # destroy (§7.2-4): consume outstanding init-direction tokens
        T.wait_flag("V", "MTE2", SIG_IO_UB)
        T.wait_flag("V", "MTE2", SIG_IO_UB + 1)
        if not io3_is_alias:
            T.wait_flag("V", "MTE2", sig_io3_base)
            T.wait_flag("V", "MTE2", sig_io3_base + 1)
        T.wait_flag("MTE3", "V", SIG_S_HALF)

    # ==================== rev4 Form B-mma note: the §3.3 GEMM1_BODY/GEMM2_BODY
    # macros are INLINE-EXPANDED at their four call sites (single-block/generic
    # x GEMM1/GEMM2). A plain-Python closure emitter was probed FIRST and is
    # UNSAFE: its T.* emissions are silently dropped by the TVMScript parser
    # (compiled "OK" but the generated C-scope contained only caller-level
    # instructions — evidence /tmp/opencode/rev4_probes/smoke_fbm_src.cc,
    # aicore timeout at the orphaned WaitFlag<M_FIX>). Inline expansion = the
    # §8.3 verbatim listing = Form A's all-inline proven style. ==============
    # ==================== V-side buffer allocation is form-shared; emit inside
    # each prim_func (TVMScript scoping). Form dispatch = Python-level branch
    # building three self-contained prim_funcs (§6.1 note, rev4 three-way). ====
    if form_a:

        @T.prim_func
        def main(
            Q: T.Tensor([B, S, N, D], dtype),  # type: ignore
            K: T.Tensor([B, S_kv, N, D], dtype),  # type: ignore
            V: T.Tensor([B, S_kv, N, D], dtype),  # type: ignore
            Output: T.Tensor([B, S, N, D], dtype),  # type: ignore
            workspace_1: T.Tensor([core_num, NUM_STAGES, block_M, block_N], accum_dtype),  # type: ignore
            workspace_2: T.Tensor([core_num, NUM_STAGES, block_M, block_N], dtype),  # type: ignore
            workspace_3: T.Tensor([core_num, NUM_STAGES, block_M, D], accum_dtype),  # type: ignore
        ):
            with T.Kernel(core_num, is_npu=True) as (cid, vid):
                # ---- C-side L1 (ZN/NZ annotated for mma operands, §3.2) ----
                q_l1 = T.alloc_L1([block_M, D], dtype)
                k_l1 = T.alloc_L1([block_N, D], dtype)
                v_l1 = T.alloc_L1([block_N, D], dtype)
                p_l1 = T.alloc_L1([block_M, block_N], dtype)
                T.annotate_layout(
                    {
                        q_l1: make_zn_layout(q_l1),
                        k_l1: make_nz_layout(k_l1),
                        p_l1: make_zn_layout(p_l1),
                        v_l1: make_zn_layout(v_l1),
                    }
                )
                # ---- L0 double-slot pools (§4.5; P2 shares one pool, P1 splits) ----
                l0a_s = T.alloc_L0A([2, block_M, D], dtype)
                l0b_s = T.alloc_L0B([2, D, block_N], dtype)
                l0c_s = T.alloc_L0C([2, block_M, block_N], accum_dtype)
                l0a_o = l0a_s if block_N == D else T.alloc_L0A([2, block_M, block_N], dtype)
                l0b_o = l0b_s if block_N == D else T.alloc_L0B([2, block_N, D], dtype)
                l0c_o = l0c_s if block_N == D else T.alloc_L0C([2, block_M, D], accum_dtype)
                # ---- V-side UB (vid half rows; §4.3 + adaptations) ----
                io_buf = T.alloc_ub([2, half_M, block_N], accum_dtype)
                io3 = io_buf if io3_is_alias else T.alloc_ub([2, half_M, D], accum_dtype)
                work_ub = T.alloc_ub([half_M, block_N], accum_dtype)
                buf_2d = T.alloc_ub([half_M, block_N], accum_dtype)
                rt_2d = T.alloc_ub([half_M, block_N], accum_dtype)
                r_2d = buf_2d if block_N == D else T.alloc_ub([half_M, D], accum_dtype)
                mask_2d = T.alloc_ub([half_M * block_N // 8], "uint8")
                col_pos = T.alloc_ub([block_N], accum_dtype)
                row_thresh = T.alloc_ub([half_M], accum_dtype)
                rr_ub = T.alloc_ub([half_M, block_N], dtype)
                acc_s_half = T.alloc_ub([half_M, block_N], dtype)
                acc_o = T.alloc_ub([half_M, D], accum_dtype)
                out_half = T.alloc_ub([half_M, D], dtype)
                m_i = T.alloc_ub([half_M, 1], accum_dtype)
                m_i_prev = T.alloc_ub([half_M, 1], accum_dtype)
                sumexp = T.alloc_ub([half_M, 1], accum_dtype)
                r_factors = T.alloc_ub([NUM_STAGES, half_M, 1], accum_dtype)
                sumexp_is = T.alloc_ub([NUM_STAGES, half_M, 1], accum_dtype)

                my_start = cid * q_tasks + T.if_then_else(cid < r_tasks, cid, r_tasks)
                my_count = q_tasks + T.if_then_else(cid < r_tasks, 1, 0)

                # ==================== Cube core (Form A: mma full pipeline) ====================
                with T.Scope("C"):
                    # init: pretend consumers already released (§7.2-4)
                    T.set_cross_flag("MTE2", SEM_WS2_C2V)
                    T.set_flag("MTE1", "MTE2", SIG_K_L1)
                    T.set_flag("MTE1", "MTE2", SIG_P_L1)
                    T.set_flag("MTE1", "MTE2", SIG_V_L1)
                    T.set_flag("MTE1", "MTE2", SIG_Q_L1)
                    T.set_flag("M", "MTE1", SIG_L0AB)
                    T.set_flag("M", "MTE1", SIG_L0AB + 1)
                    T.set_flag("FIX", "M", SIG_L0C)
                    T.set_flag("FIX", "M", SIG_L0C + 1)

                    if single_kv_block:
                        # ---- #R5-1 task-batched single-KV-block C path (Form A) ----
                        # Mirror of the Form B batched branch below, with the Form A
                        # mma pipeline body (GEMM1/GEMM2 batches) carried over verbatim
                        # from the generic per-task flow: slot i <-> task
                        # my_start + kb*NUM_STAGES + i. num_tb/batch_tasks from the SAME
                        # formulas as the V mirror (§7.2-5 flag-pairing hard constraint).
                        # Rationale: kv_iters==1 collapses eff_num_outer to 1 and
                        # batch_iters to 1, so the NUM_STAGES pipeline (and the io_buf[2]
                        # ping-pong it feeds) never had a second slot in flight — every
                        # task ran a strictly serial C-QK -> V-softmax -> C-PV -> V-epi
                        # chain (camodel case1: C FLOWCTRL 83.2% of wall, V MTE/VECTOR
                        # overlap 5.3%). Batching the TASK dimension is the in-rules
                        # Expert expression of CV overlap (T.Pipelined is banned,
                        # §2.2-d/R9; performance-antipatterns "AIC/AIV no CV overlap").
                        # Two Form-A-specific deltas vs the generic body (both are Q
                        # scheduling, zero arithmetic change):
                        #   (a) the Q GM->L1 load moves INSIDE the slot loop (each task
                        #       has its own s0), and
                        #   (b) q_l1 -> l0a_s[side] is issued for EVERY slot (generic
                        #       guards it with `if i < 2` because Q is kv-invariant
                        #       there; here Q changes per slot).
                        # SIG_Q_L1 token balance: per slot wait(MTE1->MTE2) +
                        # set(MTE2->MTE1) + wait(MTE2->MTE1) + set(MTE1->MTE2) right
                        # after the single MTE1 read of q_l1 = net 0, identical to the
                        # generic task-level accounting — so the task-end return at the
                        # bottom of the else branch is NOT emitted here.
                        num_tb = T.ceildiv(my_count, NUM_STAGES)
                        for kb in T.serial(num_tb):
                            _remaining_t = my_count - kb * NUM_STAGES
                            batch_tasks = T.if_then_else(_remaining_t < NUM_STAGES, _remaining_t, NUM_STAGES)
                            valid_n = min(block_N, S_kv)  # single block: idx == 0

                            # --- GEMM1 batch: per-task scores -> w1 slot i ---
                            T.wait_cross_flag(SEM_WS1_V2C)
                            for i in T.serial(batch_tasks):
                                side = i % 2
                                task = my_start + kb * NUM_STAGES + i
                                bx = task % s_blocks
                                by = task // s_blocks % N
                                bz = task // s_blocks // N % B
                                s0 = bx * block_M
                                valid_m = T.min(block_M, S - s0)
                                # Q: GM -> L1 (per-slot residency; BSHD row-stride)
                                T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                                T.copy(Q[bz, s0 : s0 + valid_m, by, :], q_l1)
                                T.set_flag("MTE2", "MTE1", SIG_Q_L1)
                                T.wait_flag("MTE2", "MTE1", SIG_Q_L1)
                                # K: GM -> L1 (single block => j0 == 0)
                                T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                                T.copy(K[bz, 0:valid_n, by, :], k_l1)
                                T.set_flag("MTE2", "MTE1", SIG_K_L1)
                                # Q: L1 -> L0A (EVERY slot: task-dependent Q)
                                T.wait_flag("M", "MTE1", SIG_L0AB + side)
                                T.copy(q_l1, l0a_s[side, :, :])
                                T.set_flag("MTE1", "MTE2", SIG_Q_L1)
                                # K: L1 -> L0B transposed (NZ)
                                T.wait_flag("MTE2", "MTE1", SIG_K_L1)
                                T.copy(k_l1, l0b_s[side, :, :], transpose=True)
                                T.set_flag("MTE1", "MTE2", SIG_K_L1)
                                T.set_flag("MTE1", "M", SIG_L0AB + side)
                                # mma: Q K^T -> L0C
                                T.wait_flag("MTE1", "M", SIG_L0AB + side)
                                T.wait_flag("FIX", "M", SIG_L0C + side)
                                T.mma(l0a_s[side, :, :], l0b_s[side, :, :], l0c_s[side, :, :], init=True)
                                T.set_flag("M", "MTE1", SIG_L0AB + side)
                                T.set_flag("M", "FIX", SIG_L0C + side)
                                # L0C -> w1 (FIX)
                                T.wait_flag("M", "FIX", SIG_L0C + side)
                                T.copy(l0c_s[side, :, :], workspace_1[cid, i, :, :])
                                T.set_flag("FIX", "M", SIG_L0C + side)
                                if (i + 1) % CROSS_INTERVAL == 0 or i == batch_tasks - 1:
                                    T.set_cross_flag("FIX", SEM_WS1_C2V)

                            # --- GEMM2 batch: per-task P (w2 slot i) @ V -> Output
                            # direct FIX write (#R5-2 session-6: single-block O =
                            # P_hat @ V is FINAL (pre-normalization specialization,
                            # §3.2.1-⑤) so the w3 fp32 GM round-trip, the V
                            # phase-2 acc_o pass and the WS3 SEM pair are all
                            # removable.
                            # fixpipe fp32->dtype RNE == V-side vcast (probe #R3-3
                            # 11/11 BITEXACT); L0C->BSHD 4D runtime-row-slice FIX
                            # gate probe 8/8 (probe_r52_l0c_bhsd, int-exact
                            # BITEXACT incl. tail valid_m< bM / bf16 / D=128).
                            # C writes ALL valid_m rows at once (no vid split).
                            for i in T.serial(batch_tasks):
                                side = i % 2
                                task = my_start + kb * NUM_STAGES + i
                                bx = task % s_blocks
                                by = task // s_blocks % N
                                bz = task // s_blocks // N % B
                                s0 = bx * block_M
                                valid_m = T.min(block_M, S - s0)
                                # V: GM -> L1
                                T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                                T.copy(V[bz, 0:valid_n, by, :], v_l1)
                                T.set_flag("MTE2", "MTE1", SIG_V_L1)
                                # P: w2 -> p_l1
                                T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                                if i % CROSS_INTERVAL == 0:
                                    T.wait_cross_flag(SEM_WS2_V2C)
                                T.copy(workspace_2[cid, i, :, :], p_l1)
                                T.set_flag("MTE2", "MTE1", SIG_P_L1)
                                # V: L1 -> L0B (no transpose for PV)
                                T.wait_flag("MTE2", "MTE1", SIG_V_L1)
                                T.wait_flag("M", "MTE1", SIG_L0AB + side)
                                T.copy(v_l1, l0b_o[side, :, :])
                                T.set_flag("MTE1", "MTE2", SIG_V_L1)
                                # P: L1 -> L0A
                                T.wait_flag("MTE2", "MTE1", SIG_P_L1)
                                T.copy(p_l1, l0a_o[side, :, :])
                                T.set_flag("MTE1", "MTE2", SIG_P_L1)
                                T.set_flag("MTE1", "M", SIG_L0AB + side)
                                # mma: P V -> L0C
                                T.wait_flag("MTE1", "M", SIG_L0AB + side)
                                T.wait_flag("FIX", "M", SIG_L0C + side)
                                T.mma(l0a_o[side, :, :], l0b_o[side, :, :], l0c_o[side, :, :], init=True)
                                T.set_flag("M", "MTE1", SIG_L0AB + side)
                                T.set_flag("M", "FIX", SIG_L0C + side)
                                # L0C -> Output (FIX direct write, runtime row
                                # extent; SIG_L0C slot ownership unchanged)
                                T.wait_flag("M", "FIX", SIG_L0C + side)
                                T.copy(l0c_o[side, :, :], Output[bz, s0 : s0 + valid_m, by, :])
                                T.set_flag("FIX", "M", SIG_L0C + side)

                            T.set_cross_flag("MTE2", SEM_WS2_C2V)

                    else:
                        # ---- #R8-3 (session-8 Stage-3, mission domain order
                        # step 2) FLAT (task,kv) continuous token stream, Form A
                        # generic multi-kv domain (case7/8/3/2/5). Mirror of the
                        # form_b_mma flat branch (#R8-2, ADOPTED case9 -7.96% /
                        # case4 -20.72%) with TWO Form-A-specific deltas (both
                        # Q/l0a_s scheduling, zero arithmetic change):
                        # (a) l0a_s reload condition `i < 2 or kv < 2` — the P2
                        #     shared pool (l0a_o aliases l0a_s) is overwritten by
                        #     every GEMM2 batch, so batch-head slots MUST reload
                        #     Q (legacy `if i < 2` semantics preserved), AND flat
                        #     mid-batch task switches need the new task's first
                        #     two slots to reload their sides (Q is task-
                        #     invariant; kv>=2 slots reuse the resident copy).
                        # (b) q_l1 return (SIG_Q_L1 MTE1->MTE2 set) moves to the
                        #     task's LAST slot (kv==eff-1), NOT kv==1: cross-batch
                        #     tasks re-read q_l1 at subsequent batch-head slots
                        #     (delta-a reloads); an earlier release would let the
                        #     NEXT task's kv0 MTE2 load overwrite q_l1 while those
                        #     reads are still MTE1-queued (race). M-pipe/MTE1
                        #     in-order from the last slot covers every q_l1 read
                        #     of the task. Per-task accounting unchanged: 1 full
                        #     SIG_Q_L1 round (kv==0, co-conditioned 同进同出) +
                        #     1 return (kv==eff-1) = net 0, init/destroy verbatim.
                        # Walk state machine + batch skeleton + V mirror =
                        # #R8-2 verbatim (§7.2-5 same-formula across C/V). ----
                        total_flat = T.alloc_var("int32", init=0)
                        for t in T.serial(my_count):
                            _tw = my_start + t
                            _s0w = (_tw % s_blocks) * block_M
                            _effw = T.floordiv(T.min(_s0w + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                            total_flat += _effw
                        num_fb = T.ceildiv(total_flat, NUM_STAGES)
                        tl_head = T.alloc_var("int32", init=0)  # persistent batch-head walk state
                        kv_head = T.alloc_var("int32", init=0)
                        for fb in T.serial(num_fb):
                            _rem_f = total_flat - fb * NUM_STAGES
                            batch_iters = T.if_then_else(_rem_f < NUM_STAGES, _rem_f, NUM_STAGES)

                            # --- GEMM1 batch: flat slots -> w1 ---
                            T.wait_cross_flag(SEM_WS1_V2C)
                            tl_g1 = T.alloc_var("int32", init=tl_head)
                            kv_g1 = T.alloc_var("int32", init=kv_head)
                            for i in T.serial(batch_iters):
                                side = i % 2
                                task = my_start + tl_g1
                                bx = task % s_blocks
                                by = task // s_blocks % N
                                bz = task // s_blocks // N % B
                                s0 = bx * block_M
                                valid_m = T.min(block_M, S - s0)
                                eff_kv_iters = (
                                    T.floordiv(T.min(s0 + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                                )
                                idx = kv_g1
                                j0 = idx * block_N
                                valid_n = T.min(block_N, S_kv - j0)
                                if kv_g1 == 0:
                                    # ---- task switch: Q GM->L1 (full SIG_Q_L1
                                    # MTE2 round inside the condition; the l0a_s
                                    # reload rides this slot's L0AB round below) ----
                                    T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                                    T.copy(Q[bz, s0 : s0 + valid_m, by, :], q_l1)
                                    T.set_flag("MTE2", "MTE1", SIG_Q_L1)
                                    T.wait_flag("MTE2", "MTE1", SIG_Q_L1)
                                # K: GM -> L1 (BSHD row-stride, runtime extent)
                                T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                                T.copy(K[bz, j0 : j0 + valid_n, by, :], k_l1)
                                T.set_flag("MTE2", "MTE1", SIG_K_L1)
                                # Q: L1 -> L0A (delta-a: batch-head OR task-head sides)
                                T.wait_flag("M", "MTE1", SIG_L0AB + side)
                                if i < 2 or kv_g1 < 2:
                                    T.copy(q_l1, l0a_s[side, :, :])
                                # K: L1 -> L0B transposed (NZ)
                                T.wait_flag("MTE2", "MTE1", SIG_K_L1)
                                T.copy(k_l1, l0b_s[side, :, :], transpose=True)
                                T.set_flag("MTE1", "MTE2", SIG_K_L1)
                                if kv_g1 == eff_kv_iters - 1:
                                    # ---- task-end q_l1 return (delta-b: LAST
                                    # slot; MTE1 in-order covers every q_l1 read
                                    # of the task incl. later batch-head reloads) ----
                                    T.set_flag("MTE1", "MTE2", SIG_Q_L1)
                                T.set_flag("MTE1", "M", SIG_L0AB + side)
                                # mma: Q K^T -> L0C
                                T.wait_flag("MTE1", "M", SIG_L0AB + side)
                                T.wait_flag("FIX", "M", SIG_L0C + side)
                                T.mma(l0a_s[side, :, :], l0b_s[side, :, :], l0c_s[side, :, :], init=True)
                                T.set_flag("M", "MTE1", SIG_L0AB + side)
                                T.set_flag("M", "FIX", SIG_L0C + side)
                                # L0C -> w1 (FIX)
                                T.wait_flag("M", "FIX", SIG_L0C + side)
                                T.copy(l0c_s[side, :, :], workspace_1[cid, i, :, :])
                                T.set_flag("FIX", "M", SIG_L0C + side)
                                if (i + 1) % CROSS_INTERVAL == 0 or i == batch_iters - 1:
                                    T.set_cross_flag("FIX", SEM_WS1_C2V)
                                # walk advance (pure scalar, no DMA/flags)
                                kv_g1 += 1
                                if kv_g1 == eff_kv_iters:
                                    kv_g1 = 0
                                    tl_g1 += 1

                            # --- GEMM2 batch: consume w2 (P), produce PV -> w3 ---
                            T.wait_cross_flag(SEM_WS3_V2C)
                            tl_g2 = T.alloc_var("int32", init=tl_head)
                            kv_g2 = T.alloc_var("int32", init=kv_head)
                            for i in T.serial(batch_iters):
                                side = i % 2
                                task = my_start + tl_g2
                                bx = task % s_blocks
                                by = task // s_blocks % N
                                bz = task // s_blocks // N % B
                                s0 = bx * block_M
                                eff_kv_iters = (
                                    T.floordiv(T.min(s0 + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                                )
                                idx = kv_g2
                                j0 = idx * block_N
                                valid_n = T.min(block_N, S_kv - j0)
                                # V: GM -> L1
                                T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                                T.copy(V[bz, j0 : j0 + valid_n, by, :], v_l1)
                                T.set_flag("MTE2", "MTE1", SIG_V_L1)
                                # P: w2 -> p_l1
                                T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                                if i % CROSS_INTERVAL == 0:
                                    T.wait_cross_flag(SEM_WS2_V2C)
                                T.copy(workspace_2[cid, i, :, :], p_l1)
                                T.set_flag("MTE2", "MTE1", SIG_P_L1)
                                # V: L1 -> L0B (no transpose for PV)
                                T.wait_flag("MTE2", "MTE1", SIG_V_L1)
                                T.wait_flag("M", "MTE1", SIG_L0AB + side)
                                T.copy(v_l1, l0b_o[side, :, :])
                                T.set_flag("MTE1", "MTE2", SIG_V_L1)
                                # P: L1 -> L0A
                                T.wait_flag("MTE2", "MTE1", SIG_P_L1)
                                T.copy(p_l1, l0a_o[side, :, :])
                                T.set_flag("MTE1", "MTE2", SIG_P_L1)
                                T.set_flag("MTE1", "M", SIG_L0AB + side)
                                # mma: P V -> L0C
                                T.wait_flag("MTE1", "M", SIG_L0AB + side)
                                T.wait_flag("FIX", "M", SIG_L0C + side)
                                T.mma(l0a_o[side, :, :], l0b_o[side, :, :], l0c_o[side, :, :], init=True)
                                T.set_flag("M", "MTE1", SIG_L0AB + side)
                                T.set_flag("M", "FIX", SIG_L0C + side)
                                # L0C -> w3 (FIX)
                                T.wait_flag("M", "FIX", SIG_L0C + side)
                                T.copy(l0c_o[side, :, :], workspace_3[cid, i, :, :])
                                T.set_flag("FIX", "M", SIG_L0C + side)
                                if (i + 1) % CROSS_INTERVAL == 0 or i == batch_iters - 1:
                                    T.set_cross_flag("FIX", SEM_WS3_C2V)
                                # walk advance (pure scalar, no DMA/flags)
                                kv_g2 += 1
                                if kv_g2 == eff_kv_iters:
                                    kv_g2 = 0
                                    tl_g2 += 1

                            T.set_cross_flag("MTE2", SEM_WS2_C2V)
                            # batch-head state advance (GEMM2 walk ended at the next head)
                            tl_head = tl_g2
                            kv_head = kv_g2

                    # destroy: consume outstanding init-direction flags
                    T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                    T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                    T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                    T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                    T.wait_flag("M", "MTE1", SIG_L0AB)
                    T.wait_flag("M", "MTE1", SIG_L0AB + 1)
                    T.wait_flag("FIX", "M", SIG_L0C)
                    T.wait_flag("FIX", "M", SIG_L0C + 1)

                # ==================== Vector core (shared emitter) ====================
                with T.Scope("V"):
                    emit_v(
                        io_buf,
                        io3,
                        work_ub,
                        buf_2d,
                        rt_2d,
                        r_2d,
                        mask_2d,
                        col_pos,
                        row_thresh,
                        rr_ub,
                        acc_s_half,
                        acc_o,
                        out_half,
                        m_i,
                        m_i_prev,
                        sumexp,
                        r_factors,
                        sumexp_is,
                        Output,
                        workspace_1,
                        workspace_2,
                        workspace_3,
                        cid,
                        vid,
                        my_start,
                        my_count,
                        sig_io3,
                    )

        return main

    elif form_b_mma:
        # ==================== rev4 Form B-mma (prefill S>16 & D=256): manual
        # K/N-split T.mma pipeline (§3.3/§8.3). Replaces gemm_v0's per-step
        # PipeBarrier<PIPE_M> drain (case9 scalar_mte1_stall ~78% wall,
        # §S4-3-2) with chunk-level token flow: MTE1(c+1) || M(c) || FIX(c-1)
        # three-pipe overlap (S1 gain mechanism, §6.3).
        # Structure mirrors: batch protocol = #R3-1/#R5-1 (single_kv_block
        # task batch) + Form A generic (multi-kv); GEMM2 = PLAN B (narrow
        # l0c_o + per-chunk colseg FIX — plan-a C-region mma probe P2
        # crashed, §3.2 alt row). Token accounting = §7.2-2a table: init 13
        # set / destroy 12 wait, every round self-paired, no dangling tokens,
        # no conditional DMA (#R5-3 discipline). BANNED in this prim_func:
        # T.gemm_v0 (event-id-0 collision defense, §3.4), T.barrier_all
        # (§8.6-6), threads= (§8.6-1), T.Pipelined (R9). ====================
        @T.prim_func
        def main(
            Q: T.Tensor([B, S, N, D], dtype),  # type: ignore
            K: T.Tensor([B, S_kv, N, D], dtype),  # type: ignore
            V: T.Tensor([B, S_kv, N, D], dtype),  # type: ignore
            Output: T.Tensor([B, S, N, D], dtype),  # type: ignore
            workspace_1: T.Tensor([core_num, NUM_STAGES, block_M, block_N], accum_dtype),  # type: ignore
            workspace_2: T.Tensor([core_num, NUM_STAGES, block_M, block_N], dtype),  # type: ignore
            workspace_3: T.Tensor([core_num, NUM_STAGES, block_M, D], accum_dtype),  # type: ignore
        ):
            with T.Kernel(core_num, is_npu=True) as (cid, vid):
                # ---- C-side L1 (§4.3 with the P1-probe contingency applied:
                # k_l1 is ZN-annotated, NOT NZ. Root cause (segment-localized
                # via w1 readback + generated-code offset diff): the
                # copy_gm_to_l1 template ALWAYS writes canonical zN
                # (common.h:35 LayoutL1=layout::zN, tag hardcoded), while a
                # COLSEG transpose read computes its src offset as
                # layout_map.Forward(region_min) — under an nZ annotation that
                # yields the nZ-formula offset (2048 for col 128 of [128,256])
                # which does not match the physical zN fill (correct base =
                # 16384 = zN-formula = nZ(256,128) tile-1 base). Full-buffer
                # reads (Form A) are offset-0 and layout-agnostic, which is why
                # Form A's k NZ annotation is and stays correct there. ZN here
                # = design P1 contingency "drop the k NZ annotation"
                # (sparse_flash_mla no-annotation precedent; probe-verified
                # default state), expressed as the explicit ZN annotation —
                # same Forward formula as the production-verified q/v colsegs.
                q_l1 = T.alloc_L1([block_M, D], dtype)
                k_l1 = T.alloc_L1([block_N, D], dtype)
                v_l1 = T.alloc_L1([block_N, D], dtype)
                p_l1 = T.alloc_L1([block_M, block_N], dtype)
                T.annotate_layout(
                    {
                        q_l1: make_zn_layout(q_l1),
                        k_l1: make_zn_layout(k_l1),
                        p_l1: make_zn_layout(p_l1),
                        v_l1: make_zn_layout(v_l1),
                    }
                )
                # ---- L0 explicit pools (P3-mma, §4.3/§4.5; probe P4 readout:
                # L0A 32KB / L0B 64KB shared pool exact-top / L0C 64KB plan b) ----
                l0a_s = T.alloc_L0A([2, block_M, KC], dtype)  # GEMM1 A: Q k-chunks (task-resident)
                l0a_o = T.alloc_L0A([2, block_M, block_N], dtype)  # GEMM2 A: P, side ping-pong
                l0b_pool = T.alloc_L0B([2, KC, block_N], dtype)  # SHARED: GEMM1 K^T kc / GEMM2 V nc (§4.5-d)
                l0c_s = T.alloc_L0C([2, block_M, block_N], accum_dtype)  # scores: K-split chain target
                l0c_o = T.alloc_L0C([2, block_M, NC], accum_dtype)  # PV: PLAN B narrow pool (slot=chunk)
                # ---- V-side UB (verbatim rev3 — emit_v zero-change premise) ----
                io_buf = T.alloc_ub([2, half_M, block_N], accum_dtype)
                io3 = io_buf if io3_is_alias else T.alloc_ub([2, half_M, D], accum_dtype)
                work_ub = T.alloc_ub([half_M, block_N], accum_dtype)
                buf_2d = T.alloc_ub([half_M, block_N], accum_dtype)
                rt_2d = T.alloc_ub([half_M, block_N], accum_dtype)
                r_2d = buf_2d if block_N == D else T.alloc_ub([half_M, D], accum_dtype)
                mask_2d = T.alloc_ub([half_M * block_N // 8], "uint8")
                col_pos = T.alloc_ub([block_N], accum_dtype)
                row_thresh = T.alloc_ub([half_M], accum_dtype)
                rr_ub = T.alloc_ub([half_M, block_N], dtype)
                acc_s_half = T.alloc_ub([half_M, block_N], dtype)
                acc_o = T.alloc_ub([half_M, D], accum_dtype)
                out_half = T.alloc_ub([half_M, D], dtype)
                m_i = T.alloc_ub([half_M, 1], accum_dtype)
                m_i_prev = T.alloc_ub([half_M, 1], accum_dtype)
                sumexp = T.alloc_ub([half_M, 1], accum_dtype)
                r_factors = T.alloc_ub([NUM_STAGES, half_M, 1], accum_dtype)
                sumexp_is = T.alloc_ub([NUM_STAGES, half_M, 1], accum_dtype)

                my_start = cid * q_tasks + T.if_then_else(cid < r_tasks, cid, r_tasks)
                my_count = q_tasks + T.if_then_else(cid < r_tasks, 1, 0)

                # ==================== Cube core (Form B-mma) ====================
                with T.Scope("C"):
                    # init (§7.2-4 extended): 13 pre-set = 1 SEM + 4 L1 + 2 L0KC
                    # + 2 L0AO + 2 L0AS + 2 L0C (§7.2-2a accounting table)
                    T.set_cross_flag("MTE2", SEM_WS2_C2V)
                    T.set_flag("MTE1", "MTE2", SIG_K_L1)
                    T.set_flag("MTE1", "MTE2", SIG_P_L1)
                    T.set_flag("MTE1", "MTE2", SIG_V_L1)
                    T.set_flag("MTE1", "MTE2", SIG_Q_L1)
                    T.set_flag("M", "MTE1", SIG_L0KC)
                    T.set_flag("M", "MTE1", SIG_L0KC + 1)
                    T.set_flag("M", "MTE1", SIG_L0AO)
                    T.set_flag("M", "MTE1", SIG_L0AO + 1)
                    T.set_flag("M", "MTE1", SIG_L0AS)
                    T.set_flag("M", "MTE1", SIG_L0AS + 1)
                    T.set_flag("FIX", "M", SIG_L0C)
                    T.set_flag("FIX", "M", SIG_L0C + 1)

                    if single_kv_block:
                        # ---- #R3-1/#R5-1 task-batched single-KV-block C path
                        # (protocol mirror + mma body; L0 #10 path). Slot i <->
                        # task my_start + kb*NUM_STAGES + i; num_tb/batch_tasks
                        # from the SAME formulas as the V mirror (§7.2-5). ----
                        num_tb = T.ceildiv(my_count, NUM_STAGES)
                        for kb in T.serial(num_tb):
                            _remaining_t = my_count - kb * NUM_STAGES
                            batch_tasks = T.if_then_else(_remaining_t < NUM_STAGES, _remaining_t, NUM_STAGES)
                            valid_n = min(block_N, S_kv)  # single block: idx == 0

                            # --- GEMM1 batch: per-task scores -> w1 slot i ---
                            T.wait_cross_flag(SEM_WS1_V2C)
                            for i in T.serial(batch_tasks):
                                side = i % 2
                                task = my_start + kb * NUM_STAGES + i
                                bx = task % s_blocks
                                by = task // s_blocks % N
                                bz = task // s_blocks // N % B
                                s0 = bx * block_M
                                valid_m = T.min(block_M, S - s0)
                                # Q: GM -> L1 per-slot (#R5-1 delta-a mirror)
                                T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                                T.copy(Q[bz, s0 : s0 + valid_m, by, :], q_l1)
                                T.set_flag("MTE2", "MTE1", SIG_Q_L1)
                                T.wait_flag("MTE2", "MTE1", SIG_Q_L1)
                                # K: GM -> L1 (single block => j0 == 0)
                                T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                                T.copy(K[bz, 0:valid_n, by, :], k_l1)
                                T.set_flag("MTE2", "MTE1", SIG_K_L1)
                                # Q: L1 colsegs -> L0A, EVERY slot (#R5-1 delta-b
                                # mirror: Q is task-dependent here). SIG_L0AS one
                                # round per slot (§3.3 token-flow note): the
                                # trailing M waits move the M-pipe block to a
                                # point with no M work in flight (zero loss);
                                # task-internal mma reads of l0a_s are covered by
                                # M-pipe in-order — zero SIG_L0AS in the chain.
                                T.wait_flag("M", "MTE1", SIG_L0AS)
                                T.copy(q_l1[:, 0:KC], l0a_s[0, :, :])
                                T.wait_flag("M", "MTE1", SIG_L0AS + 1)
                                T.copy(q_l1[:, KC : 2 * KC], l0a_s[1, :, :])
                                T.set_flag("MTE1", "M", SIG_L0AS)
                                T.set_flag("MTE1", "M", SIG_L0AS + 1)
                                T.wait_flag("MTE1", "M", SIG_L0AS)
                                T.wait_flag("MTE1", "M", SIG_L0AS + 1)
                                T.set_flag("MTE1", "MTE2", SIG_Q_L1)  # q_l1 MTE1 read done
                                # K-split mma chain (GEMM1_BODY §8.3 inline, c=0
                                # init=True -> c=1 init=False; probe P3 16/16
                                # BITEXACT vs gemm_v0 kL0Size=128)
                                T.wait_flag("MTE2", "MTE1", SIG_K_L1)
                                # ---- k-chunk c=0 (K columns 0:KC) ----
                                T.wait_flag("M", "MTE1", SIG_L0KC)  # pool chunk 0 released
                                T.copy(k_l1[:, 0:KC], l0b_pool[0, :, :], transpose=True)
                                T.set_flag("MTE1", "M", SIG_L0KC)
                                T.wait_flag("MTE1", "M", SIG_L0KC)
                                T.wait_flag("FIX", "M", SIG_L0C + side)  # l0c_s slot writable
                                T.mma(l0a_s[0, :, :], l0b_pool[0, :, :], l0c_s[side, :, :], init=True)
                                T.set_flag("M", "MTE1", SIG_L0KC)  # pool chunk 0 released
                                # ---- k-chunk c=1 (K columns KC:2*KC, chained) ----
                                T.wait_flag("M", "MTE1", SIG_L0KC + 1)
                                T.copy(k_l1[:, KC : 2 * KC], l0b_pool[1, :, :], transpose=True)
                                T.set_flag("MTE1", "M", SIG_L0KC + 1)
                                T.wait_flag("MTE1", "M", SIG_L0KC + 1)
                                T.mma(l0a_s[1, :, :], l0b_pool[1, :, :], l0c_s[side, :, :], init=False)
                                T.set_flag("M", "MTE1", SIG_L0KC + 1)
                                T.set_flag("M", "FIX", SIG_L0C + side)  # scores full block ready
                                T.set_flag("MTE1", "MTE2", SIG_K_L1)  # k_l1 read done (after c1 load)
                                # L0C -> w1 (FIX full block)
                                T.wait_flag("M", "FIX", SIG_L0C + side)
                                T.copy(l0c_s[side, :, :], workspace_1[cid, i, :, :])
                                T.set_flag("FIX", "M", SIG_L0C + side)
                                # slot-end l0a_s return (M-pipe order: this slot's
                                # last GEMM1 mma has read l0a_s; GEMM2 never reads
                                # it — next slot head wait consumes; net 0/slot)
                                T.set_flag("M", "MTE1", SIG_L0AS)
                                T.set_flag("M", "MTE1", SIG_L0AS + 1)
                                if (i + 1) % CROSS_INTERVAL == 0 or i == batch_tasks - 1:
                                    T.set_cross_flag("FIX", SEM_WS1_C2V)

                            # --- GEMM2 batch: per-task P (w2 slot i) @ V -> w3 slot i
                            # (v1 keeps w3+phase-2: emit_v `if not form_a:` gate
                            # verbatim => V-side zero change, §0.4-b-#R6-1) ---
                            T.wait_cross_flag(SEM_WS3_V2C)
                            for i in T.serial(batch_tasks):
                                side = i % 2
                                task = my_start + kb * NUM_STAGES + i
                                by = task // s_blocks % N
                                bz = task // s_blocks // N % B
                                # V: GM -> L1
                                T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                                T.copy(V[bz, 0:valid_n, by, :], v_l1)
                                T.set_flag("MTE2", "MTE1", SIG_V_L1)
                                # P: w2 -> p_l1
                                T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                                if i % CROSS_INTERVAL == 0:
                                    T.wait_cross_flag(SEM_WS2_V2C)
                                T.copy(workspace_2[cid, i, :, :], p_l1)
                                T.set_flag("MTE2", "MTE1", SIG_P_L1)
                                # P: L1 -> L0A (side ping-pong)
                                T.wait_flag("M", "MTE1", SIG_L0AO + side)
                                T.wait_flag("MTE2", "MTE1", SIG_P_L1)
                                T.copy(p_l1, l0a_o[side, :, :])
                                T.set_flag("MTE1", "MTE2", SIG_P_L1)
                                T.set_flag("MTE1", "M", SIG_L0AO + side)
                                T.wait_flag("MTE2", "MTE1", SIG_V_L1)
                                # N-split mma + per-chunk colseg FIX (GEMM2_BODY
                                # PLAN B §3.2/§8.3 inline; probe P2b/P5 16/16
                                # BITEXACT vs gemm_v0 full-N)
                                # ---- n-chunk c=0 (V cols 0:NC -> w3 cols 0:NC) ----
                                T.wait_flag("M", "MTE1", SIG_L0KC)
                                T.copy(v_l1[:, 0:NC], l0b_pool[0, :, :])  # direct, no transpose
                                T.set_flag("MTE1", "M", SIG_L0KC)
                                T.wait_flag("MTE1", "M", SIG_L0KC)
                                T.wait_flag("MTE1", "M", SIG_L0AO + side)  # p ready
                                T.wait_flag("FIX", "M", SIG_L0C)  # narrow slot 0 writable
                                T.mma(l0a_o[side, :, :], l0b_pool[0, :, :], l0c_o[0, :, :], init=True)
                                T.set_flag("M", "MTE1", SIG_L0KC)
                                T.set_flag("M", "FIX", SIG_L0C)  # chunk 0 ready
                                T.wait_flag("M", "FIX", SIG_L0C)
                                T.copy(l0c_o[0, :, :], workspace_3[cid, i, :, 0:NC])  # colseg FIX
                                T.set_flag("FIX", "M", SIG_L0C)
                                # ---- n-chunk c=1 (V cols NC:2*NC -> w3 cols NC:2*NC) ----
                                T.wait_flag("M", "MTE1", SIG_L0KC + 1)
                                T.copy(v_l1[:, NC : 2 * NC], l0b_pool[1, :, :])
                                T.set_flag("MTE1", "M", SIG_L0KC + 1)
                                T.wait_flag("MTE1", "M", SIG_L0KC + 1)
                                T.wait_flag("FIX", "M", SIG_L0C + 1)  # narrow slot 1 writable
                                T.mma(l0a_o[side, :, :], l0b_pool[1, :, :], l0c_o[1, :, :], init=True)
                                T.set_flag("M", "MTE1", SIG_L0KC + 1)
                                T.set_flag("M", "FIX", SIG_L0C + 1)
                                T.set_flag("M", "MTE1", SIG_L0AO + side)  # l0a_o both chunks read
                                T.wait_flag("M", "FIX", SIG_L0C + 1)
                                T.copy(l0c_o[1, :, :], workspace_3[cid, i, :, NC : 2 * NC])
                                T.set_flag("FIX", "M", SIG_L0C + 1)
                                T.set_flag("MTE1", "MTE2", SIG_V_L1)  # v_l1 read done (both chunks)
                                if (i + 1) % CROSS_INTERVAL == 0 or i == batch_tasks - 1:
                                    # after BOTH colseg FIX writes (FIX pipe in-order)
                                    T.set_cross_flag("FIX", SEM_WS3_C2V)

                            T.set_cross_flag("MTE2", SEM_WS2_C2V)

                    else:
                        # ---- #R8-2 (session-8 Stage-3) FLAT (task,kv) continuous
                        # token stream (form_b_mma generic domain = case9/case4;
                        # replaces the per-task nested [task][outer-batch] loops).
                        # Structure mirrors the emit_v flat branch VERBATIM
                        # (§7.2-5 C/V same-formula): one-time O(my_count) scalar
                        # walk builds total_flat = Σ eff_kv(t); flat batches of
                        # NUM_STAGES slots; per-slot sequential walk state
                        # (T.alloc_var) maps slot -> (task,kv) with RUNTIME-
                        # nonuniform causal eff_kv (prefill delta vs the #R4-3
                        # decode blueprint's compile-time-uniform mapping).
                        # Task switch (kv==0): Q GM->L1 + l0a_s preload — the
                        # FULL SIG_Q_L1 round + SIG_L0AS round live INSIDE the
                        # runtime if (wait/set enter-exit together, per-task
                        # net 0 = #R5-1 accounting migrated from task granularity
                        # to switch granularity; #R5-3 conditional-DMA ban
                        # compliance: DMA and ALL its flags are co-conditioned).
                        # Task end (kv==eff-1): L0AS return sets (M-pipe order:
                        # after this task's last GEMM1 mma read). Batch
                        # structure preserved: GEMM1 batch fully precedes GEMM2
                        # batch (§8.6-10 shared-pool phase discipline, no
                        # iter-level interleave). Per-slot bodies = generic
                        # verbatim (BITEXACT expected, probe-gated). ----
                        total_flat = T.alloc_var("int32", init=0)
                        for t in T.serial(my_count):
                            _tw = my_start + t
                            _s0w = (_tw % s_blocks) * block_M
                            _effw = T.floordiv(T.min(_s0w + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                            total_flat += _effw
                        num_fb = T.ceildiv(total_flat, NUM_STAGES)
                        tl_head = T.alloc_var("int32", init=0)  # persistent batch-head walk state
                        kv_head = T.alloc_var("int32", init=0)
                        for fb in T.serial(num_fb):
                            _rem_f = total_flat - fb * NUM_STAGES
                            batch_iters = T.if_then_else(_rem_f < NUM_STAGES, _rem_f, NUM_STAGES)

                            # --- GEMM1 batch: flat slots -> w1 ---
                            T.wait_cross_flag(SEM_WS1_V2C)
                            tl_g1 = T.alloc_var("int32", init=tl_head)
                            kv_g1 = T.alloc_var("int32", init=kv_head)
                            for i in T.serial(batch_iters):
                                side = i % 2
                                task = my_start + tl_g1
                                bx = task % s_blocks
                                by = task // s_blocks % N
                                bz = task // s_blocks // N % B
                                s0 = bx * block_M
                                valid_m = T.min(block_M, S - s0)
                                eff_kv_iters = (
                                    T.floordiv(T.min(s0 + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                                )
                                idx = kv_g1
                                j0 = idx * block_N
                                valid_n = T.min(block_N, S_kv - j0)
                                if kv_g1 == 0:
                                    # ---- task switch: Q load (SIG_Q_L1 round) +
                                    # l0a_s preload (SIG_L0AS round), full rounds
                                    # inside the condition (同进同出) ----
                                    T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                                    T.copy(Q[bz, s0 : s0 + valid_m, by, :], q_l1)
                                    T.set_flag("MTE2", "MTE1", SIG_Q_L1)
                                    T.wait_flag("MTE2", "MTE1", SIG_Q_L1)
                                    T.wait_flag("M", "MTE1", SIG_L0AS)
                                    T.copy(q_l1[:, 0:KC], l0a_s[0, :, :])
                                    T.wait_flag("M", "MTE1", SIG_L0AS + 1)
                                    T.copy(q_l1[:, KC : 2 * KC], l0a_s[1, :, :])
                                    T.set_flag("MTE1", "M", SIG_L0AS)
                                    T.set_flag("MTE1", "M", SIG_L0AS + 1)
                                    T.wait_flag("MTE1", "M", SIG_L0AS)
                                    T.wait_flag("MTE1", "M", SIG_L0AS + 1)
                                    T.set_flag("MTE1", "MTE2", SIG_Q_L1)  # q_l1 MTE1 read done
                                # K: GM -> L1 (BSHD row-stride, runtime extent)
                                T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                                T.copy(K[bz, j0 : j0 + valid_n, by, :], k_l1)
                                T.set_flag("MTE2", "MTE1", SIG_K_L1)
                                T.wait_flag("MTE2", "MTE1", SIG_K_L1)
                                # K-split mma chain (GEMM1_BODY §8.3 inline,
                                # c=0 init=True -> c=1 init=False; probe P3)
                                # ---- k-chunk c=0 (K columns 0:KC) ----
                                T.wait_flag("M", "MTE1", SIG_L0KC)  # pool chunk 0 released
                                T.copy(k_l1[:, 0:KC], l0b_pool[0, :, :], transpose=True)
                                T.set_flag("MTE1", "M", SIG_L0KC)
                                T.wait_flag("MTE1", "M", SIG_L0KC)
                                T.wait_flag("FIX", "M", SIG_L0C + side)  # l0c_s slot writable
                                T.mma(l0a_s[0, :, :], l0b_pool[0, :, :], l0c_s[side, :, :], init=True)
                                T.set_flag("M", "MTE1", SIG_L0KC)  # pool chunk 0 released
                                # ---- k-chunk c=1 (K columns KC:2*KC, chained) ----
                                T.wait_flag("M", "MTE1", SIG_L0KC + 1)
                                T.copy(k_l1[:, KC : 2 * KC], l0b_pool[1, :, :], transpose=True)
                                T.set_flag("MTE1", "M", SIG_L0KC + 1)
                                T.wait_flag("MTE1", "M", SIG_L0KC + 1)
                                T.mma(l0a_s[1, :, :], l0b_pool[1, :, :], l0c_s[side, :, :], init=False)
                                T.set_flag("M", "MTE1", SIG_L0KC + 1)
                                T.set_flag("M", "FIX", SIG_L0C + side)  # scores full block ready
                                T.set_flag("MTE1", "MTE2", SIG_K_L1)  # k_l1 read done
                                # L0C -> w1 (FIX full block)
                                T.wait_flag("M", "FIX", SIG_L0C + side)
                                T.copy(l0c_s[side, :, :], workspace_1[cid, i, :, :])
                                T.set_flag("FIX", "M", SIG_L0C + side)
                                if kv_g1 == eff_kv_iters - 1:
                                    # ---- task end: l0a_s return (M-pipe order:
                                    # this task's last GEMM1 mma has read l0a_s;
                                    # next switch's head wait consumes; GEMM2
                                    # never reads l0a_s — one round per task) ----
                                    T.set_flag("M", "MTE1", SIG_L0AS)
                                    T.set_flag("M", "MTE1", SIG_L0AS + 1)
                                if (i + 1) % CROSS_INTERVAL == 0 or i == batch_iters - 1:
                                    T.set_cross_flag("FIX", SEM_WS1_C2V)
                                # walk advance (pure scalar, no DMA/flags)
                                kv_g1 += 1
                                if kv_g1 == eff_kv_iters:
                                    kv_g1 = 0
                                    tl_g1 += 1

                            # --- GEMM2 batch: consume w2 (P), produce PV -> w3 ---
                            T.wait_cross_flag(SEM_WS3_V2C)
                            tl_g2 = T.alloc_var("int32", init=tl_head)
                            kv_g2 = T.alloc_var("int32", init=kv_head)
                            for i in T.serial(batch_iters):
                                side = i % 2
                                task = my_start + tl_g2
                                bx = task % s_blocks
                                by = task // s_blocks % N
                                bz = task // s_blocks // N % B
                                s0 = bx * block_M
                                eff_kv_iters = (
                                    T.floordiv(T.min(s0 + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                                )
                                idx = kv_g2
                                j0 = idx * block_N
                                valid_n = T.min(block_N, S_kv - j0)
                                # V: GM -> L1
                                T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                                T.copy(V[bz, j0 : j0 + valid_n, by, :], v_l1)
                                T.set_flag("MTE2", "MTE1", SIG_V_L1)
                                # P: w2 -> p_l1
                                T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                                if i % CROSS_INTERVAL == 0:
                                    T.wait_cross_flag(SEM_WS2_V2C)
                                T.copy(workspace_2[cid, i, :, :], p_l1)
                                T.set_flag("MTE2", "MTE1", SIG_P_L1)
                                # P: L1 -> L0A (side ping-pong)
                                T.wait_flag("M", "MTE1", SIG_L0AO + side)
                                T.wait_flag("MTE2", "MTE1", SIG_P_L1)
                                T.copy(p_l1, l0a_o[side, :, :])
                                T.set_flag("MTE1", "MTE2", SIG_P_L1)
                                T.set_flag("MTE1", "M", SIG_L0AO + side)
                                T.wait_flag("MTE2", "MTE1", SIG_V_L1)
                                # N-split mma + per-chunk colseg FIX
                                # (GEMM2_BODY PLAN B §3.2/§8.3 inline; probe
                                # P2b/P5 16/16 BITEXACT vs gemm_v0 full-N)
                                # ---- n-chunk c=0 (V cols 0:NC -> w3 cols 0:NC) ----
                                T.wait_flag("M", "MTE1", SIG_L0KC)
                                T.copy(v_l1[:, 0:NC], l0b_pool[0, :, :])  # direct, no transpose
                                T.set_flag("MTE1", "M", SIG_L0KC)
                                T.wait_flag("MTE1", "M", SIG_L0KC)
                                T.wait_flag("MTE1", "M", SIG_L0AO + side)  # p ready
                                T.wait_flag("FIX", "M", SIG_L0C)  # narrow slot 0 writable
                                T.mma(l0a_o[side, :, :], l0b_pool[0, :, :], l0c_o[0, :, :], init=True)
                                T.set_flag("M", "MTE1", SIG_L0KC)
                                T.set_flag("M", "FIX", SIG_L0C)  # chunk 0 ready
                                T.wait_flag("M", "FIX", SIG_L0C)
                                T.copy(l0c_o[0, :, :], workspace_3[cid, i, :, 0:NC])  # colseg FIX
                                T.set_flag("FIX", "M", SIG_L0C)
                                # ---- n-chunk c=1 (V cols NC:2*NC -> w3 cols NC:2*NC) ----
                                T.wait_flag("M", "MTE1", SIG_L0KC + 1)
                                T.copy(v_l1[:, NC : 2 * NC], l0b_pool[1, :, :])
                                T.set_flag("MTE1", "M", SIG_L0KC + 1)
                                T.wait_flag("MTE1", "M", SIG_L0KC + 1)
                                T.wait_flag("FIX", "M", SIG_L0C + 1)  # narrow slot 1 writable
                                T.mma(l0a_o[side, :, :], l0b_pool[1, :, :], l0c_o[1, :, :], init=True)
                                T.set_flag("M", "MTE1", SIG_L0KC + 1)
                                T.set_flag("M", "FIX", SIG_L0C + 1)
                                T.set_flag("M", "MTE1", SIG_L0AO + side)  # l0a_o both chunks read
                                T.wait_flag("M", "FIX", SIG_L0C + 1)
                                T.copy(l0c_o[1, :, :], workspace_3[cid, i, :, NC : 2 * NC])
                                T.set_flag("FIX", "M", SIG_L0C + 1)
                                T.set_flag("MTE1", "MTE2", SIG_V_L1)  # v_l1 read done
                                if (i + 1) % CROSS_INTERVAL == 0 or i == batch_iters - 1:
                                    # after BOTH colseg FIX writes (FIX pipe in-order)
                                    T.set_cross_flag("FIX", SEM_WS3_C2V)
                                # walk advance (pure scalar, no DMA/flags)
                                kv_g2 += 1
                                if kv_g2 == eff_kv_iters:
                                    kv_g2 = 0
                                    tl_g2 += 1

                            T.set_cross_flag("MTE2", SEM_WS2_C2V)
                            # batch-head state advance (GEMM2 walk ended at the next head)
                            tl_head = tl_g2
                            kv_head = kv_g2

                    # destroy (§7.2-4): symmetric wait x12
                    T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                    T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                    T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                    T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                    T.wait_flag("M", "MTE1", SIG_L0KC)
                    T.wait_flag("M", "MTE1", SIG_L0KC + 1)
                    T.wait_flag("M", "MTE1", SIG_L0AO)
                    T.wait_flag("M", "MTE1", SIG_L0AO + 1)
                    T.wait_flag("M", "MTE1", SIG_L0AS)
                    T.wait_flag("M", "MTE1", SIG_L0AS + 1)
                    T.wait_flag("FIX", "M", SIG_L0C)
                    T.wait_flag("FIX", "M", SIG_L0C + 1)

                # ==================== Vector core (shared emitter, ZERO change) ====================
                with T.Scope("V"):
                    emit_v(
                        io_buf,
                        io3,
                        work_ub,
                        buf_2d,
                        rt_2d,
                        r_2d,
                        mask_2d,
                        col_pos,
                        row_thresh,
                        rr_ub,
                        acc_s_half,
                        acc_o,
                        out_half,
                        m_i,
                        m_i_prev,
                        sumexp,
                        r_factors,
                        sumexp_is,
                        Output,
                        workspace_1,
                        workspace_2,
                        workspace_3,
                        cid,
                        vid,
                        my_start,
                        my_count,
                        sig_io3,
                    )

        return main

    else:

        @T.prim_func
        def main(
            Q: T.Tensor([B, S, N, D], dtype),  # type: ignore
            K: T.Tensor([B, S_kv, N, D], dtype),  # type: ignore
            V: T.Tensor([B, S_kv, N, D], dtype),  # type: ignore
            Output: T.Tensor([B, S, N, D], dtype),  # type: ignore
            workspace_1: T.Tensor([core_num, NUM_STAGES, block_M, block_N], accum_dtype),  # type: ignore
            workspace_2: T.Tensor([core_num, NUM_STAGES, block_M, block_N], dtype),  # type: ignore
            workspace_3: T.Tensor([core_num, NUM_STAGES, block_M, D], accum_dtype),  # type: ignore
        ):
            with T.Kernel(core_num, is_npu=True) as (cid, vid):
                # ---- C-side L1 (no annotation: gemm_v0 handles L0 internally) ----
                q_l1 = T.alloc_L1([block_M, D], dtype)
                # #R11-1: K-side double slot (kv_db domain only; 3D no-annotate
                # + runtime slice == 2D serial, S4-3-4②/probe_r42b mechanism).
                k_l1 = T.alloc_L1([2, block_N, D], dtype) if kv_db else T.alloc_L1([block_N, D], dtype)
                v_l1 = T.alloc_L1([block_N, D], dtype)
                p_l1 = T.alloc_L1([block_M, block_N], dtype)
                # ---- L0C single slots (gemm_v0 manages L0A/L0B by kL0Size) ----
                acc_s_l0c = T.alloc_L0C([block_M, block_N], accum_dtype)
                acc_o_l0c = T.alloc_L0C([block_M, D], accum_dtype)
                # ---- V-side UB (identical to Form A; §4.3 + adaptations) ----
                io_buf = T.alloc_ub([2, half_M, block_N], accum_dtype)
                io3 = io_buf if io3_is_alias else T.alloc_ub([2, half_M, D], accum_dtype)
                work_ub = T.alloc_ub([half_M, block_N], accum_dtype)
                buf_2d = T.alloc_ub([half_M, block_N], accum_dtype)
                rt_2d = T.alloc_ub([half_M, block_N], accum_dtype)
                r_2d = buf_2d if block_N == D else T.alloc_ub([half_M, D], accum_dtype)
                mask_2d = T.alloc_ub([half_M * block_N // 8], "uint8")
                col_pos = T.alloc_ub([block_N], accum_dtype)
                row_thresh = T.alloc_ub([half_M], accum_dtype)
                rr_ub = T.alloc_ub([half_M, block_N], dtype)
                acc_s_half = T.alloc_ub([half_M, block_N], dtype)
                acc_o = T.alloc_ub([half_M, D], accum_dtype)
                out_half = T.alloc_ub([half_M, D], dtype)
                m_i = T.alloc_ub([half_M, 1], accum_dtype)
                m_i_prev = T.alloc_ub([half_M, 1], accum_dtype)
                sumexp = T.alloc_ub([half_M, 1], accum_dtype)
                r_factors = T.alloc_ub([NUM_STAGES, half_M, 1], accum_dtype)
                sumexp_is = T.alloc_ub([NUM_STAGES, half_M, 1], accum_dtype)

                my_start = cid * q_tasks + T.if_then_else(cid < r_tasks, cid, r_tasks)
                my_count = q_tasks + T.if_then_else(cid < r_tasks, 1, 0)

                # ==================== Cube core (Form B: gemm_v0, barrier-free #R4-1) ====================
                with T.Scope("C"):
                    # init (§7.2-4): pretend the w2 slots are already released by
                    # the consumer — V's first phase-1 batch waits SEM_WS2_C2V.
                    # (No SIG_* inits: Form B has no intra-core user flags, §7.2-2.)
                    T.set_cross_flag("MTE2", SEM_WS2_C2V)
                    if single_kv_block:
                        # ---- #R3-1 task-batched single-KV-block C path ----
                        # One protocol round batches NUM_STAGES TASKS (kv_iters==1
                        # => every task is exactly one KV block): slot i <-> task
                        # my_start + kb*NUM_STAGES + i. Per-slot ops identical to
                        # the generic flow; V mirror derives num_tb/batch_tasks
                        # from the SAME my_start/my_count formulas (§7.2-5).
                        num_tb = T.ceildiv(my_count, NUM_STAGES)
                        for kb in T.serial(num_tb):
                            _remaining_t = my_count - kb * NUM_STAGES
                            batch_tasks = T.if_then_else(_remaining_t < NUM_STAGES, _remaining_t, NUM_STAGES)
                            valid_n = min(block_N, S_kv)  # single block: idx == 0

                            # --- GEMM1 batch: per-task scores -> w1 slot i ---
                            T.wait_cross_flag(SEM_WS1_V2C)
                            for i in T.serial(batch_tasks):
                                task = my_start + kb * NUM_STAGES + i
                                bx = task % s_blocks
                                by = task // s_blocks % N
                                bz = task // s_blocks // N % B
                                s0 = bx * block_M
                                valid_m = T.min(block_M, S - s0)
                                # #R4-1 barrier-free: gemm_v0 prologue orders prior
                                # MTE2 (Q/K loads) before its MTE1 reads and prior FIX
                                # (w1/w3 writes) before its mma; epilogue orders MTE1
                                # before subsequent MTE2 (L1 slot reuse) and mma before
                                # subsequent FIX (L0C readout). Same-pipe ops are
                                # in-order; cross-core hazards stay on the SEM protocol
                                # (set/wait positions unchanged). See S4-Step 2.
                                T.copy(Q[bz, s0 : s0 + valid_m, by, :], q_l1)
                                T.copy(K[bz, 0:valid_n, by, :], k_l1)
                                T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True, kL0Size=gemm1_kL0)
                                T.copy(acc_s_l0c, workspace_1[cid, i, :, :])
                                if (i + 1) % CROSS_INTERVAL == 0 or i == batch_tasks - 1:
                                    T.set_cross_flag("FIX", SEM_WS1_C2V)

                            # --- GEMM2 batch: per-task P (w2 slot i) @ V -> w3 slot i ---
                            T.wait_cross_flag(SEM_WS3_V2C)
                            for i in T.serial(batch_tasks):
                                task = my_start + kb * NUM_STAGES + i
                                by = task // s_blocks % N
                                bz = task // s_blocks // N % B
                                T.copy(V[bz, 0:valid_n, by, :], v_l1)
                                if i % CROSS_INTERVAL == 0:
                                    T.wait_cross_flag(SEM_WS2_V2C)
                                T.copy(workspace_2[cid, i, :, :], p_l1)
                                T.gemm_v0(p_l1, v_l1, acc_o_l0c, init=True)
                                T.copy(acc_o_l0c, workspace_3[cid, i, :, :])
                                if (i + 1) % CROSS_INTERVAL == 0 or i == batch_tasks - 1:
                                    T.set_cross_flag("FIX", SEM_WS3_C2V)

                            T.set_cross_flag("MTE2", SEM_WS2_C2V)
                    elif kv_db:
                        # ---- #R12-2 (A10): per-buffer MTE2 sync + pre-call
                        # K prefetch (gemm_v0 opt-in mte2_sync=SIG_*). Breaks
                        # #R11-1 conservation: the prologue waits only the
                        # caller FIFO-prefix flag; prefetches enqueued after a
                        # Set keep streaming during the gemm call.
                        # K slots: side = idx % 2 (idx = k*NUM_STAGES + i; NS
                        # even => round/task head slots are always side 0).
                        # K load issue sites (#prefetch == #consumed ==
                        # Sum eff_kv_iters, #R4-2b accounting identity kept):
                        #   S0 task head (t==0)   -> Q(0), K_0 -> slot 0, Set SIG_KDB0
                        #   S1 iter idx PRE-CALL  -> K_{idx+1} (guard idx+1<eff,
                        #      GLOBAL => crosses rounds; #R11-1's round-end site
                        #      is redundant and removed) -> slot (idx+1)%2,
                        #      Set SIG_KDB0+(idx+1)%2 (runtime id, form_a prec.)
                        #   S4 task end (t+1<my_count) -> Q(t+1) THEN K'_0 (FIFO:
                        #      Set prefix must cover Q) -> slot 0, Set SIG_KDB0
                        # G1 consume: parity-unrolled call sites bind the
                        # compile-time mte2_sync template arg (runtime template
                        # args do not exist; runtime-id Sets do -- form_a prec.).
                        # PRE-CALL placement (vs #R11-1 after-return) is what
                        # gives K(idx+1) a full gemm period to stream: after-
                        # return issue == next-entry issue point minus epsilon
                        # ONLY when scalar exits at load pace; under the old
                        # pipe-level drain it was timing-neutral (measured).
                        # V side: v_l1 single slot => a V load can never be
                        # issued before the gemm2 that reads it returned
                        # (epilogue MTE1_MTE2 release). Sites:
                        #   S3 G1-batch tail (i==batch_iters-1) -> V(round idx0)
                        #      (after ALL this round's K issues = FIFO consume
                        #      order; v_l1 released by the previous G2 batch's
                        #      last epilogue, a full round earlier; no Set --
                        #      its Set is S6's at G2 iter 0)
                        #   S5 G2 iter idx after-return -> V_{idx+1} (guard
                        #      i+1<batch_iters LOCAL: round-head V comes from
                        #      S3, never double-loaded)
                        #   S6 G2 iter idx: SEM_WS2 wait + P copy at verbatim
                        #      positions, then Set SIG_VP (FIFO prefix covers
                        #      V(idx)+P(idx)), gemm(mte2_sync=SIG_VP)
                        # Slot release unchanged: gemm epilogue SetFlag<
                        # MTE1_MTE2> orders every MTE2 issued after a call
                        # behind that call's MTE1 reads -- S1's K(idx+1)
                        # overwrites the slot gemm(idx-1) read and is issued
                        # after gemm(idx-1)'s call => ordered. Q(t+1) at S4:
                        # q_l1 released by task t's last G1 epilogue (G1 phase
                        # precedes the whole G2 phase). SEM protocol positions
                        # verbatim; #R6-2 (7): S4 priming sits after the task's
                        # last set_cross_flag("MTE2", SEM_WS2_C2V), and at that
                        # issue point the MTE2 queue is empty (the last P copy
                        # is 8KB and the round's K/V loads retired during the
                        # ~18us G2 phase) => no w2-release delay.
                        for t in T.serial(my_count):
                            task = my_start + t
                            bx = task % s_blocks
                            by = task // s_blocks % N
                            bz = task // s_blocks // N % B
                            s0 = bx * block_M
                            valid_m = T.min(block_M, S - s0)
                            eff_kv_iters = (
                                T.floordiv(T.min(s0 + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                            )
                            eff_num_outer = T.ceildiv(eff_kv_iters, NUM_STAGES)

                            if t == 0:
                                # S0: task-0 head priming -- Q FIRST, then K_0,
                                # then Set (FIFO prefix => SIG_KDB0 covers both
                                # G1(0) operands). t>0: Q(t)+K_0(t) were primed
                                # at task t-1's end (S4), so the Q copy here is
                                # t==0-guarded (no double load).
                                hvalid = T.min(block_N, S_kv)
                                T.copy(Q[bz, s0 : s0 + valid_m, by, :], q_l1)
                                T.copy(K[bz, 0:hvalid, by, :], k_l1[0, :, :])
                                T.set_flag("MTE2", "MTE1", SIG_KDB0)

                            for k in T.serial(eff_num_outer):
                                _remaining = eff_kv_iters - k * NUM_STAGES
                                batch_iters = T.if_then_else(_remaining < NUM_STAGES, _remaining, NUM_STAGES)

                                # --- GEMM1 batch: scores -> w1 ---
                                T.wait_cross_flag(SEM_WS1_V2C)
                                for i in T.serial(batch_iters):
                                    idx = k * NUM_STAGES + i
                                    # S1: PRE-CALL prefetch K(idx+1) + Set
                                    # (copy and Set share one guard => 1:1 with
                                    # the G1(idx+1) wait; co-conditioned DMA
                                    # discipline #R5-3).
                                    if idx + 1 < eff_kv_iters:
                                        nidx = idx + 1
                                        nj0 = nidx * block_N
                                        nvalid = T.min(block_N, S_kv - nj0)
                                        T.copy(K[bz, nj0 : nj0 + nvalid, by, :], k_l1[nidx % 2, :, :])
                                        T.set_flag("MTE2", "MTE1", SIG_KDB0 + nidx % 2)
                                    # G1 consume: parity-unrolled compile-time
                                    # mte2_sync (operands Q+K(idx): both issued
                                    # before this id's Set => prefix-covered).
                                    if idx % 2 == 0:
                                        T.gemm_v0(
                                            q_l1,
                                            k_l1[0, :, :],
                                            acc_s_l0c,
                                            transpose_B=True,
                                            init=True,
                                            kL0Size=gemm1_kL0,
                                            mte2_sync=SIG_KDB0,
                                        )
                                    else:
                                        T.gemm_v0(
                                            q_l1,
                                            k_l1[1, :, :],
                                            acc_s_l0c,
                                            transpose_B=True,
                                            init=True,
                                            kL0Size=gemm1_kL0,
                                            mte2_sync=SIG_KDB1,
                                        )
                                    T.copy(acc_s_l0c, workspace_1[cid, i, :, :])
                                    if (i + 1) % CROSS_INTERVAL == 0 or i == batch_iters - 1:
                                        T.set_cross_flag("FIX", SEM_WS1_C2V)
                                    if i == batch_iters - 1:
                                        # S3: round-head V priming at the G1
                                        # batch tail (FIFO: after every K issue
                                        # of this round; consumed by G2 iter 0).
                                        vj0 = k * NUM_STAGES * block_N
                                        vvalid = T.min(block_N, S_kv - vj0)
                                        T.copy(V[bz, vj0 : vj0 + vvalid, by, :], v_l1)

                                # --- GEMM2 batch: consume w2 (P), produce PV -> w3 ---
                                T.wait_cross_flag(SEM_WS3_V2C)
                                for i in T.serial(batch_iters):
                                    idx = k * NUM_STAGES + i
                                    if i % CROSS_INTERVAL == 0:
                                        T.wait_cross_flag(SEM_WS2_V2C)
                                    # S6: P copy (verbatim position) then Set
                                    # SIG_VP -- FIFO prefix covers V(idx)+P(idx).
                                    T.copy(workspace_2[cid, i, :, :], p_l1)
                                    T.set_flag("MTE2", "MTE1", SIG_VP)
                                    T.gemm_v0(p_l1, v_l1, acc_o_l0c, init=True, mte2_sync=SIG_VP)
                                    # S5: after-return V(idx+1) prefetch (v_l1
                                    # single slot: earliest legal issue point =
                                    # after this call; overlaps the w3 writeout
                                    # + next slot's SEM_WS2 wait).
                                    if i + 1 < batch_iters:
                                        nvidx = idx + 1
                                        nvj0 = nvidx * block_N
                                        nvvalid = T.min(block_N, S_kv - nvj0)
                                        T.copy(V[bz, nvj0 : nvj0 + nvvalid, by, :], v_l1)
                                    T.copy(acc_o_l0c, workspace_3[cid, i, :, :])
                                    if (i + 1) % CROSS_INTERVAL == 0 or i == batch_iters - 1:
                                        T.set_cross_flag("FIX", SEM_WS3_C2V)

                                T.set_cross_flag("MTE2", SEM_WS2_C2V)

                            if t + 1 < my_count:
                                # S4: task-end priming (after the task's last
                                # set WS2_C2V). FIFO mandates Q(t+1) BEFORE
                                # K'_0 so the Set's prefix covers both G1'(0)
                                # operands. bz'/by'/extent re-derived from
                                # task+1 (pure scalar, #R11-1 precedent).
                                ntask = task + 1
                                nbx = ntask % s_blocks
                                nby = ntask // s_blocks % N
                                nbz = ntask // s_blocks // N % B
                                ns0 = nbx * block_M
                                nvalid_m = T.min(block_M, S - ns0)
                                tvalid = T.min(block_N, S_kv)
                                T.copy(Q[nbz, ns0 : ns0 + nvalid_m, nby, :], q_l1)
                                T.copy(K[nbz, 0:tvalid, nby, :], k_l1[0, :, :])
                                T.set_flag("MTE2", "MTE1", SIG_KDB0)
                    else:
                        for t in T.serial(my_count):
                            task = my_start + t
                            bx = task % s_blocks
                            by = task // s_blocks % N
                            bz = task // s_blocks // N % B
                            s0 = bx * block_M
                            valid_m = T.min(block_M, S - s0)
                            eff_kv_iters = (
                                T.floordiv(T.min(s0 + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                            )
                            eff_num_outer = T.ceildiv(eff_kv_iters, NUM_STAGES)

                            # #R4-1 barrier-free: the first gemm_v0 prologue of the task
                            # drains this MTE2 Q load before its MTE1 reads; q_l1 reuse
                            # across tasks is ordered by the previous gemm epilogue
                            # (MTE1 -> MTE2). See S4-Step 2 hazard coverage.
                            T.copy(Q[bz, s0 : s0 + valid_m, by, :], q_l1)

                            for k in T.serial(eff_num_outer):
                                _remaining = eff_kv_iters - k * NUM_STAGES
                                batch_iters = T.if_then_else(_remaining < NUM_STAGES, _remaining, NUM_STAGES)

                                # --- GEMM1 batch: scores -> w1 ---
                                T.wait_cross_flag(SEM_WS1_V2C)
                                for i in T.serial(batch_iters):
                                    idx = k * NUM_STAGES + i
                                    j0 = idx * block_N
                                    valid_n = T.min(block_N, S_kv - j0)
                                    T.copy(K[bz, j0 : j0 + valid_n, by, :], k_l1)
                                    T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True, kL0Size=gemm1_kL0)
                                    T.copy(acc_s_l0c, workspace_1[cid, i, :, :])
                                    if (i + 1) % CROSS_INTERVAL == 0 or i == batch_iters - 1:
                                        T.set_cross_flag("FIX", SEM_WS1_C2V)

                                # --- GEMM2 batch: consume w2 (P), produce PV -> w3 ---
                                T.wait_cross_flag(SEM_WS3_V2C)
                                for i in T.serial(batch_iters):
                                    idx = k * NUM_STAGES + i
                                    j0 = idx * block_N
                                    valid_n = T.min(block_N, S_kv - j0)
                                    T.copy(V[bz, j0 : j0 + valid_n, by, :], v_l1)
                                    if i % CROSS_INTERVAL == 0:
                                        T.wait_cross_flag(SEM_WS2_V2C)
                                    T.copy(workspace_2[cid, i, :, :], p_l1)
                                    T.gemm_v0(p_l1, v_l1, acc_o_l0c, init=True)
                                    T.copy(acc_o_l0c, workspace_3[cid, i, :, :])
                                    if (i + 1) % CROSS_INTERVAL == 0 or i == batch_iters - 1:
                                        T.set_cross_flag("FIX", SEM_WS3_C2V)

                                T.set_cross_flag("MTE2", SEM_WS2_C2V)

                # ==================== Vector core (shared emitter) ====================
                with T.Scope("V"):
                    emit_v(
                        io_buf,
                        io3,
                        work_ub,
                        buf_2d,
                        rt_2d,
                        r_2d,
                        mask_2d,
                        col_pos,
                        row_thresh,
                        rr_ub,
                        acc_s_half,
                        acc_o,
                        out_half,
                        m_i,
                        m_i_prev,
                        sumexp,
                        r_factors,
                        sumexp_is,
                        Output,
                        workspace_1,
                        workspace_2,
                        workspace_3,
                        cid,
                        vid,
                        my_start,
                        my_count,
                        sig_io3,
                    )

        return main


def mha(query, key, value, scaleValue=-1.0, is_causal=False):
    """Host wrapper: metadata validation + JIT dispatch only (zero tensor ops).

    Args:
        query: [B, S, N, D] float16/bfloat16, contiguous, finite values.
        key: [B, S_kv, N, D] same dtype; B/N/D must match query.
        value: [B, S_kv, N, D] same as key.
        scaleValue: <= 0 -> auto 1/sqrt(D); > 0 -> used as-is (JIT constant).
        is_causal: bottom-right aligned causal mask; requires S <= S_kv.

    Returns:
        [B, S, N, D] output, same dtype as query, contiguous (framework-allocated).
    """
    assert query.dtype == key.dtype == value.dtype, "Q/K/V dtypes must match"
    assert query.dtype in (torch.float16, torch.bfloat16), f"unsupported dtype {query.dtype}; only float16/bfloat16"
    assert query.is_contiguous() and key.is_contiguous() and value.is_contiguous(), "Q/K/V must be contiguous BSHD tensors"
    B, S, N, D = query.shape
    S_kv = key.shape[1]
    assert tuple(key.shape) == (B, S_kv, N, D), "key shape must be (B, S_kv, N, D)"
    assert tuple(value.shape) == (B, S_kv, N, D), "value shape must equal key shape"
    assert D in (64, 128, 256), f"D must be in {{64, 128, 256}}, got {D}"
    assert 1 <= B <= 128, f"B out of range [1, 128]: {B}"
    assert 1 <= S <= 2048, f"S out of range [1, 2048]: {S}"
    assert 1 <= S_kv <= 4096, f"S_kv out of range [1, 4096]: {S_kv}"
    assert 1 <= N <= 64, f"N out of range [1, 64]: {N}"
    if is_causal:
        assert S_kv >= S, "is_causal=True requires S <= S_kv (else all-masked rows)"
    kernel = mha_kernel(
        B=B,
        S=S,
        S_kv=S_kv,
        N=N,
        D=D,
        dtype=str(query.dtype).removeprefix("torch."),
        scaleValue=float(scaleValue),
        is_causal=bool(is_causal),
    )
    return kernel(query, key, value)
