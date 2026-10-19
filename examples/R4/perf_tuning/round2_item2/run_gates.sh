#!/bin/bash
# Four mandatory gates on the UNMODIFIED delivered kernel (item #2 produced no candidate).
# cwd = examples/R4 (outside the repo root, per the env recipe). Env exported by the caller.
cd /mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4 || exit 9
D=perf_tuning/round2_item2
sha256sum r4_fused_sampling.py test_r4_fused_sampling.py \
          cann_bench_submission/cann_bench/_r4_kernels.py > "$D/gate_sha256_before.txt"

python test_r4_fused_sampling.py --level all        > "$D/gate_all.log"       2>&1; echo "gate_all=$?"       >> "$D/gate_exits.txt"
python test_r4_fused_sampling.py --level stability  > "$D/gate_stability.log" 2>&1; echo "gate_stability=$?" >> "$D/gate_exits.txt"
python test_r4_fused_sampling.py --level stress     > "$D/gate_stress.log"    2>&1; echo "gate_stress=$?"    >> "$D/gate_exits.txt"
python ../../.agents/skills/tilelang-op-test-design/scripts/coverage_check.py \
       test_r4_fused_sampling.py --proto proto.yaml > "$D/gate_coverage.log"  2>&1; echo "gate_coverage=$?"  >> "$D/gate_exits.txt"

sha256sum r4_fused_sampling.py test_r4_fused_sampling.py \
          cann_bench_submission/cann_bench/_r4_kernels.py > "$D/gate_sha256_after.txt"
echo "GATES_DONE" >> "$D/gate_exits.txt"
