#!/usr/bin/bash
#
# FSDP config sweep on 8×H200 (single node).
#
# Phase A: 6 cells over (ac_granularity, reshard_after_forward) at block_only=true.
# Phase B: 2 cells with the Phase A winner repeated at block_only=false.
#
# Each cell runs configs/tasks/smoke_fsdp.yaml for max_steps=10. The
# FSDPMetricsCallback writes one row to ./experiments/smoke/sweep_results.csv.
#
# Usage:
#   bash scripts/sweep_fsdp.sh                # full sweep (Phase A only by default)
#   PHASE=both bash scripts/sweep_fsdp.sh     # Phase A then Phase B
#   DEVICES=1 bash scripts/sweep_fsdp.sh      # 1-GPU sanity run
#
# Results are in ./experiments/smoke/sweep_results.csv. The script prints a
# sorted table at the end.

# Don't `set -e`: a single OOM in one cell shouldn't kill the rest of the
# sweep. We log the failure and move on.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK="${TASK:-configs/tasks/smoke_fsdp.yaml}"
DEVICES="${DEVICES:-8}"
PHASE="${PHASE:-A}"

RESULTS_CSV="./experiments/smoke/sweep_results.csv"
rm -f "${RESULTS_CSV}"

run_cell() {
    local tag="$1" ac="$2" reshard="$3" block_only="$4" compile="$5"
    echo
    echo "================================================================"
    echo "[sweep] cell tag=${tag} ac=${ac} reshard=${reshard} block_only=${block_only} compile=${compile} devices=${DEVICES}"
    echo "================================================================"
    if VENV_PATH="${VENV_PATH:-/home/zjp/anaconda3/envs/fa4}" \
       TASK="${TASK}" \
       bash "${SCRIPT_DIR}/run_va_posttrain.sh" \
            --devices "${DEVICES}" \
            --fsdp-ac "${ac}" \
            --fsdp-reshard "${reshard}" \
            --fsdp-block-only "${block_only}" \
            --compile-model "${compile}" \
            --fsdp-metrics-tag "${tag}" \
            --wandb-mode disabled
    then
        echo "[sweep] cell ${tag} OK"
    else
        rc=$?
        echo "[sweep] cell ${tag} FAILED (exit=${rc}); continuing sweep"
    fi
    # Give NCCL a beat to clean up before the next launch.
    sleep 5
}

# Sweep matrix v2: previous sweep picked reshard=False + block_only=False as
# the FSDP-side winner (~2-4% step-time gain). Holding those fixed, this round
# tests the two real speed levers — AC granularity and torch.compile.
# 4 cells: ac ∈ {all, every_2} × compile ∈ {false, true}.
# (every_4 / none would OOM at max_tokens=64000 — see plan doc for math.)
if [[ "${PHASE}" == "A" || "${PHASE}" == "both" ]]; then
    for ac in all every_2; do
        for compile in false true; do
            run_cell "ac-${ac}_compile-${compile}" \
                "${ac}" "false" "false" "${compile}"
        done
    done
fi

echo
echo "================================================================"
echo "[sweep] results — sorted by step_ms_p50 ascending"
echo "================================================================"
python -c "
import csv, sys
rows = list(csv.DictReader(open('${RESULTS_CSV}')))
rows.sort(key=lambda r: float(r['step_ms_p50']) if r['step_ms_p50'] != 'nan' else 1e9)
hdr = ['tag','ac','reshard','block_only','step_ms_p50','step_ms_p10','peak_alloc_gb','peak_reserved_gb']
widths = {h: max(len(h), max((len(r[h]) for r in rows), default=0)) for h in hdr}
print('  '.join(h.ljust(widths[h]) for h in hdr))
for r in rows:
    print('  '.join(r[h].ljust(widths[h]) for h in hdr))
"
