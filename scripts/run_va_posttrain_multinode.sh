#!/usr/bin/bash
# Multi-node launcher for script/run_va_posttrain.sh.
#
# Fans out via ssh to every IP in the node list, assigns NODE_RANK by
# position, and uses the first IP as MASTER_ADDR.
#
# Usage (on the login node, from the repo root):
#   # A) use the cluster-injected full node list
#   TASK=configs/tasks/train_test.yaml bash scripts/run_va_posttrain_multinode.sh
#   # B) run on a hand-picked subset (one-shot override)
#   IP_LIST="ip1:8,ip2:8" TASK=configs/tasks/train_test.yaml \
#       bash script/run_va_posttrain_multinode.sh
#
# Env vars:
#   IP_LIST                      preferred override of the node list. Same
#                                format as NODE_IP_LIST but not clobbered by
#                                the cluster's shell-init hooks.
#   NODE_IP_LIST                 fallback. "ip1:N,ip2:N,..." — ":N" is the
#                                per-node GPU count, required on every entry
#                                and identical across entries. Whitespace
#                                is stripped.
#   MASTER_PORT=29501            rendezvous port.
#   TASK=configs/tasks/train_robotwin.yaml   path to a training task YAML.
#   SAVE_ROOT                    optional. Overrides config.save_root via
#                                train.py's --save-root. Prefer this over
#                                editing shared_config.py for parallel jobs.
#   WANDB_NAME                   optional. W&B run name (e.g. "wan-base").
#   VENV_PATH                    required. venv path visible on every node
#                                (shared mount that all cluster nodes see).
#   NIC_OVERRIDE                 optional. Overrides bond1 for
#                                NCCL_SOCKET_IFNAME / UCX_NET_DEVICES.
#   SSH_OPTS                     extra ssh options.
#   LOG_DIR                      per-node log directory
#                                (default: ${REPO_DIR}/logs/multinode/<ts>/).

set -euo pipefail

log() { echo "[multinode] $*"; }
die() { log "ERROR: $*"; exit 1; }

# IP_LIST wins: the cluster's shell-init hooks silently re-export NODE_IP_LIST
# in subshells, clobbering command-line overrides.
if [[ -n "${IP_LIST:-}" ]]; then
    RAW_LIST="${IP_LIST}"; LIST_SRC="IP_LIST"
elif [[ -n "${NODE_IP_LIST:-}" ]]; then
    RAW_LIST="${NODE_IP_LIST}"; LIST_SRC="NODE_IP_LIST"
else
    die "set IP_LIST (preferred) or NODE_IP_LIST."
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MASTER_PORT="${MASTER_PORT:-29501}"
TASK="${TASK:-configs/tasks/train_robotwin.yaml}"
[[ -n "${VENV_PATH:-}" ]] || die "VENV_PATH is required (shared-disk venv path; see README)."
SSH_OPTS="${SSH_OPTS:--o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs/multinode/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_DIR}"

# Parse "ip1:N, ip2:N, ..." into IPS=(ip1 ip2 ...) and derive NGPU from the
# ":N" suffix. Every entry must carry ":N" and share the same N.
stripped="$(echo "${RAW_LIST}" | tr -d '[:space:]')"
IFS=',' read -r -a ENTRIES <<< "${stripped}"
(( ${#ENTRIES[@]} > 0 )) || die "no IPs parsed from ${LIST_SRC}='${RAW_LIST}'."

IPS=()
NGPU=""
for entry in "${ENTRIES[@]}"; do
    [[ "${entry}" == *:* ]] || die "entry '${entry}' is missing ':N' GPU-count suffix."
    ip="${entry%:*}"
    n="${entry##*:}"
    [[ "${n}" =~ ^[1-9][0-9]*$ ]] || die "entry '${entry}' has non-numeric GPU count ':${n}'."
    [[ -n "${ip}" ]] || die "entry '${entry}' has empty IP."
    if [[ -z "${NGPU}" ]]; then
        NGPU="${n}"
    elif [[ "${n}" != "${NGPU}" ]]; then
        die "mixed GPU counts in ${LIST_SRC} (${NGPU} vs ${n}); all entries must match."
    fi
    IPS+=("${ip}")
done
NNODES="${#IPS[@]}"
MASTER_ADDR="${IPS[0]}"

log "node list from ${LIST_SRC}: ${RAW_LIST}"
log "NNODES=${NNODES}  NGPU=${NGPU}  WORLD_SIZE=$(( NNODES * NGPU ))"
log "MASTER=${MASTER_ADDR}:${MASTER_PORT}  TASK=${TASK}  REPO=${REPO_DIR}"
log "VENV=${VENV_PATH}"
log "SAVE_ROOT=${SAVE_ROOT:-<default>}"
log "WANDB_NAME=${WANDB_NAME:-<default>}"
log "LOG_DIR=${LOG_DIR}"

# NCCL/UCX tuning from the cluster maintainer. Evaluated on each remote node
# (ssh drops non-whitelisted env vars). bond1 is what every node here uses —
# verified via script/nccl_sanity_multinode.sh; override with NIC_OVERRIDE.
IFS= read -r -d '' NCCL_ENV <<NCCL_EOF || true
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECKS_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME='${NIC_OVERRIDE:-bond1}'
export UCX_NET_DEVICES='${NIC_OVERRIDE:-bond1}'
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=0
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=0
export NCCL_TIMEOUT=1800
NCCL_EOF

remote_cmd() {
    local rank=$1
    cat <<EOF
set -eo pipefail
cd '${REPO_DIR}'
${NCCL_ENV}
export VENV_PATH='${VENV_PATH}'
export NGPU='${NGPU}' TASK='${TASK}'
export NNODES='${NNODES}' NODE_RANK='${rank}'
export MASTER_ADDR='${MASTER_ADDR}' MASTER_PORT='${MASTER_PORT}'
${SAVE_ROOT:+export SAVE_ROOT='${SAVE_ROOT}'}
${WANDB_NAME:+export WANDB_NAME='${WANDB_NAME}'}
bash scripts/run_va_posttrain.sh
EOF
}

PIDS=()

_cleanup_ran=0
cleanup() {
    (( _cleanup_ran )) && return 0
    _cleanup_ran=1

    # Scope via MASTER_PORT so parallel launchers on overlapping nodes don't
    # take each other out. SIGTERM first lets torchrun tear down workers
    # cleanly — bare SIGKILL orphans them and leaks NCCL state.
    local pat="master_port[ =]${MASTER_PORT}[^0-9]"
    local ip_csv; ip_csv="$(IFS=,; echo "${IPS[*]}")"
    log "stopping job (master_port=${MASTER_PORT}) on ${NNODES} nodes..."
    PDSH_RCMD_TYPE=ssh pdsh -f 100 -w "${ip_csv}" "pkill -TERM -f -- '${pat}'" >/dev/null 2>&1 || true
    sleep 5
    PDSH_RCMD_TYPE=ssh pdsh -f 100 -w "${ip_csv}" "pkill -KILL -f -- '${pat}'" >/dev/null 2>&1 || true
}
trap cleanup INT TERM

# Pre-create log files so `tail -F` doesn't race the first ssh fork.
for i in "${!IPS[@]}"; do
    : > "${LOG_DIR}/node${i}_${IPS[$i]}.log"
done

for i in "${!IPS[@]}"; do
    ip="${IPS[$i]}"
    log_file="${LOG_DIR}/node${i}_${ip}.log"
    log "launching node ${i} @ ${ip} -> ${log_file}"
    # shellcheck disable=SC2086
    ssh ${SSH_OPTS} "${ip}" "$(remote_cmd "${i}")" > "${log_file}" 2>&1 &
    PIDS+=($!)
done

log "all ${NNODES} nodes launched. Tailing node0 (Ctrl-C stops everything)."
tail -F "${LOG_DIR}/node0_${IPS[0]}.log" &
tail_pid=$!

fail=0
for pid in "${PIDS[@]}"; do
    wait "${pid}" || fail=1
done
kill "${tail_pid}" 2>/dev/null || true

(( fail )) && die "at least one node exited non-zero. See ${LOG_DIR}/."
log "all nodes finished successfully."
