#!/usr/bin/bash
# Multi-node NCCL connectivity smoke test. Does NOT start training.
#
# Usage (from the repo root on the login node):
#   # inspect-only: per-node view of bonds, routes, IB HCAs. No NCCL yet.
#   MODE=inspect IP_LIST="ip1:8,ip2:8" bash script/nccl_sanity_multinode.sh
#
#   # real connectivity test: small NCCL all_reduce across nodes.
#   MODE=nccl IP_LIST="ip1:8,ip2:8" VENV_PATH=/home/uv_env/lingbot-va \
#       bash script/nccl_sanity_multinode.sh
#
# Env vars:
#   MODE=inspect|nccl            default inspect. nccl runs torchrun + all_reduce.
#   IP_LIST / NODE_IP_LIST       "ip1:N,ip2:N,...". First IP = master.
#   VENV_PATH                    required for MODE=nccl. Ignored in inspect.
#   MASTER_PORT=29501
#   NCCL_DEBUG=INFO              default for MODE=nccl.
#   GPUS_PER_NODE                overrides the ":N" for MODE=nccl
#                                (e.g. 1 for a fast first check).
#   NIC_OVERRIDE                 overrides the hard-coded bond1.
#   SSH_OPTS                     extra ssh options.

set -euo pipefail

log() { echo "[sanity] $*"; }
die() { log "ERROR: $*"; exit 1; }

MODE="${MODE:-inspect}"
[[ "${MODE}" == "inspect" || "${MODE}" == "nccl" ]] || die "MODE must be 'inspect' or 'nccl'."

if [[ -n "${IP_LIST:-}" ]]; then
    RAW_LIST="${IP_LIST}"; LIST_SRC="IP_LIST"
elif [[ -n "${NODE_IP_LIST:-}" ]]; then
    RAW_LIST="${NODE_IP_LIST}"; LIST_SRC="NODE_IP_LIST"
else
    die "set IP_LIST or NODE_IP_LIST."
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MASTER_PORT="${MASTER_PORT:-29501}"
SSH_OPTS="${SSH_OPTS:--o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs/sanity/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_DIR}"

NIC="${NIC_OVERRIDE:-bond1}"

stripped="$(echo "${RAW_LIST}" | tr -d '[:space:]')"
IFS=',' read -r -a ENTRIES <<< "${stripped}"
(( ${#ENTRIES[@]} > 0 )) || die "no IPs parsed from ${LIST_SRC}='${RAW_LIST}'."

IPS=()
NGPU=""
for entry in "${ENTRIES[@]}"; do
    [[ "${entry}" == *:* ]] || die "entry '${entry}' is missing ':N' GPU-count suffix."
    ip="${entry%:*}"
    n="${entry##*:}"
    [[ "${n}" =~ ^[1-9][0-9]*$ ]] || die "entry '${entry}' has non-numeric GPU count."
    [[ -n "${ip}" ]] || die "entry '${entry}' has empty IP."
    if [[ -z "${NGPU}" ]]; then NGPU="${n}"
    elif [[ "${n}" != "${NGPU}" ]]; then die "mixed GPU counts (${NGPU} vs ${n})."
    fi
    IPS+=("${ip}")
done
NNODES="${#IPS[@]}"
MASTER_ADDR="${IPS[0]}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${NGPU}}"

log "MODE=${MODE}  NNODES=${NNODES}  MASTER=${MASTER_ADDR}:${MASTER_PORT}  NIC=${NIC}"
log "LOG_DIR=${LOG_DIR}"

inspect_cmd() {
    local ip="$1"
    cat <<EOF
set -u
echo "==== host: \$(hostname)  ip-we-reached-you-at: ${ip} ===="
echo "-- bonds --"
for b in /proc/net/bonding/*; do
    [[ -f "\$b" ]] || continue
    echo "\$(basename "\$b"):"
    grep -E 'Bonding Mode|Slave Interface|MII Status' "\$b" | sed 's/^/  /'
done
echo
echo "-- IB HCAs (ibdev2netdev if available) --"
command -v ibdev2netdev >/dev/null && ibdev2netdev || echo "(ibdev2netdev not installed)"
echo "==== end host \$(hostname) ===="
EOF
}

if [[ "${MODE}" == "inspect" ]]; then
    for i in "${!IPS[@]}"; do
        ip="${IPS[$i]}"
        f="${LOG_DIR}/node${i}_${ip}.log"
        log "inspecting node ${i} @ ${ip} -> ${f}"
        # shellcheck disable=SC2086
        ssh ${SSH_OPTS} "${ip}" "$(inspect_cmd "${ip}")" > "${f}" 2>&1 &
    done
    wait
    log "done. Full logs in ${LOG_DIR}/."
    exit 0
fi

# ---------------------------- MODE=nccl ----------------------------
[[ -n "${VENV_PATH:-}" ]] || die "VENV_PATH is required for MODE=nccl."
[[ -f "${SCRIPT_DIR}/_nccl_sanity.py" ]] || die "missing ${SCRIPT_DIR}/_nccl_sanity.py"

NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

# Same NCCL env as run_va_posttrain_multinode.sh.
IFS= read -r -d '' NCCL_ENV <<NCCL_EOF || true
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECKS_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME='${NIC}'
export UCX_NET_DEVICES='${NIC}'
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=0
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=0
export NCCL_TIMEOUT=1800
NCCL_EOF

remote_nccl_cmd() {
    local rank=$1
    cat <<EOF
set -eo pipefail
cd '${REPO_DIR}'
${NCCL_ENV}
export NCCL_DEBUG='${NCCL_DEBUG}'
export NCCL_DEBUG_SUBSYS=INIT,NET
source '${VENV_PATH}/bin/activate'
python -m torch.distributed.run \\
    --nnodes='${NNODES}' --node_rank='${rank}' \\
    --master_addr='${MASTER_ADDR}' --master_port='${MASTER_PORT}' \\
    --nproc_per_node='${GPUS_PER_NODE}' \\
    --tee 3 \\
    '${SCRIPT_DIR}/_nccl_sanity.py'
EOF
}

PIDS=()
_cleanup_ran=0
cleanup() {
    (( _cleanup_ran )) && return 0
    _cleanup_ran=1
    local pat="master_port[ =]${MASTER_PORT}[^0-9]"
    local ip_csv; ip_csv="$(IFS=,; echo "${IPS[*]}")"
    log "stopping sanity run on ${NNODES} nodes..."
    if command -v pdsh >/dev/null 2>&1; then
        PDSH_RCMD_TYPE=ssh pdsh -f 100 -w "${ip_csv}" "pkill -TERM -f -- '${pat}'" >/dev/null 2>&1 || true
        sleep 3
        PDSH_RCMD_TYPE=ssh pdsh -f 100 -w "${ip_csv}" "pkill -KILL -f -- '${pat}'" >/dev/null 2>&1 || true
    else
        for ip in "${IPS[@]}"; do
            ssh ${SSH_OPTS} "${ip}" "pkill -TERM -f -- '${pat}' || true; sleep 3; pkill -KILL -f -- '${pat}' || true" &
        done
        wait || true
    fi
}
trap cleanup INT TERM

# Pre-create log files so `tail -F` below doesn't race the first ssh fork.
for i in "${!IPS[@]}"; do
    : > "${LOG_DIR}/node${i}_${IPS[$i]}.log"
done

log "NCCL mode: GPUS_PER_NODE=${GPUS_PER_NODE}  WORLD=$(( NNODES * GPUS_PER_NODE ))"
for i in "${!IPS[@]}"; do
    ip="${IPS[$i]}"
    f="${LOG_DIR}/node${i}_${ip}.log"
    log "launching sanity on node ${i} @ ${ip} -> ${f}"
    # shellcheck disable=SC2086
    ssh ${SSH_OPTS} "${ip}" "$(remote_nccl_cmd "${i}")" > "${f}" 2>&1 &
    PIDS+=($!)
done

log "tailing node0 (Ctrl-C stops everything)."
tail -F "${LOG_DIR}/node0_${IPS[0]}.log" &
tail_pid=$!

fail=0
for pid in "${PIDS[@]}"; do wait "${pid}" || fail=1; done
kill "${tail_pid}" 2>/dev/null || true

log "per-node result:"
for i in "${!IPS[@]}"; do
    ip="${IPS[$i]}"
    f="${LOG_DIR}/node${i}_${ip}.log"
    ok_line="$(grep -m1 '] OK '   "${f}" || true)"
    fail_line="$(grep -m1 '] FAIL' "${f}" || true)"
    if [[ -n "${ok_line}" && -z "${fail_line}" ]]; then
        echo "  node${i} ${ip}  OK"
    else
        echo "  node${i} ${ip}  FAIL  (see ${f})"
        [[ -n "${fail_line}" ]] && echo "    ${fail_line}"
    fi
done

(( fail )) && die "at least one node failed. Full logs in ${LOG_DIR}/."
log "all ${NNODES} nodes passed the sanity check."
