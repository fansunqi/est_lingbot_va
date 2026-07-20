#!/bin/bash
# Hang-watchdog for GRPO rollout/eval clients — covers the SILENT hang class:
# SAPIEN ray-trace render / websocket recv / curobo plan freeze where the process
# stays ALIVE and the GPU goes idle, so the supervisor's exit-respawn loop never
# fires and no barrier timeout applies (client.log mtime just stops advancing).
#
# It kills the client whose log has been stale longest once it exceeds STALE_S,
# so the supervisor respawns it (progress is persisted, so it resumes). One kill
# per tick, so we don't mass-kill peers merely blocked waiting on the hung one.
#
# The OTHER hang class — a barrier deadlock (peer crashed, this client spins in
# wait_for_group_barrier / _wait_for_eval_slice_dones) — is NOT caught here (its
# log stays fresh, printing a wait notice every 60s). That one is handled by
# grpo_rollout_client's --group_barrier_timeout (default 900s: raise -> exit ->
# respawn). Run both together.
#
# ⚠️ STALE_S must exceed the worst-case wall time of a single alive-but-slow
# episode, which scales with the clients-per-server ratio: at 1:1 an episode is
# ~90-160s so 900s is very safe; at 4:1 (4 clients serialized on one server GPU)
# a 400-step episode can exceed 15 min and 900s will false-kill. Size STALE_S to
# your ratio (rule of thumb: >= 4x the observed slowest healthy episode).
#
# Env: LOGDIR (client<i>.log dir), CLIENTS ("0 1 2 3"), STALE_S, TICK, PATTERN.
LOGDIR=${LOGDIR:?set LOGDIR to the dir containing client<i>.log}
CLIENTS=${CLIENTS:-"0 1 2 3"}
STALE_S=${STALE_S:-900}
TICK=${TICK:-120}
PATTERN=${PATTERN:-"grpo_rollout_client.*client_id"}
echo "[watchdog start $(date)] LOGDIR=$LOGDIR clients=[$CLIENTS] stale_s=$STALE_S tick=$TICK"
while :; do
  now=$(date +%s)
  worst_c=""; worst_age=0
  for c in $CLIENTS; do
    f=$LOGDIR/client$c.log
    [ -f "$f" ] || continue
    age=$(( now - $(stat -c %Y "$f") ))
    if [ "$age" -gt "$worst_age" ]; then worst_age=$age; worst_c=$c; fi
  done
  if [ -n "$worst_c" ] && [ "$worst_age" -gt "$STALE_S" ]; then
    pid=$(pgrep -f "$PATTERN $worst_c" | head -1)
    if [ -n "$pid" ]; then
      echo "[watchdog $(date)] client$worst_c stale ${worst_age}s > ${STALE_S}s -> kill pid $pid (supervisor respawns)"
      kill -9 "$pid" 2>/dev/null
    fi
  fi
  sleep "$TICK"
done
