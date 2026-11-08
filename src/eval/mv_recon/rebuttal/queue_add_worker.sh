#!/bin/bash
# Add a GPU worker to a queue that run_queue.sh is already running: it shares that queue's job counter, lock and
# status file (nothing is reset), so it takes the next pending jobs alongside the existing workers.
#
# Usage: QUEUE_DIR=<queue dir> GPU=4 PORT=29710 bash queue_add_worker.sh jobs.txt
#   jobs.txt must be the same file the running queue was started with.
set -u
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
jobs_file="$(realpath "$1")"
queue_dir="$(realpath "${QUEUE_DIR:?set QUEUE_DIR to the running queue directory}")"
gpu="${GPU:?set GPU}"
port="${PORT:-29710}"
counter="$queue_dir/next_job"
lock="$queue_dir/lock"
[ -f "$counter" ] || { echo "no running queue in $queue_dir" >&2; exit 1; }
mapfile -t lines < <(grep -v '^\s*#' "$jobs_file" | grep -v '^\s*$')
echo "$(date '+%F %T') gpu${gpu} joined the queue (${#lines[@]} jobs)"
while true; do
    idx=$(flock "$lock" bash -c "i=\$(cat '$counter'); echo \$((i+1)) > '$counter'; echo \$i")
    [ "$idx" -ge "${#lines[@]}" ] && break
    line="${lines[$idx]}"
    name="${line%%$'\t'*}"
    cmd="${line#*$'\t'}"
    for attempt in 1 2; do
        echo "$(date '+%F %T') gpu${gpu} start ${name} (attempt ${attempt})"
        (cd "$script_dir" && CUDA_VISIBLE_DEVICES="$gpu" MAIN_PROCESS_PORT="$port" bash -c "$cmd") \
            > "$queue_dir/${name}.log" 2>&1
        rc=$?
        [ "$rc" -eq 0 ] && break
        sleep 30
    done
    echo "$(date '+%F %T') gpu${gpu} done ${name} rc=${rc}"
    flock "$lock" bash -c "printf '%s\t%s\t%s\n' '$name' 'gpu$gpu' '$rc' >> '$queue_dir/status.tsv'"
done
echo "$(date '+%F %T') gpu${gpu} worker finished"
