#!/bin/bash
# Run a list of rebuttal jobs over several GPUs: one job per GPU at a time, each GPU takes the next job when it
# finishes. A failed job is retried once (GPUs here are shared, so a run can die from someone else's OOM).
#
# Usage: GPUS="0 1 2 3" bash run_queue.sh jobs.txt
#   jobs.txt: one job per line, "name<TAB>command" (command runs from this directory with bash -c);
#   blank lines and lines starting with # are ignored.
# Logs: $QUEUE_DIR/<name>.log (default ../../../../eval_results/mv_recon/rebuttal/queue), status in status.tsv.
set -u
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
jobs_file="$(realpath "$1")"
gpus=(${GPUS:-0})
queue_dir="${QUEUE_DIR:-${script_dir}/../../../../eval_results/mv_recon/rebuttal/queue}"
mkdir -p "$queue_dir"
queue_dir="$(realpath "$queue_dir")"
counter="$queue_dir/next_job"
lock="$queue_dir/lock"
echo 0 > "$counter"
: > "$queue_dir/status.tsv"
mapfile -t lines < <(grep -v '^\s*#' "$jobs_file" | grep -v '^\s*$')
echo "queue: ${#lines[@]} jobs on GPUs ${gpus[*]} (logs in $queue_dir)"

worker() {
    local gpu="$1" port="$2" idx line name cmd attempt rc
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
}

i=0
for gpu in "${gpus[@]}"; do
    worker "$gpu" $((29700 + i)) &
    i=$((i + 1))
done
wait
echo "$(date '+%F %T') queue finished"
column -t "$queue_dir/status.tsv"
