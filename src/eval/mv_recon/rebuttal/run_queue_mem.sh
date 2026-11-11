#!/bin/bash
# Memory-aware job queue: up to SLOTS jobs per GPU, a job starts on a GPU only if
#   used memory (nvidia-smi) + memory reserved by jobs this queue started there in the last RESERVE_SECONDS
#   + the job's own need  <=  CAP_MIB.
# A job's need comes from NEED_RULES (first matching name regex wins, else DEFAULT_NEED_MIB). Jobs are taken in file
# order, skipping ones that do not fit on that GPU right now.
# One dispatcher lock covers the whole decision, so two slots never start on the same free memory.
#
# Usage: GPUS="0 2 3 4" bash run_queue_mem.sh jobs.txt
#   jobs.txt: "name<TAB>command" per line (command runs from this directory), # comments and blank lines ignored.
# Logs: $QUEUE_DIR/<name>.log, status in $QUEUE_DIR/status.tsv (name, gpu, rc).
# Trajectory jobs whose log reports "OOM error in sequence" are marked rc=97 (pose evaluation skips such sequences
# and exits 0).
set -u
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
jobs_file="$(realpath "$1")"
gpus=(${GPUS:-0})
slots=${SLOTS:-2}
cap=${CAP_MIB:-22500}
default_need=${DEFAULT_NEED_MIB:-10000}
reserve_seconds=${RESERVE_SECONDS:-180}
need_rules="${NEED_RULES:-}"   # "regex=mib;regex=mib"
queue_dir="$(realpath -m "${QUEUE_DIR:?set QUEUE_DIR}")"
mkdir -p "$queue_dir"
lock="$queue_dir/lock"
taken="$queue_dir/taken"   # indices of started jobs
reserv="$queue_dir/reservations"   # lines: gpu start_epoch need
: > "$taken"
: > "$reserv"
: > "$queue_dir/status.tsv"
mapfile -t lines < <(grep -v '^\s*#' "$jobs_file" | grep -v '^\s*$')
echo "queue: ${#lines[@]} jobs on GPUs ${gpus[*]} (${slots} slots each, cap ${cap} MiB; logs in $queue_dir)"

job_need() {
    local name="$1" rule re mib
    IFS=';' read -ra rules <<< "$need_rules"
    for rule in "${rules[@]}"; do
        re="${rule%=*}"; mib="${rule##*=}"
        [ -n "$re" ] && [[ "$name" =~ $re ]] && { echo "$mib"; return; }
    done
    echo "$default_need"
}

# Under the lock: take the first pending job (in file order) that fits on GPU $1, reserve its memory and print its
# index; print "done" when nothing is pending, nothing when no pending job fits right now.
try_take() {
    local gpu="$1" idx line name need used now reserved pending=0
    used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    now=$(date +%s)
    reserved=$(awk -v g="$gpu" -v now="$now" -v w="$reserve_seconds" '$1==g && now-$2<w {s+=$3} END {print s+0}' "$reserv")
    for ((idx = 0; idx < ${#lines[@]}; idx++)); do
        grep -qx "$idx" "$taken" && continue
        pending=1
        line="${lines[$idx]}"; name="${line%%$'\t'*}"
        need=$(job_need "$name")
        if [ $((used + reserved + need)) -le "$cap" ]; then
            echo "$idx" >> "$taken"
            echo "$gpu $now $need" >> "$reserv"
            echo "$idx"
            return
        fi
    done
    [ "$pending" -eq 0 ] && echo done
}

worker() {
    local gpu="$1" port="$2" idx line name cmd rc attempt
    while true; do
        idx=$(flock "$lock" bash -c "$(declare -p lines taken reserv cap reserve_seconds default_need need_rules); $(declare -f try_take job_need); try_take $gpu")
        if [ "$idx" = done ]; then break; fi
        if [ -z "$idx" ]; then sleep "${WAIT_SECONDS:-30}"; continue; fi
        line="${lines[$idx]}"; name="${line%%$'\t'*}"; cmd="${line#*$'\t'}"
        for attempt in 1 2; do
            echo "$(date '+%F %T') gpu${gpu} start ${name} (attempt ${attempt})"
            (cd "$script_dir" && CUDA_VISIBLE_DEVICES="$gpu" MAIN_PROCESS_PORT="$port" bash -c "$cmd") \
                > "$queue_dir/${name}.log" 2>&1
            rc=$?
            if [ "$rc" -eq 0 ] && grep -q "OOM error in sequence" "$queue_dir/${name}.log"; then rc=97; fi
            [ "$rc" -eq 0 ] && break
            sleep 60
        done
        echo "$(date '+%F %T') gpu${gpu} done ${name} rc=${rc}"
        printf '%s\t%s\t%s\n' "$name" "gpu${gpu}" "$rc" >> "$queue_dir/status.tsv"
    done
}

port=29600
for gpu in "${gpus[@]}"; do
    for ((s = 0; s < slots; s++)); do
        worker "$gpu" "$port" &
        port=$((port + 1))
        sleep 2
    done
done
wait
echo "$(date '+%F %T') queue finished"
