#!/usr/bin/env bash
#
# Show Ascend NPU users with process, cwd, start time, and Docker container info.
#
# Usage:
#   ./look.sh                 # show all NPU processes, plus idle NPU cards
#   ./look.sh 3040460         # show details for one PID
#   ./look.sh 3040460 12345   # show details for several PIDs
#   ./look.sh --tsv           # print tab-separated output
#   ./look.sh --wide          # do not truncate long cwd/cmd/image fields
#   ./look.sh --watch 5       # refresh every 5 seconds
#
# Tip: run as root if cwd/cgroup/container info is hidden by /proc permissions.

set -o pipefail

FORMAT="table"
WIDE=0
NO_DOCKER=0
WATCH_INTERVAL=""
SHOW_RAW=0
PIDS=()

usage() {
    sed -n '1,22p' "$0" | sed 's/^# \{0,1\}//'
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

trim() {
    local value="${1:-}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

shorten() {
    local value="${1:-}"
    local max="${2:-}"
    if [ "$WIDE" -eq 1 ] || [ -z "$max" ] || [ "${#value}" -le "$max" ]; then
        printf '%s' "$value"
        return
    fi
    if [ "$max" -le 1 ]; then
        printf '%.*s' "$max" "$value"
        return
    fi
    printf '%s...' "${value:0:$((max - 3))}"
}

parse_args() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            -h|--help)
                usage
                exit 0
                ;;
            --tsv)
                FORMAT="tsv"
                shift
                ;;
            --wide)
                WIDE=1
                shift
                ;;
            --no-docker)
                NO_DOCKER=1
                shift
                ;;
            --raw)
                SHOW_RAW=1
                shift
                ;;
            -w|--watch)
                [ "$#" -ge 2 ] || die "--watch needs seconds"
                WATCH_INTERVAL="$2"
                shift 2
                ;;
            --)
                shift
                while [ "$#" -gt 0 ]; do
                    PIDS+=("$1")
                    shift
                done
                ;;
            -*)
                die "unknown option: $1"
                ;;
            *)
                PIDS+=("$1")
                shift
                ;;
        esac
    done
}

read_npu_smi() {
    local out_file="$1"
    if ! command -v npu-smi >/dev/null 2>&1; then
        return 127
    fi
    npu-smi info >"$out_file" 2>/dev/null
}

parse_npu_smi_file() {
    local file="$1"
    awk '
        BEGIN { in_proc = 0 }

        /Process id/ {
            in_proc = 1
            next
        }

        in_proc && /^\+/ { next }

        in_proc && /^\|/ {
            line = $0
            gsub(/\|/, " ", line)
            gsub(/^[ \t]+|[ \t]+$/, "", line)
            n = split(line, a, /[ \t]+/)
            if (n >= 4 && a[1] ~ /^[0-9]+$/ && a[3] ~ /^[0-9]+$/) {
                pname = a[4]
                for (i = 5; i < n; i++) {
                    pname = pname " " a[i]
                }
                mem = a[n]
                print "PROC\t" a[1] "\t" a[2] "\t" a[3] "\t" pname "\t" mem
            }
            next
        }

        !in_proc && /^\|[[:space:]]*[0-9]+[[:space:]]/ {
            line = $0
            gsub(/\|/, " ", line)
            gsub(/^[ \t]+|[ \t]+$/, "", line)
            n = split(line, a, /[ \t]+/)
            if (n >= 1 && a[1] ~ /^[0-9]+$/) {
                print "CARD\t" a[1]
            }
            next
        }
    ' "$file"
}

pid_exists() {
    [ -d "/proc/$1" ]
}

proc_user() {
    local pid="$1"
    ps -o user= -p "$pid" 2>/dev/null | awk '{$1=$1; print}'
}

proc_start() {
    local pid="$1"
    ps -o lstart= -p "$pid" 2>/dev/null | awk '{$1=$1; print}'
}

proc_age() {
    local pid="$1"
    ps -o etime= -p "$pid" 2>/dev/null | awk '{$1=$1; print}'
}

proc_cmd() {
    local pid="$1"
    local cmd=""
    if [ -r "/proc/$pid/cmdline" ]; then
        cmd="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null)"
        cmd="$(trim "$cmd")"
    fi
    if [ -z "$cmd" ]; then
        cmd="$(ps -o args= -p "$pid" 2>/dev/null | awk '{$1=$1; print}')"
    fi
    printf '%s' "${cmd:-?}"
}

proc_cwd() {
    local pid="$1"
    local cwd=""
    cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null)"
    printf '%s' "${cwd:-?}"
}

container_id_from_cgroup() {
    local pid="$1"
    local cgroup="/proc/$pid/cgroup"
    local id=""
    [ -r "$cgroup" ] || return 0

    id="$(grep -aoE '[0-9a-f]{64}' "$cgroup" 2>/dev/null | head -n 1)"
    if [ -z "$id" ]; then
        id="$(grep -aoE '[0-9a-f]{32,63}' "$cgroup" 2>/dev/null | head -n 1)"
    fi
    printf '%s' "$id"
}

docker_info_by_id() {
    local container_id="$1"
    local info=""

    if [ -z "$container_id" ]; then
        printf '%s\t%s\t%s\t%s' "-" "-" "-" "-"
        return
    fi

    if [ "$NO_DOCKER" -eq 1 ] || ! command -v docker >/dev/null 2>&1; then
        printf '%s\t%s\t%s\t%s' "${container_id:0:12}" "-" "-" "-"
        return
    fi

    info="$(docker ps -a --no-trunc --filter "id=$container_id" \
        --format '{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}' 2>/dev/null | head -n 1)"

    if [ -z "$info" ]; then
        printf '%s\t%s\t%s\t%s' "${container_id:0:12}" "?" "?" "?"
        return
    fi

    local full_id name image status
    IFS=$'\t' read -r full_id name image status <<<"$info"
    printf '%s\t%s\t%s\t%s' "${full_id:0:12}" "${name:-?}" "${image:-?}" "${status:-?}"
}

print_header() {
    if [ "$FORMAT" = "tsv" ]; then
        printf 'NPU\tCHIP\tPID\tUSER\tSTART\tAGE\tNPU_MEM_MB\tPROC\tCWD\tCONTAINER_ID\tCONTAINER_NAME\tIMAGE\tSTATUS\tCMD\n'
        return
    fi

    printf '%-4s %-4s %-10s %-10s %-24s %-12s %-10s %-18s %-34s %-12s %-18s %-24s %-18s %s\n' \
        "NPU" "CHIP" "PID" "USER" "START" "AGE" "NPU_MEM" "PROC" "CWD" \
        "CONTAINER" "NAME" "IMAGE" "STATUS" "CMD"
}

print_row() {
    local npu="$1"
    local chip="$2"
    local pid="$3"
    local user="$4"
    local start="$5"
    local age="$6"
    local npu_mem="$7"
    local proc_name="$8"
    local cwd="$9"
    local container_id="${10}"
    local container_name="${11}"
    local image="${12}"
    local status="${13}"
    local cmd="${14}"

    if [ "$FORMAT" = "tsv" ]; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$npu" "$chip" "$pid" "$user" "$start" "$age" "$npu_mem" "$proc_name" "$cwd" \
            "$container_id" "$container_name" "$image" "$status" "$cmd"
        return
    fi

    printf '%-4s %-4s %-10s %-10s %-24s %-12s %-10s %-18s %-34s %-12s %-18s %-24s %-18s %s\n' \
        "$(shorten "$npu" 4)" \
        "$(shorten "$chip" 4)" \
        "$(shorten "$pid" 10)" \
        "$(shorten "$user" 10)" \
        "$(shorten "$start" 24)" \
        "$(shorten "$age" 12)" \
        "$(shorten "$npu_mem" 10)" \
        "$(shorten "$proc_name" 18)" \
        "$(shorten "$cwd" 34)" \
        "$(shorten "$container_id" 12)" \
        "$(shorten "$container_name" 18)" \
        "$(shorten "$image" 24)" \
        "$(shorten "$status" 18)" \
        "$(shorten "$cmd" 80)"
}

print_process_row() {
    local npu="$1"
    local chip="$2"
    local pid="$3"
    local proc_name="$4"
    local npu_mem="$5"

    if ! pid_exists "$pid"; then
        print_row "$npu" "$chip" "$pid" "exited" "-" "-" "$npu_mem" "$proc_name" "-" "-" "-" "-" "-" "-"
        return
    fi

    local user start age cwd cmd container_id docker_fields container_short container_name image status
    user="$(proc_user "$pid")"
    start="$(proc_start "$pid")"
    age="$(proc_age "$pid")"
    cwd="$(proc_cwd "$pid")"
    cmd="$(proc_cmd "$pid")"
    container_id="$(container_id_from_cgroup "$pid")"
    docker_fields="$(docker_info_by_id "$container_id")"
    IFS=$'\t' read -r container_short container_name image status <<<"$docker_fields"

    print_row \
        "$npu" \
        "$chip" \
        "$pid" \
        "${user:-?}" \
        "${start:-?}" \
        "${age:-?}" \
        "${npu_mem:-?}" \
        "${proc_name:-?}" \
        "$cwd" \
        "${container_short:-?}" \
        "${container_name:-?}" \
        "${image:-?}" \
        "${status:-?}" \
        "$cmd"
}

main_once() {
    local tmp_file
    tmp_file="$(mktemp)"

    local npu_smi_rc=0
    read_npu_smi "$tmp_file" || npu_smi_rc=$?

    if [ "$SHOW_RAW" -eq 1 ]; then
        if [ "$npu_smi_rc" -eq 0 ]; then
            cat "$tmp_file"
        else
            echo "npu-smi info is unavailable"
        fi
        rm -f "$tmp_file"
        return
    fi

    local parsed_lines=()
    if [ "$npu_smi_rc" -eq 0 ]; then
        mapfile -t parsed_lines < <(parse_npu_smi_file "$tmp_file")
    elif [ "${#PIDS[@]}" -eq 0 ]; then
        rm -f "$tmp_file"
        die "npu-smi info failed or npu-smi is not in PATH"
    fi
    rm -f "$tmp_file"

    local cards=()
    local procs=()
    local line kind rest
    declare -A card_seen=()
    declare -A card_has_proc=()

    for line in "${parsed_lines[@]}"; do
        IFS=$'\t' read -r kind rest <<<"$line"
        if [ "$kind" = "CARD" ]; then
            if [ -z "${card_seen[$rest]+x}" ]; then
                card_seen["$rest"]=1
                cards+=("$rest")
            fi
        elif [ "$kind" = "PROC" ]; then
            procs+=("$line")
            local npu chip pid pname mem
            IFS=$'\t' read -r _ npu chip pid pname mem <<<"$line"
            card_has_proc["$npu"]=1
        fi
    done

    print_header

    if [ "${#PIDS[@]}" -gt 0 ]; then
        declare -A want=()
        declare -A found=()
        local pid_arg
        for pid_arg in "${PIDS[@]}"; do
            want["$pid_arg"]=1
        done

        for line in "${procs[@]}"; do
            local npu chip pid pname mem
            IFS=$'\t' read -r _ npu chip pid pname mem <<<"$line"
            if [ -n "${want[$pid]+x}" ]; then
                found["$pid"]=1
                print_process_row "$npu" "$chip" "$pid" "$pname" "$mem"
            fi
        done

        for pid_arg in "${PIDS[@]}"; do
            if [ -z "${found[$pid_arg]+x}" ]; then
                local fallback_name
                fallback_name="$(ps -o comm= -p "$pid_arg" 2>/dev/null | awk '{$1=$1; print}')"
                print_process_row "-" "-" "$pid_arg" "${fallback_name:-?}" "-"
            fi
        done
        return
    fi

    for line in "${procs[@]}"; do
        local npu chip pid pname mem
        IFS=$'\t' read -r _ npu chip pid pname mem <<<"$line"
        print_process_row "$npu" "$chip" "$pid" "$pname" "$mem"
    done

    local card
    for card in $(printf '%s\n' "${cards[@]}" | sort -n); do
        if [ -z "${card_has_proc[$card]+x}" ]; then
            print_row "$card" "-" "-" "-" "-" "-" "-" "idle" "-" "-" "-" "-" "-" "-"
        fi
    done
}

parse_args "$@"

if [ -n "$WATCH_INTERVAL" ]; then
    while true; do
        clear
        date '+%F %T %Z'
        main_once
        sleep "$WATCH_INTERVAL"
    done
else
    main_once
fi
