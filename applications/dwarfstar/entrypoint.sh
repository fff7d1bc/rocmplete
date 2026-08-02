#!/usr/bin/env bash
set -Eeuo pipefail

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

profile="${ROCMLETE_PROFILE:-auto}"
mode="${ROCMLETE_DWARFSTAR_MODE:-server}"
model="${ROCMLETE_DWARFSTAR_MODEL:-}"
listen="${ROCMLETE_LISTEN:-0.0.0.0}"
host_listen="${ROCMLETE_HOST_LISTEN:-unknown}"
port="${ROCMLETE_PORT:-8000}"
context="${ROCMLETE_DWARFSTAR_CONTEXT:-131072}"
output_tokens="${ROCMLETE_DWARFSTAR_OUTPUT_TOKENS:-16000}"
prompt="${ROCMLETE_DWARFSTAR_PROMPT:-}"
no_thinking="${ROCMLETE_DWARFSTAR_NO_THINKING:-0}"

case "$profile" in
    auto|rdna4|strix-halo|strix-point) ;;
    *) die "unknown DwarfStar profile '$profile'" ;;
esac
case "$mode" in
    server|cli) ;;
    *) die "unknown DwarfStar mode '$mode'" ;;
esac
case "$no_thinking" in
    0|1) ;;
    *) die "invalid DwarfStar thinking setting '$no_thinking'" ;;
esac
[[ "$port" =~ ^[0-9]+$ ]] && ((port >= 1 && port <= 65535)) ||
    die "invalid DwarfStar port '$port'"
[[ "$context" =~ ^[0-9]+$ ]] && ((context >= 4096 && context <= 1048576)) ||
    die "DwarfStar context must be between 4096 and 1048576 tokens"
[[ "$output_tokens" =~ ^[0-9]+$ ]] &&
    ((output_tokens >= 1 && output_tokens < context)) ||
    die "DwarfStar output tokens must be positive and smaller than context"
[[ -n "$model" ]] || die "no DwarfStar GGUF model was selected"
[[ -f "$model" && -r "$model" ]] ||
    die "DwarfStar GGUF model is not a readable regular file: $model"

mapfile -t architectures < <(
    rocminfo 2>/dev/null |
        awk '$1 == "Name:" && $2 ~ /^gfx[0-9]+$/ { print $2 }' |
        sort -u
)
((${#architectures[@]} == 1)) ||
    die "expected exactly one visible AMD GPU architecture"
architecture="${architectures[0]}"
case "$architecture" in
    gfx1200|gfx1201) detected_profile=rdna4 ;;
    gfx1151) detected_profile=strix-halo ;;
    gfx1150) detected_profile=strix-point ;;
    *) die "unsupported GPU architecture '$architecture'" ;;
esac
if [[ "$profile" != auto && "$profile" != "$detected_profile" ]]; then
    die "profile '$profile' does not match detected architecture '$architecture'"
fi
profile="$detected_profile"

mkdir -p /data/home

printf '\nROCmplete: DwarfStar %s\n' "$mode"
printf '  profile:        %s\n' "$profile"
printf '  architecture:   %s\n' "$architecture"
printf '  model:          %s\n' "$model"
printf '  context:        %s tokens\n' "$context"
printf '  output limit:   %s tokens\n' "$output_tokens"
if [[ "$mode" == server ]]; then
    printf '  container bind: %s:%s (container namespace)\n' "$listen" "$port"
    printf '  host publish:   %s:%s\n' "$host_listen" "$port"
fi
printf '\n'

common=(
    --rocm
    --model "$model"
    --ctx "$context"
    --tokens "$output_tokens"
)

if [[ "$mode" == server ]]; then
    exec ds4-server \
        "${common[@]}" \
        --host "$listen" \
        --port "$port"
fi

thinking=(--think)
if [[ "$no_thinking" == 1 ]]; then
    thinking=(--nothink)
fi
if [[ -n "$prompt" ]]; then
    exec ds4 "${common[@]}" "${thinking[@]}" --prompt "$prompt"
fi
exec ds4 "${common[@]}" "${thinking[@]}"
