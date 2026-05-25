#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${WORLD_SIZE:-}" && "${WORLD_SIZE}" -gt 1 && -n "${RANK:-}" && -n "${LOCAL_RANK:-}" ]]; then
    LAUNCH_CMD=(python3 -u)
else
    if [[ -n "${NPROC_PER_NODE:-}" ]]; then
        PROC_COUNT="${NPROC_PER_NODE}"
    elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -ra CUDA_DEVICES_ARR <<< "${CUDA_VISIBLE_DEVICES}"
        PROC_COUNT="${#CUDA_DEVICES_ARR[@]}"
    else
        PROC_COUNT=1
    fi

    if [[ "${PROC_COUNT}" -gt 1 ]]; then
        LAUNCH_CMD=(torchrun --standalone --nproc_per_node="${PROC_COUNT}")
    else
        LAUNCH_CMD=(python3 -u)
    fi
fi

# ---- Active config: RankMixer + SlidingWindow RelativeTimeBias + DeepFeatureInteraction ----
# Contrastive loss disabled by default; enable with --contrastive_loss_weight 0.02 ~ 0.05 after baseline verified.
# Token budget: num_queries*4 + user_ns(4) + aligned_pair(4) + user_dense(1) + item_ns(3) = 16.
"${LAUNCH_CMD[@]}" "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 4 \
    --item_ns_tokens 3 \
    --num_queries 1 \
    --use_aligned_pair_tokens \
    --aligned_pair_token_count 4 \
    --aligned_pair_value_interaction scale_shift \
    --drop_aligned_pair_from_base \
    --ns_groups_json "" \
    --emb_skip_threshold 60000000 \
    --num_workers 8 \
    --num_interaction_layers 2 \
    --seed 2002 \
    "$@"



