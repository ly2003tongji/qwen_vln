#!/bin/bash
# 单卡 (H20) 评估 StreamVLN on R2R-CE val_unseen
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export HF_HOME=/mnt/cpfs/prediction/lyyy/myself/VLN/hf_cache
export CUDA_VISIBLE_DEVICES=0
MASTER_PORT=$((RANDOM % 101 + 20000))

ENVBIN=/mnt/cpfs/prediction/lyyy/myself/VLN/conda_envs/streamvln/bin
CHECKPOINT="checkpoints/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"
echo "CHECKPOINT: ${CHECKPOINT}"

cd /mnt/cpfs/prediction/lyyy/myself/VLN/StreamVLN
$ENVBIN/torchrun --nproc_per_node=1 --master_port=$MASTER_PORT streamvln/streamvln_eval.py \
    --model_path "$CHECKPOINT" \
    --habitat_config_path config/vln_r2r.yaml \
    --eval_split val_unseen \
    --output_path ./results/r2r_val_unseen/streamvln_v1_3
