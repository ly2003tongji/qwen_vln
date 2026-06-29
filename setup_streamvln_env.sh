#!/bin/bash
# 在持久目录(/mnt/cpfs)重建 streamvln conda 环境。/root 非持久，重启会丢，故必须装在 cpfs。
# 重启后若环境丢失，重新运行本脚本即可。
set -e
source /root/anaconda3/etc/profile.d/conda.sh
conda config --set safety_checks disabled || true

ENVS_DIR=/mnt/cpfs/prediction/lyyy/myself/VLN/conda_envs
ENV=$ENVS_DIR/streamvln
VLN=/mnt/cpfs/prediction/lyyy/myself/VLN
mkdir -p "$ENVS_DIR"
conda config --prepend envs_dirs "$ENVS_DIR" || true

echo "===== [1/8] 创建 python 3.9 环境 ====="
conda create -p "$ENV" -c conda-forge python=3.9 -y

PY="$ENV/bin/python"
PIP="$PY -m pip"

echo "===== [2/8] 安装 habitat-sim 0.2.4 (headless+bullet) ====="
conda install -p "$ENV" habitat-sim==0.2.4 withbullet headless -c conda-forge -c aihabitat -y

echo "===== [3/8] 安装 habitat-lab / habitat-baselines ====="
cd "$VLN/habitat-lab"
$PIP install -e habitat-lab
$PIP install -e habitat-baselines --no-deps

echo "===== [4/8] 安装 torch 2.1.2 cu121 ====="
$PIP install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

echo "===== [5/8] 安装 av 14.2.0 (wheel) ====="
$PIP install av==14.2.0 --only-binary=:all: --index-url https://pypi.org/simple

echo "===== [6/8] 安装 StreamVLN requirements (去 av/wavedrom) + torch-scatter ====="
cd "$VLN"
$PIP install -r requirements_clean.txt -f https://data.pyg.org/whl/torch-2.1.2+cu121.html --prefer-binary

echo "===== [7/8] 安装 habitat-baselines 运行期依赖 + gdown ====="
$PIP install tensorboard ifcfg webdataset==0.1.40 moviepy gdown --prefer-binary

echo "===== [8/8] 安装 flash-attn (本地 wheel) ====="
$PIP install "$VLN/flash_attn-2.5.8+cu122torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl" --no-deps

echo "ENV_SETUP_DONE env=$ENV"
