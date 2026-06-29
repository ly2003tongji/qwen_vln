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

echo "===== [1/12] 创建 python 3.9 环境 ====="
conda create -p "$ENV" -c conda-forge python=3.9 -y

PY="$ENV/bin/python"
PIP="$PY -m pip"

echo "===== [2/12] 安装 habitat-sim 0.2.4 (headless+bullet) ====="
conda install -p "$ENV" habitat-sim==0.2.4 withbullet headless -c conda-forge -c aihabitat -y

echo "===== [3/12] 安装 habitat-lab / habitat-baselines ====="
cd "$VLN/habitat-lab"
$PIP install -e habitat-lab
$PIP install -e habitat-baselines --no-deps

echo "===== [4/12] 安装 torch 2.1.2 cu121 ====="
$PIP install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

echo "===== [5/12] 安装 av 14.2.0 (wheel) ====="
$PIP install av==14.2.0 --only-binary=:all: --index-url https://pypi.org/simple

echo "===== [6/12] 安装 StreamVLN requirements (去 av/wavedrom) + torch-scatter ====="
cd "$VLN"
$PIP install -r requirements_clean.txt -f https://data.pyg.org/whl/torch-2.1.2+cu121.html --prefer-binary

echo "===== [7/12] 安装 habitat-baselines 运行期依赖 + gdown ====="
$PIP install tensorboard ifcfg webdataset==0.1.40 moviepy gdown --prefer-binary

echo "===== [8/12] 安装 flash-attn (本地 wheel) ====="
$PIP install "$VLN/flash_attn-2.5.8+cu122torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl" --no-deps

echo "===== [9/12] 修复 cuBLAS (torch2.1 自带的 cuBLAS12.1 在 H20 上小矩阵乘 SIGFPE) ====="
# 用 cuBLAS 12.4 替换 torch 自带的 12.1，否则 LLM forward 在 H20 上崩溃(SIGFPE)
$PIP install "nvidia-cublas-cu12==12.4.5.8" --prefer-binary
TL="$ENV/lib/python3.9/site-packages/torch/lib"
NV="$ENV/lib/python3.9/site-packages/nvidia/cublas/lib"
mkdir -p "$TL/_bak_cublas121"
cp -f "$TL/libcublas.so.12" "$TL/_bak_cublas121/" 2>/dev/null || true
cp -f "$TL/libcublasLt.so.12" "$TL/_bak_cublas121/" 2>/dev/null || true
cp -f "$NV/libcublas.so.12" "$TL/libcublas.so.12"
cp -f "$NV/libcublasLt.so.12" "$TL/libcublasLt.so.12"

echo "===== [10/12] 为权重补 tokenizer (HF 权重仓库未带 tokenizer，取自基座 LLaVA-Video-7B-Qwen2) ====="
CKPT="$VLN/StreamVLN/checkpoints/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"
if [ ! -f "$CKPT/tokenizer.json" ]; then
  HF_HOME="$VLN/hf_cache" "$PY" - <<PYEOF
from huggingface_hub import hf_hub_download
import shutil, os
ckpt="$CKPT"
for f in ["added_tokens.json","merges.txt","special_tokens_map.json","tokenizer.json","tokenizer_config.json","vocab.json"]:
    p=hf_hub_download("lmms-lab/LLaVA-Video-7B-Qwen2", f)
    shutil.copy(p, os.path.join(ckpt,f))
print("tokenizer ready")
PYEOF
fi

echo "===== [11/12] 修补 habitat-baselines ver/queue.py (BatchedQueue 在本环境继承 torch.multiprocessing.Queue 报错) ====="
QFILE="$VLN/habitat-lab/habitat-baselines/habitat_baselines/rl/ver/queue.py"
if ! grep -q "multiprocessing.queues.Queue" "$QFILE"; then
  "$PY" - <<PYEOF
f="$QFILE"; s=open(f).read()
old="    class BatchedQueue(torch.multiprocessing.Queue):\n        def get_many("
new=("    import multiprocessing.queues\n\n"
     "    class BatchedQueue(multiprocessing.queues.Queue):\n"
     "        def __init__(self, *args, ctx=None, **kwargs):\n"
     "            if ctx is None:\n"
     "                ctx = torch.multiprocessing.get_context()\n"
     "            super().__init__(*args, ctx=ctx, **kwargs)\n\n"
     "        def get_many(")
assert old in s, "pattern not found, queue.py 可能已变更"
open(f,"w").write(s.replace(old,new)); print("queue.py patched")
PYEOF
else
  echo "queue.py 已是补丁版本，跳过"
fi

echo "===== [12/12] 生成单卡评估脚本 ====="
EVAL_SH="$VLN/StreamVLN/scripts/streamvln_eval_single_gpu.sh"
cat > "$EVAL_SH" <<EOF
#!/bin/bash
# 单卡 (H20) 评估 StreamVLN on R2R-CE val_unseen
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export HF_HOME=$VLN/hf_cache
export CUDA_VISIBLE_DEVICES=0
MASTER_PORT=\$((RANDOM % 101 + 20000))
ENVBIN=$ENV/bin
CHECKPOINT="checkpoints/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"
cd $VLN/StreamVLN
\$ENVBIN/torchrun --nproc_per_node=1 --master_port=\$MASTER_PORT streamvln/streamvln_eval.py \\
    --model_path "\$CHECKPOINT" \\
    --habitat_config_path config/vln_r2r.yaml \\
    --eval_split val_unseen \\
    --output_path ./results/r2r_val_unseen/streamvln_v1_3
EOF
chmod +x "$EVAL_SH"
echo "eval 脚本: $EVAL_SH"

echo "ENV_SETUP_DONE env=$ENV"
