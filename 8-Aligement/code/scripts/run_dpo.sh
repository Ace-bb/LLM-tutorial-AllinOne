#!/bin/bash
# DPO (直接偏好优化) 训练脚本
# 使用 DPO 算法微调语言模型

set -e

# 配置
CONFIG_FILE="${CONFIG_FILE:-config.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/dpo}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-2-7b-hf}"

echo "======================================"
echo "DPO 训练"
echo "======================================"
echo "配置文件：$CONFIG_FILE"
echo "输出目录：$OUTPUT_DIR"
echo "基础模型：$MODEL_NAME"
echo "======================================"

# 检查 GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU 信息:"
    nvidia-smi --query-gpu=name,memory.total --format=csv
fi

# 运行训练
python -m src.train.train_dpo \
    --config "$CONFIG_FILE" \
    --output "$OUTPUT_DIR"

echo "======================================"
echo "DPO 训练完成!"
echo "模型保存到：$OUTPUT_DIR"
echo "======================================"
