#!/bin/bash
# SFT (监督微调) 训练脚本
# 用于对语言模型进行监督微调

set -e

# 配置
CONFIG_FILE="${CONFIG_FILE:-config.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sft}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-2-7b-hf}"

echo "======================================"
echo "SFT 训练"
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
python -m src.train.train_sft \
    --config "$CONFIG_FILE" \
    --output "$OUTPUT_DIR"

echo "======================================"
echo "SFT 训练完成!"
echo "模型保存到：$OUTPUT_DIR"
echo "======================================"
