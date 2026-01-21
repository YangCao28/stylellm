#!/bin/bash
# 清理HuggingFace缓存中的旧模型

echo "🔍 查找HuggingFace缓存目录..."

# 可能的缓存位置
CACHE_DIRS=(
    "$HOME/.cache/huggingface/hub"
    "$HOME/.cache/huggingface/transformers"
    "$HOME/.cache/huggingface"
    "./models"
    "../models"
    "/root/workspace/stylellm/models"
    "${HF_HOME:-}"
    "${TRANSFORMERS_CACHE:-}"
)

FOUND_DIR=""
for dir in "${CACHE_DIRS[@]}"; do
    if [ -n "$dir" ] && [ -d "$dir" ]; then
        echo "✅ 找到缓存目录: $dir"
        FOUND_DIR="$dir"
        break
    fi
done

if [ -z "$FOUND_DIR" ]; then
    echo "❌ 未找到HuggingFace缓存目录"
    echo "可能的原因："
    echo "  1. 还没有下载过模型"
    echo "  2. 使用了自定义缓存路径"
    echo ""
    echo "当前环境变量："
    echo "  HF_HOME=${HF_HOME:-未设置}"
    echo "  TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-未设置}"
    echo ""
    echo "等模型下载后会自动创建缓存目录"
    exit 0
fi

echo "📂 缓存目录: $FOUND_DIR"
echo ""

# 查找所有Qwen模型
echo "🔍 查找所有Qwen模型缓存..."
QWEN_DIRS=$(find "$FOUND_DIR" -type d \( -name "*Qwen*" -o -name "*qwen*" \) 2>/dev/null)

if [ -z "$QWEN_DIRS" ]; then
    echo "✅ 未找到Qwen模型缓存"
    echo ""
    echo "📊 当前缓存使用情况:"
    du -sh "$FOUND_DIR" 2>/dev/null || echo "无法计算大小"
    exit 0
fi

echo "找到以下Qwen模型缓存:"
echo "$QWEN_DIRS"
echo ""

# 计算总大小
echo "📊 计算缓存大小..."
du -sh $QWEN_DIRS 2>/dev/null | awk '{print "  " $2 ": " $1}'
echo ""

read -p "是否删除所有Qwen模型缓存？(y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️ 删除中..."
    echo "$QWEN_DIRS" | xargs rm -rf
    echo "✅ 删除完成！"
else
    echo "❌ 取消删除"
fi

echo ""
echo "📊 当前缓存使用情况:"
du -sh "$FOUND_DIR" 2>/dev/null || echo "无法计算大小"
