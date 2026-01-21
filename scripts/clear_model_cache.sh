#!/bin/bash
# 清理HuggingFace缓存中的旧模型

echo "🔍 查找HuggingFace缓存目录..."

CACHE_DIR="$HOME/.cache/huggingface/hub"

if [ ! -d "$CACHE_DIR" ]; then
    echo "❌ 缓存目录不存在: $CACHE_DIR"
    exit 1
fi

echo "📂 缓存目录: $CACHE_DIR"
echo ""

# 查找Qwen3-8B相关的模型
echo "🔍 查找Qwen3-8B模型缓存..."
QWEN8B_DIRS=$(find "$CACHE_DIR" -type d -name "*Qwen3-8B*" -o -name "*Qwen3--8B*" 2>/dev/null)

if [ -z "$QWEN8B_DIRS" ]; then
    echo "✅ 未找到Qwen3-8B缓存"
else
    echo "找到以下Qwen3-8B缓存目录:"
    echo "$QWEN8B_DIRS"
    echo ""
    
    # 计算总大小
    TOTAL_SIZE=$(du -sh $QWEN8B_DIRS 2>/dev/null | awk '{sum+=$1} END {print sum}')
    echo "总大小: ${TOTAL_SIZE}GB"
    echo ""
    
    read -p "是否删除这些缓存？(y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️ 删除中..."
        echo "$QWEN8B_DIRS" | xargs rm -rf
        echo "✅ 删除完成！"
    else
        echo "❌ 取消删除"
    fi
fi

echo ""
echo "📊 当前缓存使用情况:"
du -sh "$CACHE_DIR" 2>/dev/null || echo "无法计算大小"
