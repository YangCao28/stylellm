#!/bin/bash

echo "清理Hugging Face模型缓存..."
echo ""

# 1. 显示当前缓存大小
if [ -d "$HOME/.cache/huggingface" ]; then
    echo "当前缓存大小:"
    du -sh $HOME/.cache/huggingface
    echo ""
fi

# 2. 列出已下载的模型
echo "已下载的模型:"
if [ -d "$HOME/.cache/huggingface/hub" ]; then
    ls -lh $HOME/.cache/huggingface/hub | grep "^d" | awk '{print $9, $5}'
    echo ""
fi

# 3. 询问确认
read -p "确认删除所有模型缓存? (y/N): " confirm

if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
    echo ""
    echo "删除中..."
    rm -rf $HOME/.cache/huggingface/hub
    echo "✓ 已删除模型缓存"
    echo ""
    
    # 显示剩余空间
    echo "磁盘空间:"
    df -h $HOME | tail -1
else
    echo "已取消"
fi
