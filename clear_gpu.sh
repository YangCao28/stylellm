#!/bin/bash

echo "清理GPU显存..."
echo ""

# 1. 显示当前GPU使用情况
echo "当前GPU状态:"
nvidia-smi
echo ""

# 2. 杀死所有Python进程
echo "杀死所有Python进程..."
pkill -9 python
pkill -9 python3
sleep 2

# 3. 清理PyTorch缓存
echo "清理PyTorch缓存..."
python3 -c "import torch; torch.cuda.empty_cache(); print('✓ PyTorch缓存已清理')" 2>/dev/null || echo "⚠ PyTorch未安装"

# 4. 再次显示GPU状态
echo ""
echo "清理后GPU状态:"
nvidia-smi

echo ""
echo "✓ GPU清理完成"
