# 安装指南

## 🚀 快速安装（推荐）

### Windows + CUDA 环境

```powershell
# 步骤1：安装PyTorch（带CUDA支持）
# 访问 https://pytorch.org/ 选择合适版本，或使用以下命令：
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 步骤2：验证PyTorch和CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 步骤3：安装其他依赖
pip install -r requirements.txt

# 步骤4：验证安装
python scripts/check_gpu.py
```

---

## 📦 详细安装步骤

### 1. Python环境
```powershell
# 确保使用Python 3.9-3.11（推荐3.10）
python --version

# 建议使用虚拟环境
python -m venv venv
.\venv\Scripts\activate
```

### 2. CUDA支持（RTX 4090需要）
```powershell
# 检查CUDA版本
nvidia-smi

# 根据CUDA版本安装PyTorch：
# CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1（推荐，RTX 4090支持更好）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 验证CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### 3. 安装核心依赖
```powershell
pip install -r requirements.txt
```

### 4. 验证安装
```powershell
python scripts/check_gpu.py
python tests/test_modules.py
```

---

## ⚠️ 常见问题

### Q1: flash-attn 安装失败？
**A:** flash-attn 已从 requirements.txt 中移除（可选依赖）。
- 不影响训练，只是速度稍慢（~10-15%）
- 如果需要安装：
  ```powershell
  # 需要Visual Studio Build Tools和CUDA Toolkit
  pip install flash-attn --no-build-isolation
  ```

### Q2: bitsandbytes 在Windows上不可用？
**A:** Windows支持有限，已注释掉。
- 不影响LoRA训练
- 如果需要量化，使用Linux环境

### Q3: torch安装很慢？
**A:** 使用国内镜像：
```powershell
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q4: CUDA out of memory？
**A:** 降低batch size：
```yaml
# train_config.yaml
per_device_train_batch_size: 2  # 4→2
```

---

## 🔧 可选依赖

### Flash Attention 2（可选，提速10-20%）
```powershell
# 需要：
# 1. Visual Studio Build Tools 2019+
# 2. CUDA Toolkit 11.8+
# 3. 从源码编译

pip install flash-attn --no-build-isolation
```

### 4-bit量化（可选，节省显存）
```powershell
# Linux/WSL2推荐
pip install bitsandbytes
```

### LLaMA-Factory集成（可选）
```powershell
pip install llamafactory
python scripts/train_llamafactory.py --install
```

---

## ✅ 安装检查清单

运行以下命令验证安装：

```powershell
# 1. Python版本
python --version  # 应该是3.9-3.11

# 2. PyTorch + CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 3. Transformers
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# 4. GPU检测
python scripts/check_gpu.py

# 5. 模块测试
python tests/test_modules.py
```

预期输出：
```
✓ PyTorch: 2.x.x, CUDA: True
✓ Transformers: 4.35+
✓ 找到 2 个GPU
✓ GPU 0: NVIDIA GeForce RTX 4090
✓ GPU 1: NVIDIA GeForce RTX 4090
✓ 所有模块测试通过
```

---

## 🚀 快速开始

安装完成后：
```powershell
# 1. 检查配置
cat train_config.yaml

# 2. 快速测试
python tests/quick_test.py

# 3. 开始训练
torchrun --nproc_per_node=2 train.py --config train_config.yaml
```

---

## 📞 获取帮助

如果遇到问题：
1. 查看 [README.md](readme/README.md)
2. 查看 [CONFIG_GUIDE.md](readme/CONFIG_GUIDE.md)
3. 运行 `python scripts/check_gpu.py --test` 进行诊断
