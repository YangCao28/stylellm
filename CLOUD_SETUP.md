# 云服务器部署指南

## 1. 克隆代码

```bash
cd /root/workspace  # 或你的工作目录
git clone https://github.com/YangCao28/stylellm.git
cd stylellm
```

## 2. 安装依赖

```bash
# 安装PyTorch (CUDA 12.1)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt
```

## 3. **上传训练数据（重要！）**

### 方法1: 使用scp从本地上传

在**本地Windows**运行：
```powershell
# 上传所有txt文件到云服务器
scp -r "C:\Users\caoya\source\repos\stylellm\《金庸全集（含三联版15部、旧版10部、新修版3部）》全（TXT）作者：金庸" username@your-server-ip:/root/workspace/stylellm/data/
scp -r "C:\Users\caoya\source\repos\stylellm\《古龙全集 76部》全(TXT)作者：古龙" username@your-server-ip:/root/workspace/stylellm/data/
scp -r "C:\Users\caoya\source\repos\stylellm\《梁羽生全集 39部》全（TXT）作者：梁羽生" username@your-server-ip:/root/workspace/stylellm/data/
```

### 方法2: 使用rsync（推荐，支持断点续传）

在**本地Windows**（需安装WSL或Git Bash）：
```bash
rsync -avz --progress "《金庸全集（含三联版15部、旧版10部、新修版3部）》全（TXT）作者：金庸/" username@your-server-ip:/root/workspace/stylellm/data/
rsync -avz --progress "《古龙全集 76部》全(TXT)作者：古龙/" username@your-server-ip:/root/workspace/stylellm/data/
rsync -avz --progress "《梁羽生全集 39部》全（TXT）作者：梁羽生/" username@your-server-ip:/root/workspace/stylellm/data/
```

### 方法3: 打包后上传（最快）

在**本地Windows**：
```powershell
# 1. 打包所有txt文件
cd C:\Users\caoya\source\repos\stylellm
tar -czf wuxia_novels.tar.gz "《金庸全集（含三联版15部、旧版10部、新修版3部）》全（TXT）作者：金庸" "《古龙全集 76部》全(TXT)作者：古龙" "《梁羽生全集 39部》全（TXT）作者：梁羽生" "地煞七十二变.txt"

# 2. 上传到云服务器
scp wuxia_novels.tar.gz username@your-server-ip:/root/workspace/stylellm/
```

在**云服务器**：
```bash
# 3. 解压
cd /root/workspace/stylellm
tar -xzf wuxia_novels.tar.gz -C data/
rm wuxia_novels.tar.gz  # 删除压缩包
```

## 4. 验证数据

```bash
# 检查data目录下是否有txt文件
find data/ -name "*.txt" | wc -l

# 应该看到200+个文件
```

## 5. 下载模型（可选）

```bash
# 方法1: 训练时自动下载（需要网络连接HuggingFace）
# 无需额外操作

# 方法2: 手动下载到本地（推荐，避免训练时下载）
python scripts/download_model.py
```

## 6. 检查GPU

```bash
python scripts/check_gpu.py
```

应该看到：
```
检测到 2 块GPU:
- GPU 0: NVIDIA GeForce RTX 4090 (24GB)
- GPU 1: NVIDIA GeForce RTX 4090 (24GB)
```

## 7. 快速测试

```bash
# 测试1步训练
python tests/quick_test.py
```

## 8. 开始训练

### 单GPU训练（调试）
```bash
python train.py --config train_config.yaml
```

### 多GPU训练（2×RTX 4090）
```bash
torchrun --nproc_per_node=2 train.py --config train_config.yaml
```

## 9. 后台训练（推荐）

```bash
# 使用nohup后台运行
nohup torchrun --nproc_per_node=2 train.py --config train_config.yaml > train.log 2>&1 &

# 查看训练日志
tail -f train.log

# 查看进程
ps aux | grep train.py
```

## 10. 监控训练

```bash
# 实时查看GPU使用情况
watch -n 1 nvidia-smi

# 查看训练日志
tail -f output/logs/train_*.log
```

## 常见问题

### 问题1: 数据文件未找到
```
FileNotFoundError: 在 data/ 目录下没有找到任何.txt文件
```

**解决**：确保已按步骤3上传数据文件到 `data/` 目录

### 问题2: CUDA out of memory
**解决**：减少 `train_config.yaml` 中的 `per_device_train_batch_size`（从4改为2）

### 问题3: 模型下载慢
**解决**：
```bash
# 设置HuggingFace镜像
export HF_ENDPOINT=https://hf-mirror.com
python scripts/download_model.py
```

### 问题4: 多GPU训练报错
**解决**：确保 PyTorch 支持 CUDA：
```bash
python -c "import torch; print(torch.cuda.is_available())"
# 应该输出: True
```

## 预期训练时间

- 数据处理：5-10分钟（首次运行）
- 每个epoch：2-4小时（取决于数据量）
- 5个epochs：10-20小时

## 训练完成后

模型保存在：`output/wuxia_style_alignment/final_model/`

测试生成：
```bash
python scripts/inference.py --model output/wuxia_style_alignment/final_model --prompt "楚留香轻功一跃"
```
