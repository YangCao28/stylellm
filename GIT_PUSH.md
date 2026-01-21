# GitHub 推送指南

## 🚀 快速推送到GitHub

### 步骤1：初始化Git仓库（如果还没有）

```powershell
# 检查是否已是git仓库
git status

# 如果不是，初始化
git init
```

### 步骤2：添加所有文件

```powershell
# 添加所有代码文件（.gitignore会自动排除大文件）
git add .

# 查看将要提交的文件
git status
```

### 步骤3：提交

```powershell
git commit -m "Initial commit: 武侠风格自监督对齐训练框架"
```

### 步骤4：关联GitHub仓库

```powershell
# 方式1：如果是新仓库
# 1. 在GitHub创建新仓库（不要初始化README）
# 2. 关联远程仓库
git remote add origin https://github.com/YangCao28/stylellm.git
git branch -M main
git push -u origin main

# 方式2：如果仓库已存在
git remote add origin https://github.com/YangCao28/stylellm.git
git pull origin main --allow-unrelated-histories
git push -u origin main
```

---

## ☁️ 在云端拉取

### 在云服务器上（Linux）

```bash
# 克隆仓库
git clone https://github.com/YangCao28/stylellm.git
cd stylellm

# 安装依赖（先安装PyTorch）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 准备数据（需要手动上传txt文件到data目录）
# 可以用scp或rsync上传
scp -r ./data/*.txt 用户名@服务器IP:/path/to/stylellm/data/

# 或者在云端直接准备数据
mkdir -p data
# 将武侠小说txt放入data目录

# 检查GPU
python scripts/check_gpu.py

# 开始训练
torchrun --nproc_per_node=2 train.py --config train_config.yaml
```

---

## 📦 提交的文件清单

**会提交：**
- ✅ 所有Python代码（.py文件）
- ✅ 配置文件（train_config.yaml, requirements.txt）
- ✅ 文档（readme/*.md, INSTALL.md）
- ✅ .gitignore

**不会提交（已忽略）：**
- ❌ data/*.txt（武侠小说原文，太大）
- ❌ output/（训练输出）
- ❌ models/（下载的模型）
- ❌ *.bin, *.safetensors（模型权重）
- ❌ __pycache__/, venv/（临时文件）

---

## 🔄 后续更新

```powershell
# 修改代码后
git add .
git commit -m "更新描述"
git push

# 在云端拉取最新代码
git pull
```

---

## 💡 数据传输建议

由于武侠小说txt文件太大，推荐：

### 方式1：使用云存储（推荐）
```bash
# 压缩数据
tar -czf wuxia_data.tar.gz data/*.txt

# 上传到云存储（阿里云OSS/腾讯云COS/AWS S3）
# 然后在云服务器下载
wget https://your-cloud-storage-url/wuxia_data.tar.gz
tar -xzf wuxia_data.tar.gz
```

### 方式2：直接上传
```bash
# 使用scp（适合小文件）
scp -r ./data/*.txt user@server:/path/to/stylellm/data/

# 使用rsync（适合大文件，支持断点续传）
rsync -avz --progress ./data/ user@server:/path/to/stylellm/data/
```

### 方式3：使用Git LFS（如果文件不太大）
```powershell
# 安装Git LFS
git lfs install

# 跟踪txt文件
git lfs track "data/*.txt"

# 提交
git add .gitattributes
git add data/*.txt
git commit -m "添加武侠小说数据"
git push
```

---

## ⚙️ 云端环境配置

```bash
# 1. 更新系统
sudo apt update && sudo apt upgrade -y

# 2. 安装Python 3.10
sudo apt install python3.10 python3.10-venv python3-pip -y

# 3. 创建虚拟环境
python3.10 -m venv venv
source venv/bin/activate

# 4. 安装CUDA（如果需要）
# 参考：https://developer.nvidia.com/cuda-downloads

# 5. 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 6. 验证GPU
python scripts/check_gpu.py
```

---

## 🎯 快速开始命令

```bash
# 一行命令完成环境准备
git clone https://github.com/YangCao28/stylellm.git && \
cd stylellm && \
pip3 install torch --index-url https://download.pytorch.org/whl/cu121 && \
pip install -r requirements.txt && \
python scripts/check_gpu.py
```

---

## 📞 提示

1. **GitHub Token**：如果推送时需要密码，建议使用Personal Access Token
2. **私有仓库**：如果不想公开，在GitHub创建Private仓库
3. **大文件**：建议使用云存储，不要直接提交到Git
4. **模型文件**：训练好的模型可以上传到HuggingFace Model Hub
