"""
下载基座模型脚本
从HuggingFace下载config.py中定义的模型到本地
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM
from config import Config


def download_model(model_name: str, save_dir: str = "./models", use_mirror: bool = False):
    """
    从HuggingFace下载模型
    
    Args:
        model_name: 模型名称（如：Qwen/Qwen2.5-7B）
        save_dir: 保存目录
        use_mirror: 是否使用镜像站（国内用户推荐）
    """
    print("="*70)
    print("🔽 下载基座模型")
    print("="*70)
    print(f"\n模型: {model_name}")
    print(f"保存到: {save_dir}")
    
    # 创建保存目录
    save_path = Path(save_dir) / model_name.replace("/", "_")
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 设置镜像（国内用户）
    if use_mirror:
        print("\n使用HuggingFace镜像站（国内加速）...")
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    try:
        # 下载tokenizer
        print("\n[1/2] 下载Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=save_path
        )
        tokenizer.save_pretrained(save_path)
        print(f"✓ Tokenizer已保存")
        
        # 下载模型
        print("\n[2/2] 下载模型（这可能需要较长时间）...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=save_path,
            torch_dtype="auto",  # 自动选择精度
        )
        model.save_pretrained(save_path)
        print(f"✓ 模型已保存")
        
        print("\n" + "="*70)
        print("✅ 下载完成！")
        print("="*70)
        print(f"\n模型保存在: {save_path}")
        print(f"\n使用方法:")
        print(f"  python train.py --model_name {save_path}")
        
        return str(save_path)
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 使用镜像站: python download_model.py --use_mirror")
        print("3. 使用VPN")
        print("4. 手动下载模型文件")
        return None


def download_multiple_models(models: list, save_dir: str = "./models", use_mirror: bool = False):
    """批量下载多个模型"""
    print(f"\n准备下载 {len(models)} 个模型...\n")
    
    results = {}
    for i, model_name in enumerate(models, 1):
        print(f"\n{'='*70}")
        print(f"下载进度: [{i}/{len(models)}]")
        print(f"{'='*70}")
        
        result = download_model(model_name, save_dir, use_mirror)
        results[model_name] = result
        
        if result:
            print(f"✓ {model_name} 下载成功")
        else:
            print(f"✗ {model_name} 下载失败")
    
    # 总结
    print("\n" + "="*70)
    print("📊 下载总结")
    print("="*70)
    
    success_count = sum(1 for v in results.values() if v)
    print(f"\n成功: {success_count}/{len(models)}")
    
    print("\n成功下载的模型:")
    for model, path in results.items():
        if path:
            print(f"  ✓ {model}")
            print(f"    → {path}")
    
    failed = [model for model, path in results.items() if not path]
    if failed:
        print("\n失败的模型:")
        for model in failed:
            print(f"  ✗ {model}")


def list_recommended_models():
    """列出推荐的模型"""
    print("\n" + "="*70)
    print("📚 推荐的武侠风格训练模型")
    print("="*70)
    
    models = {
        "测试用（小模型）": [
            ("gpt2", "124M", "英文，快速测试"),
            ("uer/gpt2-chinese-cluecorpussmall", "124M", "中文GPT2"),
        ],
        "生产用（7B级）": [
            ("Qwen/Qwen2.5-7B", "7B", "通义千问，性能优秀"),
            ("meta-llama/Llama-3.1-8B", "8B", "LLaMA3.1，需申请"),
            ("01-ai/Yi-1.5-9B", "9B", "零一万物，中文友好"),
        ],
        "轻量级（1-3B）": [
            ("Qwen/Qwen2.5-1.5B", "1.5B", "显存友好"),
            ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "1.1B", "超轻量"),
        ],
    }
    
    for category, model_list in models.items():
        print(f"\n【{category}】")
        for name, size, desc in model_list:
            print(f"  • {name}")
            print(f"    大小: {size}, {desc}")


def check_disk_space(required_gb: float = 20):
    """检查磁盘空间"""
    import shutil
    
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024**3)
    
    print(f"\n磁盘空间检查:")
    print(f"  可用空间: {free_gb:.2f} GB")
    print(f"  建议空间: {required_gb} GB")
    
    if free_gb < required_gb:
        print(f"  ⚠️  警告: 磁盘空间可能不足！")
        return False
    else:
        print(f"  ✓ 空间充足")
        return True


def main():
    parser = argparse.ArgumentParser(description="从HuggingFace下载基座模型")
    parser.add_argument("--model_name", type=str, help="模型名称")
    parser.add_argument("--save_dir", type=str, default="./models", help="保存目录")
    parser.add_argument("--use_mirror", action="store_true", help="使用镜像站（国内加速）")
    parser.add_argument("--list_models", action="store_true", help="列出推荐模型")
    parser.add_argument("--use_config", action="store_true", help="使用config.py中的模型")
    parser.add_argument("--batch", nargs="+", help="批量下载多个模型")
    parser.add_argument("--check_space", action="store_true", help="检查磁盘空间")
    
    args = parser.parse_args()
    
    # 列出推荐模型
    if args.list_models:
        list_recommended_models()
        return
    
    # 检查磁盘空间
    if args.check_space:
        check_disk_space()
        return
    
    # 从config读取
    if args.use_config:
        config = Config()
        model_name = config.model.model_name_or_path
        print(f"使用config.py中的模型: {model_name}")
        download_model(model_name, args.save_dir, args.use_mirror)
        return
    
    # 批量下载
    if args.batch:
        download_multiple_models(args.batch, args.save_dir, args.use_mirror)
        return
    
    # 单个下载
    if args.model_name:
        check_disk_space()
        download_model(args.model_name, args.save_dir, args.use_mirror)
    else:
        print("请指定要下载的模型！\n")
        print("使用示例:")
        print("  # 下载单个模型")
        print("  python download_model.py --model_name Qwen/Qwen2.5-7B")
        print("\n  # 使用镜像站（国内用户）")
        print("  python download_model.py --model_name Qwen/Qwen2.5-7B --use_mirror")
        print("\n  # 使用config.py中的模型")
        print("  python download_model.py --use_config --use_mirror")
        print("\n  # 批量下载")
        print("  python download_model.py --batch gpt2 Qwen/Qwen2.5-1.5B")
        print("\n  # 查看推荐模型")
        print("  python download_model.py --list_models")


if __name__ == "__main__":
    main()
