"""
LLaMA-Factory集成训练脚本
使用LLaMA-Factory框架进行武侠风格训练
"""

import os
import sys
import json
import yaml
import subprocess
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config


def check_llamafactory_installed():
    """检查LLaMA-Factory是否已安装"""
    try:
        import llamafactory
        print("✓ LLaMA-Factory已安装")
        return True
    except ImportError:
        print("✗ LLaMA-Factory未安装")
        return False


def install_llamafactory():
    """安装LLaMA-Factory"""
    print("\n开始安装LLaMA-Factory...")
    print("这可能需要几分钟...\n")
    
    try:
        # 方法1: 从PyPI安装（如果可用）
        subprocess.run(["pip", "install", "llamafactory"], check=True)
        print("✓ LLaMA-Factory安装成功")
        return True
    except:
        # 方法2: 从GitHub安装
        print("尝试从GitHub安装...")
        try:
            subprocess.run([
                "pip", "install", 
                "git+https://github.com/hiyouga/LLaMA-Factory.git"
            ], check=True)
            print("✓ LLaMA-Factory安装成功")
            return True
        except Exception as e:
            print(f"✗ 安装失败: {e}")
            print("\n手动安装方法:")
            print("1. git clone https://github.com/hiyouga/LLaMA-Factory.git")
            print("2. cd LLaMA-Factory")
            print("3. pip install -e .")
            return False


def create_dataset_info(data_dir: str = "./data", output_file: str = "./dataset_info.json"):
    """创建LLaMA-Factory的dataset_info.json"""
    
    dataset_info = {
        "wuxia_style": {
            "file_name": "processed_wuxia_data.jsonl",
            "formatting": "sharegpt",
            "columns": {
                "messages": "text"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content"
            }
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 数据集配置已创建: {output_file}")


def create_llamafactory_config(config: Config, output_file: str = "./llamafactory_config.yaml"):
    """创建LLaMA-Factory训练配置"""
    
    # 检测GPU
    import torch
    has_gpu = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if has_gpu else 0
    
    llamafactory_config = {
        # 模型配置
        "model_name_or_path": config.model.model_name_or_path,
        "trust_remote_code": True,
        
        # LoRA配置
        "finetuning_type": "lora",
        "lora_rank": config.model.lora_r,
        "lora_alpha": config.model.lora_alpha,
        "lora_dropout": config.model.lora_dropout,
        "lora_target": ",".join(config.model.lora_target_modules),
        
        # 数据配置
        "dataset": "wuxia_style",
        "dataset_dir": ".",
        "cutoff_len": config.data.max_length,
        "val_size": config.data.val_ratio,
        "overwrite_cache": True,
        "preprocessing_num_workers": 4,
        
        # 训练配置
        "output_dir": config.training.output_dir,
        "num_train_epochs": config.training.num_train_epochs,
        "per_device_train_batch_size": config.training.per_device_train_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "learning_rate": config.training.learning_rate,
        "warmup_steps": config.training.warmup_steps,
        "weight_decay": config.training.weight_decay,
        "max_grad_norm": config.training.max_grad_norm,
        "lr_scheduler_type": config.training.lr_scheduler_type,
        
        # 优化器
        "optim": "adamw_torch",
        "adam_beta1": config.training.adam_beta1,
        "adam_beta2": config.training.adam_beta2,
        
        # 保存和日志
        "save_strategy": config.training.save_strategy,
        "save_steps": config.training.save_steps,
        "save_total_limit": config.training.save_total_limit,
        "logging_steps": config.training.logging_steps,
        "eval_strategy": config.training.eval_strategy,
        "eval_steps": config.training.eval_steps,
        
        # GPU优化
        "fp16": config.training.fp16 and has_gpu,
        "bf16": config.training.bf16 and has_gpu,
        "gradient_checkpointing": config.training.gradient_checkpointing,
        "ddp_timeout": 180000000,
        
        # 特殊配置（武侠风格对齐）
        "plot_loss": True,
        "overwrite_output_dir": True,
        
        # GPU配置
        "device_map": "auto" if has_gpu else "cpu",
    }
    
    # 如果有多GPU，启用DDP
    if gpu_count > 1:
        llamafactory_config["ddp_find_unused_parameters"] = False
        llamafactory_config["fsdp"] = ""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(llamafactory_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✓ LLaMA-Factory配置已创建: {output_file}")
    print(f"\nGPU配置:")
    print(f"  可用: {'是' if has_gpu else '否'}")
    if has_gpu:
        print(f"  数量: {gpu_count}")
        print(f"  设备: {[torch.cuda.get_device_name(i) for i in range(gpu_count)]}")
        print(f"  显存: {[f'{torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB' for i in range(gpu_count)]}")
    
    return llamafactory_config


def prepare_data_for_llamafactory():
    """准备数据格式以适配LLaMA-Factory"""
    
    input_file = "processed_wuxia_data.jsonl"
    output_file = "processed_wuxia_data_llama.jsonl"
    
    if not os.path.exists(input_file):
        print(f"⚠️  数据文件不存在: {input_file}")
        print("请先运行数据处理:")
        print("  python train.py  # 会自动处理数据")
        return False
    
    print(f"\n转换数据格式...")
    
    # 转换为LLaMA-Factory格式
    converted = 0
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            data = json.loads(line)
            
            # 转换为ShareGPT格式（用于预训练）
            converted_data = {
                "messages": [
                    {
                        "role": "user",
                        "content": data['text'][:50]  # 前50字作为prompt
                    },
                    {
                        "role": "assistant",
                        "content": data['text']  # 完整文本作为response
                    }
                ]
            }
            
            fout.write(json.dumps(converted_data, ensure_ascii=False) + '\n')
            converted += 1
    
    print(f"✓ 数据转换完成: {converted} 条")
    print(f"  输出: {output_file}")
    
    return True


def train_with_llamafactory(config_file: str = "./llamafactory_config.yaml", use_cli: bool = True):
    """使用LLaMA-Factory进行训练"""
    
    print("\n" + "="*70)
    print("🚀 开始LLaMA-Factory训练")
    print("="*70)
    
    if use_cli:
        # 使用CLI方式（推荐）
        cmd = [
            "llamafactory-cli", "train",
            config_file
        ]
        
        print(f"\n执行命令: {' '.join(cmd)}\n")
        
        try:
            subprocess.run(cmd, check=True)
            print("\n✓ 训练完成！")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n✗ 训练失败: {e}")
            return False
    else:
        # 使用Python API
        try:
            from llamafactory.train import run_exp
            
            # 读取配置
            with open(config_file, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            
            # 运行训练
            run_exp(config_dict)
            
            print("\n✓ 训练完成！")
            return True
        except Exception as e:
            print(f"\n✗ 训练失败: {e}")
            return False


def export_model(checkpoint_dir: str, output_dir: str):
    """导出训练好的模型"""
    
    print(f"\n导出模型...")
    print(f"  从: {checkpoint_dir}")
    print(f"  到: {output_dir}")
    
    cmd = [
        "llamafactory-cli", "export",
        "--model_name_or_path", checkpoint_dir,
        "--output_dir", output_dir,
        "--export_dir", output_dir,
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✓ 模型导出成功")
        return True
    except Exception as e:
        print(f"✗ 导出失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="LLaMA-Factory集成训练")
    parser.add_argument("--install", action="store_true", help="安装LLaMA-Factory")
    parser.add_argument("--prepare", action="store_true", help="准备数据和配置")
    parser.add_argument("--train", action="store_true", help="开始训练")
    parser.add_argument("--config_file", type=str, default="./llamafactory_config.yaml", help="配置文件")
    parser.add_argument("--export", type=str, help="导出模型到指定目录")
    parser.add_argument("--checkpoint", type=str, help="checkpoint目录")
    
    args = parser.parse_args()
    
    # 安装LLaMA-Factory
    if args.install:
        install_llamafactory()
        return
    
    # 准备数据和配置
    if args.prepare:
        print("\n" + "="*70)
        print("准备LLaMA-Factory训练环境")
        print("="*70)
        
        # 检查安装
        if not check_llamafactory_installed():
            print("\n请先安装LLaMA-Factory:")
            print("  python train_llamafactory.py --install")
            return
        
        # 加载配置
        config = Config()
        
        # 创建数据集配置
        create_dataset_info()
        
        # 转换数据格式
        prepare_data_for_llamafactory()
        
        # 创建训练配置
        create_llamafactory_config(config, args.config_file)
        
        print("\n✓ 准备完成！")
        print("\n下一步:")
        print(f"  python train_llamafactory.py --train --config_file {args.config_file}")
        
        return
    
    # 训练
    if args.train:
        if not os.path.exists(args.config_file):
            print(f"✗ 配置文件不存在: {args.config_file}")
            print("请先运行: python train_llamafactory.py --prepare")
            return
        
        success = train_with_llamafactory(args.config_file)
        
        if success:
            print("\n模型已保存！")
            print("\n使用模型:")
            print("  python inference.py --model_path ./output/wuxia_model")
        
        return
    
    # 导出模型
    if args.export:
        if not args.checkpoint:
            print("✗ 请指定checkpoint目录: --checkpoint")
            return
        
        export_model(args.checkpoint, args.export)
        return
    
    # 默认：显示帮助
    print("LLaMA-Factory集成训练工具\n")
    print("使用方法:")
    print("\n1. 安装LLaMA-Factory")
    print("   python train_llamafactory.py --install")
    print("\n2. 准备数据和配置")
    print("   python train_llamafactory.py --prepare")
    print("\n3. 开始训练")
    print("   python train_llamafactory.py --train")
    print("\n4. 导出模型")
    print("   python train_llamafactory.py --export ./output/final --checkpoint ./output/checkpoint-1000")


if __name__ == "__main__":
    main()
