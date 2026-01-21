"""
配置加载器 - 支持从YAML文件加载配置
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from config import Config, ModelConfig, TrainingConfig, DataConfig, EvalConfig


def load_yaml_config(config_file: str = "train_config.yaml") -> Dict[str, Any]:
    """
    从YAML文件加载配置
    
    Args:
        config_file: YAML配置文件路径
        
    Returns:
        配置字典
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"配置文件不存在: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)
    
    return yaml_config


def merge_config(default_config: Config, yaml_config: Dict[str, Any]) -> Config:
    """
    合并默认配置和YAML配置
    
    Args:
        default_config: 默认配置对象
        yaml_config: YAML配置字典
        
    Returns:
        合并后的配置对象
    """
    config = Config()
    
    # 合并模型配置
    if 'model' in yaml_config:
        model_dict = yaml_config['model']
        if 'model_name_or_path' in model_dict:
            config.model.model_name_or_path = model_dict['model_name_or_path']
        if 'use_lora' in model_dict:
            config.model.use_lora = model_dict['use_lora']
        if 'lora_r' in model_dict:
            config.model.lora_r = model_dict['lora_r']
        if 'lora_alpha' in model_dict:
            config.model.lora_alpha = model_dict['lora_alpha']
        if 'lora_dropout' in model_dict:
            config.model.lora_dropout = model_dict['lora_dropout']
        if 'lora_target_modules' in model_dict:
            config.model.lora_target_modules = model_dict['lora_target_modules']
    
    # 合并训练配置
    if 'training' in yaml_config:
        training_dict = yaml_config['training']
        for key, value in training_dict.items():
            if hasattr(config.training, key):
                setattr(config.training, key, value)
    
    # 合并数据配置
    if 'data' in yaml_config:
        data_dict = yaml_config['data']
        for key, value in data_dict.items():
            if key == 'span_length_range' and isinstance(value, list):
                setattr(config.data, key, tuple(value))
            elif hasattr(config.data, key):
                setattr(config.data, key, value)
    
    # 合并评估配置
    if 'evaluation' in yaml_config:
        eval_dict = yaml_config['evaluation']
        for key, value in eval_dict.items():
            if hasattr(config.eval, key):
                setattr(config.eval, key, value)
    
    return config


def load_config(config_file: Optional[str] = None, use_yaml: bool = True) -> Config:
    """
    加载配置（优先使用YAML文件）
    
    Args:
        config_file: YAML配置文件路径
        use_yaml: 是否使用YAML配置
        
    Returns:
        配置对象
    """
    # 默认配置
    config = Config()
    
    if not use_yaml:
        return config
    
    # 查找YAML配置文件
    if config_file is None:
        # 按优先级查找
        possible_files = [
            "train_config.yaml",
            "config.yaml",
            "train_config.yml",
            "config.yml",
        ]
        
        for file in possible_files:
            if os.path.exists(file):
                config_file = file
                break
    
    if config_file and os.path.exists(config_file):
        print(f"📝 加载配置文件: {config_file}")
        yaml_config = load_yaml_config(config_file)
        config = merge_config(config, yaml_config)
        print(f"✓ 配置加载成功")
    else:
        print("⚠️  未找到YAML配置文件，使用默认配置")
    
    return config


def save_config_to_yaml(config: Config, output_file: str = "train_config_generated.yaml"):
    """
    将配置对象保存为YAML文件
    
    Args:
        config: 配置对象
        output_file: 输出文件路径
    """
    yaml_dict = {
        'model': {
            'model_name_or_path': config.model.model_name_or_path,
            'use_lora': config.model.use_lora,
            'lora_r': config.model.lora_r,
            'lora_alpha': config.model.lora_alpha,
            'lora_dropout': config.model.lora_dropout,
            'lora_target_modules': config.model.lora_target_modules,
        },
        'training': {
            'output_dir': config.training.output_dir,
            'num_train_epochs': config.training.num_train_epochs,
            'per_device_train_batch_size': config.training.per_device_train_batch_size,
            'per_device_eval_batch_size': config.training.per_device_eval_batch_size,
            'gradient_accumulation_steps': config.training.gradient_accumulation_steps,
            'learning_rate': config.training.learning_rate,
            'warmup_steps': config.training.warmup_steps,
            'weight_decay': config.training.weight_decay,
            'max_grad_norm': config.training.max_grad_norm,
            'kl_beta': config.training.kl_beta,
            'kl_schedule': config.training.kl_schedule,
            'kl_beta_min': config.training.kl_beta_min,
            'kl_beta_max': config.training.kl_beta_max,
            'optimizer': config.training.optimizer,
            'lr_scheduler_type': config.training.lr_scheduler_type,
            'save_strategy': config.training.save_strategy,
            'save_steps': config.training.save_steps,
            'save_total_limit': config.training.save_total_limit,
            'logging_steps': config.training.logging_steps,
            'eval_strategy': config.training.eval_strategy,
            'eval_steps': config.training.eval_steps,
            'fp16': config.training.fp16,
            'bf16': config.training.bf16,
            'gradient_checkpointing': config.training.gradient_checkpointing,
            'seed': config.training.seed,
        },
        'data': {
            'data_dir': config.data.data_dir,
            'processed_data_file': config.data.processed_data_file,
            'max_length': config.data.max_length,
            'min_length': config.data.min_length,
            'stride': config.data.stride,
            'val_ratio': config.data.val_ratio,
            'max_files': config.data.max_files,
            'span_mask_ratio': config.data.span_mask_ratio,
            'token_mask_ratio': config.data.token_mask_ratio,
            'no_mask_ratio': config.data.no_mask_ratio,
            'span_length_range': list(config.data.span_length_range),
        },
        'evaluation': {
            'eval_batch_size': config.eval.eval_batch_size,
            'compute_perplexity': config.eval.compute_perplexity,
            'compute_ngram_overlap': config.eval.compute_ngram_overlap,
            'ngram_sizes': config.eval.ngram_sizes,
            'max_new_tokens': config.eval.max_new_tokens,
            'temperature': config.eval.temperature,
            'top_p': config.eval.top_p,
            'top_k': config.eval.top_k,
            'num_samples': config.eval.num_samples,
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"✓ 配置已保存到: {output_file}")


def print_config(config: Config):
    """打印配置信息"""
    print("\n" + "="*70)
    print("当前配置")
    print("="*70)
    
    print("\n【模型配置】")
    print(f"  模型: {config.model.model_name_or_path}")
    print(f"  使用LoRA: {config.model.use_lora}")
    if config.model.use_lora:
        print(f"  LoRA Rank: {config.model.lora_r}")
        print(f"  LoRA Alpha: {config.model.lora_alpha}")
    
    print("\n【训练配置】")
    print(f"  输出目录: {config.training.output_dir}")
    print(f"  训练轮数: {config.training.num_train_epochs}")
    print(f"  Batch Size: {config.training.per_device_train_batch_size}")
    print(f"  梯度累积: {config.training.gradient_accumulation_steps}")
    print(f"  有效Batch Size: {config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps}")
    print(f"  学习率: {config.training.learning_rate}")
    print(f"  KL Beta: {config.training.kl_beta}")
    
    print("\n【数据配置】")
    print(f"  数据目录: {config.data.data_dir}")
    print(f"  最大长度: {config.data.max_length}")
    print(f"  Span掩码: {config.data.span_mask_ratio*100:.0f}%")
    print(f"  Token掩码: {config.data.token_mask_ratio*100:.0f}%")
    print(f"  无掩码: {config.data.no_mask_ratio*100:.0f}%")
    
    print("\n【GPU优化】")
    print(f"  FP16: {config.training.fp16}")
    print(f"  BF16: {config.training.bf16}")
    print(f"  梯度检查点: {config.training.gradient_checkpointing}")
    
    print("="*70 + "\n")


# 测试代码
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="配置管理工具")
    parser.add_argument("--load", type=str, help="加载YAML配置文件")
    parser.add_argument("--save", action="store_true", help="保存当前配置到YAML")
    parser.add_argument("--print", action="store_true", help="打印配置")
    
    args = parser.parse_args()
    
    if args.load:
        config = load_config(args.load)
        print_config(config)
    elif args.save:
        config = Config()
        save_config_to_yaml(config)
    elif args.print:
        config = load_config()
        print_config(config)
    else:
        print("配置管理工具\n")
        print("使用方法:")
        print("  # 加载配置")
        print("  python config_loader.py --load train_config.yaml")
        print("\n  # 打印当前配置")
        print("  python config_loader.py --print")
        print("\n  # 保存配置到YAML")
        print("  python config_loader.py --save")
