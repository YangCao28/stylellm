"""
极简版武侠风格训练脚本
纯 LoRA 微调，无 KL 散度
"""

import os
import argparse
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from style_alignment_model import StyleAlignmentModel
from data.wuxia_processor import WuxiaDataProcessor
from utils.config_loader import load_config, print_config


def setup_data(config):
    """
    数据准备
    
    Returns:
        tokenizer, train_dataset, eval_dataset
    """
    print("="*60)
    print("准备数据...")
    print("="*60)
    
    # 1. 加载tokenizer
    print(f"\n1. 加载 tokenizer: {config.model.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name_or_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"词表大小: {len(tokenizer)}")
    
    # 2. 处理数据
    print(f"\n2. 处理数据...")
    
    need_process = False
    if not os.path.exists(config.data.processed_data_file):
        need_process = True
        print(f"处理原始数据: {config.data.data_dir}")
    else:
        if os.path.getsize(config.data.processed_data_file) == 0:
            need_process = True
            print(f"文件为空，重新处理")
        else:
            print(f"使用已处理数据: {config.data.processed_data_file}")
    
    if need_process:
        processor = WuxiaDataProcessor(
            tokenizer=tokenizer,
            max_length=config.data.max_length,
            min_length=config.data.min_length,
            stride=config.data.stride,
        )
        
        chunks = processor.process_directory(
            data_dir=config.data.data_dir,
            recursive=True
        )
        
        if len(chunks) == 0:
            raise ValueError(
                f"未找到有效数据！\n"
                f"请检查 {config.data.data_dir} 目录下是否有txt文件"
            )
        
        processor.save_to_jsonl(chunks, config.data.processed_data_file)
        print(f"✓ 处理完成: {len(chunks)} 个文本块")
        
        if len(chunks) == 0:
            raise ValueError("处理后的数据为空")
    
    # 3. 加载数据集
    print(f"\n3. 加载数据集...")
    dataset = load_dataset('json', data_files=config.data.processed_data_file, split='train')
    print(f"总样本数: {len(dataset)}")
    
    # 4. Tokenize
    print(f"\n4. Tokenizing...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=config.data.max_length,
            padding=False,
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text', 'source'],
        desc="Tokenizing"
    )
    
    # 5. 划分训练/验证集
    print(f"\n5. 划分数据集...")
    split_dataset = tokenized_dataset.train_test_split(
        test_size=config.data.val_ratio,
        seed=config.training.seed
    )
    
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(eval_dataset)} 样本")
    
    return tokenizer, train_dataset, eval_dataset


def train(config):
    """训练主函数"""
    
    # 准备数据
    tokenizer, train_dataset, eval_dataset = setup_data(config)
    
    # 加载模型
    print(f"\n6. 加载模型...")
    print("="*60)
    
    lora_config = {
        "r": config.model.lora_r,
        "lora_alpha": config.model.lora_alpha,
        "lora_dropout": config.model.lora_dropout,
        "target_modules": config.model.lora_target_modules,
    }
    
    model = StyleAlignmentModel(
        model_name_or_path=config.model.model_name_or_path,
        lora_config=lora_config,
        device="cuda:0",
    )
    
    print("\n✓ 模型加载完成!")
    
    # Data collator（自动处理MLM）
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 因果语言模型
    )
    
    # 训练参数
    print(f"\n7. 配置训练参数...")
    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps,
        save_total_limit=config.training.save_total_limit,
        fp16=config.training.fp16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        logging_dir=f"{config.training.output_dir}/logs",
        ddp_find_unused_parameters=False,
    )
    
    # 创建 Trainer
    trainer = Trainer(
        model=model.model,  # 直接传入内部模型
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    print("\n" + "="*60)
    print("开始训练...")
    print("="*60)
    
    trainer.train()
    
    # 保存最终模型
    print(f"\n保存模型到: {config.training.output_dir}")
    model.save_pretrained(config.training.output_dir)
    tokenizer.save_pretrained(config.training.output_dir)
    
    print("\n✓ 训练完成!")
    
    return model, tokenizer


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="武侠风格 LoRA 微调")
    parser.add_argument("--config", type=str, default="train_config.yaml", 
                        help="配置文件路径")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    
    args = parser.parse_args()
    
    # 加载配置
    print(f"\n从配置文件加载: {args.config}")
    config = load_config(args.config)
    
    # 命令行覆盖
    if args.model_name:
        config.model.model_name_or_path = args.model_name
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.epochs:
        config.training.num_train_epochs = args.epochs
    
    # 打印配置
    print_config(config)
    
    # 训练
    train(config)


if __name__ == "__main__":
    main()
