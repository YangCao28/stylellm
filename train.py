"""
超简化训练脚本 - 无依赖版本
"""

import os
import argparse
import yaml
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from style_alignment_model import StyleAlignmentModel


def load_config(config_file):
    """加载YAML配置"""
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 转换为简单的命名空间
    class Config:
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, Config(v))
                else:
                    setattr(self, k, v)
    
    return Config(config)


def process_text_files(data_dir, output_file, tokenizer, max_length=512, min_length=50):
    """处理文本文件"""
    print(f"处理 {data_dir} 中的txt文件...")
    
    import json
    from pathlib import Path
    
    texts = []
    
    # 递归查找所有txt文件
    for txt_file in Path(data_dir).rglob("*.txt"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 分段
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            for para in paragraphs:
                if len(para) >= min_length:
                    texts.append({
                        'text': para,
                        'source': str(txt_file.name)
                    })
        except Exception as e:
            print(f"跳过 {txt_file}: {e}")
    
    # 保存为jsonl
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in texts:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✓ 处理完成: {len(texts)} 个文本块")
    return len(texts)


def train(config_file):
    """训练主函数"""
    
    # 加载配置
    print(f"加载配置: {config_file}")
    cfg = load_config(config_file)
    
    # 1. Tokenizer
    print(f"\n1. 加载 tokenizer: {cfg.model.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name_or_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 处理数据
    print(f"\n2. 准备数据...")
    
    if not os.path.exists(cfg.data.processed_data_file) or \
       os.path.getsize(cfg.data.processed_data_file) == 0:
        process_text_files(
            cfg.data.data_dir,
            cfg.data.processed_data_file,
            tokenizer,
            cfg.data.max_length,
            cfg.data.min_length
        )
    else:
        print(f"使用已处理数据: {cfg.data.processed_data_file}")
    
    # 3. 加载数据集
    print(f"\n3. 加载数据集...")
    dataset = load_dataset('json', data_files=cfg.data.processed_data_file, split='train')
    print(f"总样本数: {len(dataset)}")
    
    # 4. Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=cfg.data.max_length,
            padding=False,
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text', 'source'],
    )
    
    # 5. 划分数据集
    split = tokenized_dataset.train_test_split(test_size=cfg.data.val_ratio, seed=cfg.training.seed)
    print(f"训练集: {len(split['train'])}, 验证集: {len(split['test'])}")
    
    # 6. 加载模型
    print(f"\n4. 加载模型...")
    
    lora_config = {
        "r": cfg.model.lora_r,
        "lora_alpha": cfg.model.lora_alpha,
        "lora_dropout": cfg.model.lora_dropout,
        "target_modules": cfg.model.lora_target_modules,
    }
    
    model = StyleAlignmentModel(
        model_name_or_path=cfg.model.model_name_or_path,
        lora_config=lora_config,
        device="cuda:0",
    )
    
    # 7. Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 8. 训练参数
    print(f"\n5. 配置训练...")
    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_steps=cfg.training.warmup_steps,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        eval_steps=cfg.training.eval_steps,
        save_total_limit=cfg.training.save_total_limit,
        fp16=cfg.training.fp16,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to=["tensorboard"],
        logging_dir=f"{cfg.training.output_dir}/logs",
        ddp_find_unused_parameters=False,
    )
    
    # 9. Trainer
    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=split['train'],
        eval_dataset=split['test'],
        data_collator=data_collator,
    )
    
    # 10. 训练
    print("\n" + "="*60)
    print("开始训练...")
    print("="*60 + "\n")
    
    trainer.train()
    
    # 11. 保存
    print(f"\n保存模型到: {cfg.training.output_dir}")
    model.save_pretrained(cfg.training.output_dir)
    tokenizer.save_pretrained(cfg.training.output_dir)
    
    print("\n✓ 训练完成!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train_config.yaml")
    args = parser.parse_args()
    
    train(args.config)


if __name__ == "__main__":
    main()
