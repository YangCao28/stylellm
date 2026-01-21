"""
主训练脚本 (Main Training Script)
自监督风格对齐训练的完整流程
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import load_dataset

# 导入自定义模块
from config import Config, ModelConfig, TrainingConfig, DataConfig
from config_loader import load_config, print_config
from masking_engine import DynamicMaskingEngine, MaskingCollator
from style_alignment_model import StyleAlignmentModel
from data_processor import WuxiaDataProcessor
from evaluator import WuxiaEvaluator


class StyleAlignmentTrainer(Trainer):
    """自定义Trainer，支持动态掩码和KL散度计算"""
    
    def __init__(self, masking_engine, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.masking_engine = masking_engine
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        重写损失计算，应用动态掩码和KL散度
        """
        # 应用动态掩码
        masked_inputs = self.masking_engine.mask_batch(inputs)
        
        # 前向传播
        outputs = model(
            masked_input_ids=masked_inputs['masked_input_ids'],
            attention_mask=masked_inputs.get('attention_mask'),
            labels=masked_inputs['labels'],
            mask_positions=masked_inputs['mask_positions'],
        )
        
        loss = outputs['loss']
        
        # 记录各项损失
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                'rec_loss': outputs['rec_loss'].item(),
                'kl_loss': outputs['kl_loss'].item(),
                'total_loss': loss.item(),
            })
        
        return (loss, outputs) if return_outputs else loss


def setup_training(config: Config):
    """
    设置训练环境
    
    Args:
        config: 配置对象
        
    Returns:
        model, tokenizer, train_dataset, eval_dataset, masking_engine
    """
    print("="*60)
    print("设置训练环境...")
    print("="*60)
    
    # 1. 加载tokenizer
    print(f"\n1. 加载tokenizer: {config.model.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name_or_path,
        trust_remote_code=True
    )
    
    # 确保有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 添加[MASK] token（如果没有）
    if tokenizer.mask_token is None:
        special_tokens_dict = {'mask_token': '[MASK]'}
        num_added = tokenizer.add_special_tokens(special_tokens_dict)
        print(f"添加了 {num_added} 个特殊token")
    
    # 2. 处理数据
    print(f"\n2. 处理数据...")
    
    # 检查是否需要重新处理数据
    need_process = False
    if not os.path.exists(config.data.processed_data_file):
        need_process = True
        print(f"处理原始数据: {config.data.data_dir}（文件不存在）")
    else:
        # 检查文件是否为空
        if os.path.getsize(config.data.processed_data_file) == 0:
            need_process = True
            print(f"处理原始数据: {config.data.data_dir}（文件为空，重新处理）")
        else:
            print(f"使用已处理的数据: {config.data.processed_data_file}")
    
    if need_process:
        processor = WuxiaDataProcessor(
            tokenizer=tokenizer,
            max_length=config.data.max_length,
            min_length=config.data.min_length,
            stride=config.data.stride,
        )
        
        # 处理所有txt文件
        chunks = processor.process_directory(
            data_dir=config.data.data_dir,
            output_file=config.data.processed_data_file,
            max_files=config.data.max_files,
        )
        
        print(f"处理完成，共 {len(chunks)} 个文本块")
    
    # 3. 加载数据集
    print(f"\n3. 加载数据集...")
    if not os.path.exists(config.data.processed_data_file):
        raise FileNotFoundError(
            f"处理后的数据文件不存在: {config.data.processed_data_file}\n"
            f"请确保 {config.data.data_dir} 目录下有txt文件"
        )
    
    # 检查文件是否为空
    with open(config.data.processed_data_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        if not first_line:
            raise ValueError(
                f"处理后的数据文件为空: {config.data.processed_data_file}\n"
                f"请检查 {config.data.data_dir} 目录下是否有有效的txt文件"
            )
    
    dataset = load_dataset('json', data_files=config.data.processed_data_file, split='train')
    print(f"数据集大小: {len(dataset)}")
    
    # 4. Tokenize数据集
    print(f"\n4. Tokenizing数据集...")
    
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
    
    # 5. 划分训练集和验证集
    print(f"\n5. 划分训练集和验证集...")
    split_dataset = tokenized_dataset.train_test_split(
        test_size=config.data.val_ratio,
        seed=config.training.seed
    )
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(eval_dataset)} 样本")
    
    # 6. 创建掩码引擎
    print(f"\n6. 创建动态掩码引擎...")
    masking_engine = DynamicMaskingEngine(
        tokenizer=tokenizer,
        span_mask_ratio=config.data.span_mask_ratio,
        token_mask_ratio=config.data.token_mask_ratio,
        no_mask_ratio=config.data.no_mask_ratio,
        span_length_range=config.data.span_length_range,
    )
    
    # 7. 加载模型
    print(f"\n7. 加载双模型框架...")
    
    lora_config = {
        "r": config.model.lora_r,
        "lora_alpha": config.model.lora_alpha,
        "lora_dropout": config.model.lora_dropout,
        "target_modules": config.model.lora_target_modules,
    }
    
    model = StyleAlignmentModel(
        model_name=config.model.model_name_or_path,
        kl_beta=config.training.kl_beta,
        use_lora=config.model.use_lora,
        lora_config=lora_config,
    )
    
    # 如果添加了新token，需要调整embedding
    if tokenizer.mask_token and len(tokenizer) > model.policy_model.config.vocab_size:
        model.policy_model.resize_token_embeddings(len(tokenizer))
        print(f"调整词表大小到 {len(tokenizer)}")
    
    print("\n✓ 训练环境设置完成!")
    
    return model, tokenizer, train_dataset, eval_dataset, masking_engine


def train(config: Config):
    """
    执行训练
    
    Args:
        config: 配置对象
    """
    # 设置训练环境
    model, tokenizer, train_dataset, eval_dataset, masking_engine = setup_training(config)
    
    # 创建输出目录
    os.makedirs(config.training.output_dir, exist_ok=True)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        logging_steps=config.training.logging_steps,
        save_strategy=config.training.save_strategy,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        evaluation_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        dataloader_num_workers=config.training.dataloader_num_workers,
        seed=config.training.seed,
        lr_scheduler_type=config.training.lr_scheduler_type,
        remove_unused_columns=config.training.remove_unused_columns,
        report_to=["tensorboard"],
    )
    
    # 数据collator
    data_collator = MaskingCollator(masking_engine, tokenizer)
    
    # 创建Trainer
    trainer = StyleAlignmentTrainer(
        masking_engine=masking_engine,
        model=model,
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
    print("\n保存最终模型...")
    final_model_dir = os.path.join(config.training.output_dir, "final_model")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    print(f"✓ 训练完成! 模型保存在: {final_model_dir}")
    
    return model, tokenizer


def evaluate_model(config: Config, model, tokenizer):
    """
    评估训练后的模型
    
    Args:
        config: 配置对象
        model: 训练后的模型
        tokenizer: Tokenizer
    """
    print("\n" + "="*60)
    print("评估模型...")
    print("="*60)
    
    # 加载参考文本（用于风格评估）
    print("加载参考文本...")
    reference_texts = []
    
    if os.path.exists(config.data.processed_data_file):
        import json
        with open(config.data.processed_data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 100:  # 使用前100个作为参考
                    break
                data = json.loads(line)
                reference_texts.append(data['text'])
    
    # 生成文本样本
    print("\n生成测试样本...")
    test_prompts = [
        "剑光一闪",
        "少年抬头望",
        "江湖之中",
        "他缓缓转身",
        "血光四溅"
    ]
    
    generated_texts = []
    model.policy_model.eval()
    
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.policy_model.device)
        
        outputs = model.generate(
            input_ids=input_ids,
            max_length=config.eval.max_new_tokens,
            temperature=config.eval.temperature,
            top_p=config.eval.top_p,
            top_k=config.eval.top_k,
            do_sample=True,
            num_return_sequences=1,
        )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_texts.append(generated)
        print(f"\n提示: {prompt}")
        print(f"生成: {generated}")
    
    # 风格评估
    print("\n进行风格评估...")
    evaluator = WuxiaEvaluator(
        model=model.policy_model,
        tokenizer=tokenizer,
        reference_texts=reference_texts
    )
    
    results = evaluator.evaluate(
        generated_texts=generated_texts,
        compute_ppl=config.eval.compute_perplexity
    )
    
    evaluator.print_results(results)
    
    # 保存评估结果
    results_file = os.path.join(config.training.output_dir, "evaluation_results.json")
    import json
    with open(results_file, 'w', encoding='utf-8') as f:
        # 转换不可序列化的对象
        serializable_results = {}
        for k, v in results.items():
            if isinstance(v, (int, float, str, list)):
                serializable_results[k] = v
            else:
                serializable_results[k] = str(v)
        
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 评估结果已保存到: {results_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="武侠风格自监督对齐训练")
    parser.add_argument("--config", type=str, default=None, 
                        help="YAML配置文件路径 (例如: train_config.yaml)")
    parser.add_argument("--model_name", type=str, default=None, help="模型名称或路径")
    parser.add_argument("--data_dir", type=str, default=None, help="数据目录")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--kl_beta", type=float, default=None, help="KL散度权重")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--eval_only", action="store_true", help="仅评估模型")
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        print(f"\n从配置文件加载: {args.config}")
        config = load_config(args.config)
    else:
        print("\n使用默认配置 (config.py)")
        config = Config()
    
    # 命令行参数覆盖
    if args.model_name:
        config.model.model_name_or_path = args.model_name
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.kl_beta is not None:
        config.training.kl_beta = args.kl_beta
    if args.epochs:
        config.training.num_train_epochs = args.epochs
    
    # 打印当前配置
    print_config(config)
    
    if args.eval_only:
        # 仅评估
        print("\n仅评估模式")
        tokenizer = AutoTokenizer.from_pretrained(config.training.output_dir)
        model = StyleAlignmentModel(config.training.output_dir, kl_beta=config.training.kl_beta)
        evaluate_model(config, model, tokenizer)
    else:
        # 训练
        model, tokenizer = train(config)
        
        # 训练后评估
        evaluate_model(config, model, tokenizer)


if __name__ == "__main__":
    main()
