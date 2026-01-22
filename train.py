"""
极简双卡DDP训练 - 纯掩码还原
"""

import os
import argparse
import yaml
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from style_alignment_model import StyleAlignmentModel
from span_masking import SpanMaskingCollator


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
    """处理文本文件 - 基于Token的滑动窗口 (保证100%覆盖)"""
    print(f"处理 {data_dir} 中的txt文件 (Token滑动窗口)...")
    
    import json
    from pathlib import Path
    
    texts = []
    
    # 获取所有文件
    path_obj = Path(data_dir)
    pass_1 = list(path_obj.rglob("*.txt"))
    pass_2 = list(path_obj.rglob("*.TXT"))
    files = sorted(list(set(pass_1 + pass_2)))
    print(f"找到 {len(files)} 个文件")
    
    # 策略: 按 Token 切分
    # 留一点余量(buffer)给特殊token，防止二次tokenize时溢出
    chunk_size = max_length - 4 
    stride = int(chunk_size * 0.8) # 20% 重叠
    print(f"Token切分策略: Chunk={chunk_size}, Stride={stride}")

    total_tokens = 0
    total_files = len(files)
    
    for idx, txt_file in enumerate(files):
        try:
            # 打印进度
            if (idx + 1) % 10 == 0 or idx == 0:
                 print(f"[{idx + 1}/{total_files}] Processing: {txt_file.name}")
                 
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
            
            if not content:
                print(f"  -> Skipping empty file: {txt_file.name}")
                continue
                
            # 1. 先全部转为 Token ID (不加特殊token，纯内容)
            # 为了防止 token indices sequence length warning/error，
            # 我们不使用 tokenizer.encode 处理整个大文件，而是手动分块或者设置 truncation=False (但encode默认不truncate)
            # 不过 tokenizer.encode 对超长文本可能会报 warning。
            # 我们可以使用 tokenizer.__call__ 并且 verbose=False 来抑制警告，或者直接忽略。
            # Qwen 的 tokenizer 最大长度很大，但几十万字的文本直接 encode 确实可能触发警告。
            
            # 这里的 encode 只是为了拿 ID 列表用于切分，不涉及模型 Forward，所以长度超限没关系。
            # 我们通过 disable tracer warning 或者 simply ignore it.
            # 更好的做法是：分块读取并 encode，然后拼起来。
            
            chunks = []
            chunk_size_char = 100000 # 每次读10万字符 encode 一次，避免一次性 encode 太长
            for i in range(0, len(content), chunk_size_char):
                substring = content[i : i + chunk_size_char]
                ids = tokenizer.encode(substring, add_special_tokens=False)
                chunks.extend(ids)
            
            input_ids = chunks
            total_tokens += len(input_ids)
            
            # 2. Token 级滑动窗口
            for i in range(0, len(input_ids), stride):
                chunk_ids = input_ids[i : i + chunk_size]
                
                # 只有当它是该文件的唯一片段，或者长度足够时才保留
                if len(chunk_ids) < min_length:
                    if i == 0: 
                        pass # 文件太短也保留
                    else:
                        continue # 只有尾部丢弃
                
                # 3. 转回文本保存
                # decode 会还原成文本，虽然可能会丢失极少量的空格信息，但对语义无影响
                segment_text = tokenizer.decode(chunk_ids)
                
                texts.append({
                    'text': segment_text,
                    'source': str(txt_file.name)
                })
                
        except Exception as e:
            print(f"跳过 {txt_file}: {e}")
    
    # 保存为jsonl
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in texts:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✓ 处理完成: {len(texts)} 个样本, 总Token数(估): {total_tokens}")
    return len(texts)


def train(config_file):
    """训练主函数"""
    
    # 0. DDP 初始化检测
    if "LOCAL_RANK" in os.environ:
        try:
            torch.distributed.init_process_group(backend="nccl")
        except:
            pass # 可能已经被初始化
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        is_ddp = True
        print(f"[Init] DDP Enabled: Rank {local_rank}/{world_size}, Device {device}")
    else:
        local_rank = 0
        world_size = 1
        device = "cuda:0"
        is_ddp = False
        print(f"[Init] Single GPU Mode: Device {device}")

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
    
    # 添加[MASK] token
    if tokenizer.mask_token is None:
        special_tokens = {'mask_token': '[MASK]'}
        tokenizer.add_special_tokens(special_tokens)
        print(f"添加 [MASK] token")
    
    # 2. 处理数据 (仅Rank 0进行)
    print(f"\n2. 准备数据...")
    
    should_process = not os.path.exists(cfg.data.processed_data_file) or \
                     os.path.getsize(cfg.data.processed_data_file) == 0
    
    if local_rank == 0 and should_process:
        process_text_files(
            cfg.data.data_dir,
            cfg.data.processed_data_file,
            tokenizer,
            cfg.data.max_length,
            cfg.data.min_length
        )
    
    # 等待 Rank 0 处理完成
    if is_ddp:
        torch.distributed.barrier()
        
    if local_rank != 0 and should_process:
         print(f"Rank {local_rank} 等待数据处理完成...")

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
        device=device,
    )
    
    # 如果添加了新token，调整embedding
    if len(tokenizer) > model.model.config.vocab_size:
        model.model.resize_token_embeddings(len(tokenizer))
        print(f"调整词表: {model.model.config.vocab_size} -> {len(tokenizer)}")
    
    # 确保LoRA参数可训练
    for name, param in model.model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True
    
    # 7. Data Collator (切换为 Span Masking)
    from span_masking import SpanMaskingCollator
    data_collator = SpanMaskingCollator(
        tokenizer=tokenizer,
        mask_ratio=0.15,      # 15% 的内容被遮盖
        span_ratio=0.5,       # 其中 50% 是连续片段
        span_length=(3, 8),   # 片段长度 3-8 个 token (涵盖成语/短句)
    )
    print("使用 Span Masking 训练 (武侠风格强化)")
    
    # 8. 训练参数 (DDP优化)
    print(f"\n5. 配置训练...")
    
    # (此时 world_size 已经在开头获取正确)
    print(f"训练模式: {'DDP' if is_ddp else 'Single GPU'}")
    print(f"设备数: {world_size}")
    
    # 强制禁用缓存 (训练时必须禁用)
    model.model.config.use_cache = False
    
    # 显式控制梯度检查点
    if cfg.training.gradient_checkpointing:
        print("启用梯度检查点 (Gradient Checkpointing)")
        model.model.gradient_checkpointing_enable()
    else:
        print("禁用梯度检查点 (Gradient Checkpointing)")
        if hasattr(model.model, "gradient_checkpointing_disable"):
            model.model.gradient_checkpointing_disable()
        model.model.config.gradient_checkpointing = False

    effective_batch_size = cfg.training.per_device_train_batch_size * cfg.training.gradient_accumulation_steps * world_size
    print(f"有效Batch Size: {effective_batch_size}")
    
    # 强制将workers设为1，在DDP环境下更稳定，且避免OOM
    dataloader_num_workers = 1
    print(f"Dataloader Workers: {dataloader_num_workers}")

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
        bf16=getattr(cfg.training, 'bf16', False),
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        logging_dir=f"{cfg.training.output_dir}/logs",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=dataloader_num_workers, # 显式设置
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
    
    # 解析命令行参数中的 resume_from_checkpoint
    import sys
    resume_from_checkpoint = False
    if "--resume_from_checkpoint" in sys.argv:
        try:
            resume_idx = sys.argv.index("--resume_from_checkpoint")
            val = sys.argv[resume_idx + 1]
            if val.lower() == "true":
                resume_from_checkpoint = True
            elif os.path.isdir(val):
                resume_from_checkpoint = val
        except:
             resume_from_checkpoint = True
             
    if resume_from_checkpoint:
        print(f"!!! 接续训练 (Resume): {resume_from_checkpoint} !!!")
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
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
