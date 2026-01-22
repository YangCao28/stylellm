# ========================================
# 风格转换测试脚本 (Infer)
# 加载训练好的模型，测试“武侠化”改写能力
# ========================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import sys

def generate_wuxia(base_model_path, lora_path, prompt_text):
    print(f"Loading base model: {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, 
        trust_remote_code=True
    )
    
    # 稍微显式处理一下特殊的 padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载 Base Model (bf16, 单卡即可)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # 加载 LoRA
    print(f"Loading LoRA adapter: {lora_path}...")
    try:
        model = PeftModel.from_pretrained(model, lora_path)
    except Exception as e:
        print(f"Error loading LoRA: {e}")
        print("尝试直接使用 Base Model 进行对比...")

    model.eval()

    # 原作提示词构造
    # 我们训练的是 Causal LM (续写)，所以直接把开头给它，让它往下编
    # 或者如果我们想让它“改写”，通常最好用 Instruction Tuning 的格式，
    # 但因为我们这次只有纯文本训练，最好的测试方法是：
    # 给它每句话的开头，看它怎么把这一句“圆”成武侠风；或者直接让它续写。
    
    print("\n" + "="*40)
    print("输入文本:")
    print(prompt_text)
    print("="*40 + "\n")

    # 构造 Prompt：为了引导它进入“武侠模式”，我们可以加个强制的前缀
    # 但既然是 Base Model 微调，它应该直接就能续写武侠
    
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    print("生成中...", end="", flush=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,        # 生成长度
            temperature=0.7,           # 创造性
            top_p=0.9,
            repetition_penalty=1.1,    # 防止复读机
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    print(" Done!\n")

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("="*40)
    print("生成结果 (武侠版白雪公主?):")
    print("="*40)
    print(generated_text)
    print("="*40)

if __name__ == "__main__":
    # 解析命令行参数
    # 用法: python inference_test.py --lora ./output/full_run/checkpoint-xxx
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--lora", type=str, required=True, help="LoRA checkpoint path")
    
    # 默认测试文本
    default_text = "在很久以前，有一个美丽的公主，她的皮肤像雪一样白，加上她出生在白雪飘飘的冬天，因此大家都叫她白雪公主。 白雪公主的妈妈很早就去世了。 她有一个继母，也就是新王后。"
    parser.add_argument("--text", type=str, default=default_text)
    
    args = parser.parse_args()
    
    generate_wuxia(args.base_model, args.lora, args.text)
