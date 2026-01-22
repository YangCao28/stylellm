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

    # 构造 Prompt：使用 Few-shot 引导模型进行改写
    # 即使是 Base 模型，通过 LoRA 加持风格，配合 Few-shot 也能实现风格迁移
    few_shot_prompt = (
        "将以下文字改写为古龙武侠小说风格：\n\n"
        "原文：那天早上他吃了一个苹果，心情很好。\n"
        "改写：桌上有一个苹果。他盯着这个苹果已看了很久。忽然间，他拿起苹果咬了一口，嘴边泛起了一丝冷笑。\n\n"
        "原文：外面下雨了，他不想出门。\n"
        "改写：窗外的雨，像是一把把冰冷的刀，无情地刮着大地。这种天气，只有死人才会想出门。\n\n"
        f"原文：{prompt_text}\n"
        "改写："
    )

    print("\n" + "="*40)
    print("构造 Few-shot Prompt (引导改写):")
    print("="*40)
    print(few_shot_prompt)
    print("="*40 + "\n")
    
    inputs = tokenizer(few_shot_prompt, return_tensors="pt").to(model.device)

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
            eos_token_id=tokenizer.eos_token_id,
            stop_strings=["原文：", "改写："], # 防止它自己继续编造新的原文
            tokenizer=tokenizer
        )
    print(" Done!\n")

    # 只截取生成的部分
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = full_output[len(few_shot_prompt):].strip()
    
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
