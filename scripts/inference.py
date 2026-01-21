"""
推理示例脚本
展示如何使用训练好的武侠风格模型生成文本
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse


def load_model(model_path: str):
    """加载训练好的模型"""
    print(f"加载模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    print("✓ 模型加载完成")
    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 200,
    temperature: float = 0.9,
    top_p: float = 0.9,
    top_k: int = 50,
    num_samples: int = 1,
):
    """生成武侠风格文本"""
    
    # Tokenize输入
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    
    print(f"\n提示词: {prompt}")
    print(f"生成参数: temp={temperature}, top_p={top_p}, top_k={top_k}")
    print("-" * 60)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            num_return_sequences=num_samples,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码输出
    generated_texts = []
    for i, output in enumerate(outputs):
        text = tokenizer.decode(output, skip_special_tokens=True)
        generated_texts.append(text)
        
        print(f"\n样本 {i+1}:")
        print(text)
        print("-" * 60)
    
    return generated_texts


def interactive_mode(model, tokenizer):
    """交互式生成模式"""
    print("\n" + "="*60)
    print("🗡️  武侠风格文本生成 - 交互模式")
    print("="*60)
    print("\n输入提示词，模型将生成武侠风格的续写。")
    print("输入 'quit' 或 'exit' 退出。")
    print("输入 'config' 调整生成参数。")
    
    # 默认参数
    config = {
        'max_length': 200,
        'temperature': 0.9,
        'top_p': 0.9,
        'top_k': 50,
        'num_samples': 1,
    }
    
    while True:
        print("\n" + "-"*60)
        prompt = input("提示词 >>> ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("\n再见！江湖再见！⚔️")
            break
        
        if prompt.lower() == 'config':
            print("\n当前配置:")
            for key, value in config.items():
                print(f"  {key}: {value}")
            
            print("\n修改配置（直接Enter跳过）:")
            for key in config.keys():
                new_value = input(f"  {key} [{config[key]}]: ").strip()
                if new_value:
                    try:
                        if key in ['max_length', 'top_k', 'num_samples']:
                            config[key] = int(new_value)
                        else:
                            config[key] = float(new_value)
                    except ValueError:
                        print(f"    ⚠ 无效值，保持原值")
            
            print("\n✓ 配置已更新")
            continue
        
        if not prompt:
            print("⚠ 提示词不能为空")
            continue
        
        # 生成
        try:
            generate_text(model, tokenizer, prompt, **config)
        except Exception as e:
            print(f"⚠ 生成失败: {e}")


def batch_generate(model, tokenizer, prompts_file: str, output_file: str):
    """批量生成模式"""
    print(f"\n批量生成模式")
    print(f"输入文件: {prompts_file}")
    print(f"输出文件: {output_file}")
    
    # 读取提示词
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    print(f"共 {len(prompts)} 个提示词")
    
    # 生成
    all_results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] 生成中...")
        texts = generate_text(model, tokenizer, prompt, num_samples=1)
        all_results.append({
            'prompt': prompt,
            'generated': texts[0]
        })
    
    # 保存结果
    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 结果已保存到: {output_file}")


def demo_mode(model, tokenizer):
    """演示模式 - 使用预设提示词"""
    print("\n" + "="*60)
    print("🎭 武侠风格文本生成 - 演示模式")
    print("="*60)
    
    demo_prompts = [
        "剑光一闪",
        "少年缓缓抬起头",
        "江湖之中",
        "他突然转身",
        "血染长空",
        "月黑风高之夜",
        "一声长啸",
        "刀剑相交",
    ]
    
    print(f"\n将使用 {len(demo_prompts)} 个预设提示词进行演示：")
    for i, p in enumerate(demo_prompts, 1):
        print(f"  {i}. {p}")
    
    input("\n按Enter开始...")
    
    for prompt in demo_prompts:
        generate_text(
            model, tokenizer, prompt,
            max_length=150,
            temperature=0.9,
            num_samples=1
        )
        input("\n按Enter继续下一个...")


def main():
    parser = argparse.ArgumentParser(description="武侠风格文本生成")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--mode", type=str, default="interactive",
                       choices=["interactive", "batch", "demo"],
                       help="运行模式")
    parser.add_argument("--prompts_file", type=str, help="批量模式：提示词文件")
    parser.add_argument("--output_file", type=str, help="批量模式：输出文件")
    
    args = parser.parse_args()
    
    # 加载模型
    model, tokenizer = load_model(args.model_path)
    
    # 根据模式运行
    if args.mode == "interactive":
        interactive_mode(model, tokenizer)
    elif args.mode == "demo":
        demo_mode(model, tokenizer)
    elif args.mode == "batch":
        if not args.prompts_file or not args.output_file:
            print("⚠ 批量模式需要指定 --prompts_file 和 --output_file")
            return
        batch_generate(model, tokenizer, args.prompts_file, args.output_file)


if __name__ == "__main__":
    # 使用示例
    print("="*60)
    print("武侠风格文本生成 - 推理脚本")
    print("="*60)
    print("\n使用方法:")
    print("\n1. 交互模式:")
    print("   python inference.py --model_path ./output/wuxia_model/final_model")
    print("\n2. 演示模式:")
    print("   python inference.py --model_path ./output/wuxia_model/final_model --mode demo")
    print("\n3. 批量生成:")
    print("   python inference.py --model_path ./output/wuxia_model/final_model --mode batch \\")
    print("       --prompts_file prompts.txt --output_file results.json")
    print("="*60)
    print()
    
    main()
