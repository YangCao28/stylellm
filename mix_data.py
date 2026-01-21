"""
数据混合脚本 - 防止模型"降智"
自动在武侠数据中混入5-10%通用文本
"""

import os
import json
from pathlib import Path
from typing import List, Dict


def load_wuxia_texts(data_dir: str) -> List[Dict]:
    """加载武侠小说文本"""
    texts = []
    
    for txt_file in Path(data_dir).rglob("*.txt"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 分段
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            for para in paragraphs:
                if len(para) >= 50:
                    texts.append({
                        'text': para,
                        'source': 'wuxia',
                        'file': str(txt_file.name)
                    })
        except Exception as e:
            print(f"跳过 {txt_file}: {e}")
    
    return texts


def load_general_texts(general_file: str) -> List[Dict]:
    """
    加载通用文本（防止降智）
    
    格式: 每行一个段落，或JSONL格式
    """
    texts = []
    
    if not os.path.exists(general_file):
        print(f"未找到通用数据: {general_file}")
        print("将只使用武侠数据训练")
        return texts
    
    try:
        with open(general_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 尝试JSON格式
                try:
                    data = json.loads(line)
                    text = data.get('text', '')
                except:
                    text = line
                
                if len(text) >= 50:
                    texts.append({
                        'text': text,
                        'source': 'general',
                        'file': 'general'
                    })
    except Exception as e:
        print(f"加载通用数据失败: {e}")
    
    return texts


def mix_datasets(wuxia_texts: List[Dict], general_texts: List[Dict], 
                 general_ratio: float = 0.1) -> List[Dict]:
    """
    混合数据集
    
    Args:
        wuxia_texts: 武侠文本
        general_texts: 通用文本
        general_ratio: 通用文本比例 (0.05-0.1)
        
    Returns:
        混合后的数据集
    """
    import random
    
    if len(general_texts) == 0:
        print("没有通用数据，使用纯武侠数据")
        return wuxia_texts
    
    # 计算需要的通用数据量
    total_wuxia = len(wuxia_texts)
    num_general = int(total_wuxia * general_ratio / (1 - general_ratio))
    
    # 采样通用数据（如果不够就重复）
    if num_general <= len(general_texts):
        sampled_general = random.sample(general_texts, num_general)
    else:
        # 重复采样
        sampled_general = []
        while len(sampled_general) < num_general:
            sampled_general.extend(general_texts)
        sampled_general = sampled_general[:num_general]
    
    # 混合并打乱
    mixed = wuxia_texts + sampled_general
    random.shuffle(mixed)
    
    print(f"\n数据混合完成:")
    print(f"  武侠文本: {len(wuxia_texts)} ({len(wuxia_texts)/len(mixed)*100:.1f}%)")
    print(f"  通用文本: {len(sampled_general)} ({len(sampled_general)/len(mixed)*100:.1f}%)")
    print(f"  总计: {len(mixed)}")
    
    return mixed


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="混合武侠+通用数据")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="武侠小说目录")
    parser.add_argument("--general_file", type=str, default="./data/general.txt",
                        help="通用文本文件")
    parser.add_argument("--output", type=str, default="./data/mixed_data.jsonl",
                        help="输出文件")
    parser.add_argument("--general_ratio", type=float, default=0.1,
                        help="通用数据比例 (0.05-0.1)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("数据混合 - 防止模型降智")
    print("="*60)
    
    # 1. 加载武侠数据
    print(f"\n1. 加载武侠数据: {args.data_dir}")
    wuxia_texts = load_wuxia_texts(args.data_dir)
    print(f"✓ 武侠文本: {len(wuxia_texts)} 段")
    
    # 2. 加载通用数据
    print(f"\n2. 加载通用数据: {args.general_file}")
    general_texts = load_general_texts(args.general_file)
    print(f"✓ 通用文本: {len(general_texts)} 段")
    
    # 3. 混合
    print(f"\n3. 混合数据 (通用比例: {args.general_ratio*100:.0f}%)")
    mixed_texts = mix_datasets(wuxia_texts, general_texts, args.general_ratio)
    
    # 4. 保存
    print(f"\n4. 保存到: {args.output}")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        for item in mixed_texts:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n✓ 完成！混合数据已保存")


if __name__ == "__main__":
    main()
