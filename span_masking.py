"""
动态Span Masking - 武侠风格专用
随机遮盖连续片段，学习成语、招式、意境描写
"""

import torch
import random
from typing import Dict, List


class SpanMaskingCollator:
    """
    数据collator，动态生成span mask
    
    Args:
        tokenizer: 分词器
        mask_ratio: 总体掩码比例 (15%)
        span_ratio: span掩码占比 (50%)
        span_length: span长度范围 (3-5)
    """
    
    def __init__(
        self,
        tokenizer,
        mask_ratio: float = 0.15,
        span_ratio: float = 0.5,
        span_length: tuple = (3, 5),
        mlm_probability: float = 0.15,
    ):
        self.tokenizer = tokenizer
        self.mask_ratio = mask_ratio
        self.span_ratio = span_ratio
        self.span_length = span_length
        self.mlm_probability = mlm_probability
        
        # 特殊token不能被mask
        self.special_tokens = set([
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
            tokenizer.bos_token_id,
        ])
    
    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        批次处理
        
        Args:
            examples: [{'input_ids': [...]}, ...]
            
        Returns:
            batch: {'input_ids', 'attention_mask', 'labels'}
        """
        batch_size = len(examples)
        
        # 提取input_ids
        input_ids_list = [ex['input_ids'] for ex in examples]
        
        # Padding
        max_len = max(len(ids) for ids in input_ids_list)
        
        input_ids = torch.full((batch_size, max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        
        for i, ids in enumerate(input_ids_list):
            seq_len = len(ids)
            input_ids[i, :seq_len] = torch.tensor(ids)
            attention_mask[i, :seq_len] = 1
            
            # 动态生成mask
            masked_ids, label_ids = self._mask_sequence(ids)
            input_ids[i, :seq_len] = torch.tensor(masked_ids)
            labels[i, :seq_len] = torch.tensor(label_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
    
    def _mask_sequence(self, input_ids: List[int]) -> tuple:
        """
        对单个序列进行span masking
        
        策略：
        - 15% token被选中
        - 其中50%使用span mask (连续3-5个token)
        - 其中50%使用单token mask
        """
        seq_len = len(input_ids)
        masked_ids = list(input_ids)
        labels = [-100] * seq_len  # 默认不计算loss
        
        # 找出可以mask的位置（排除特殊token）
        maskable_positions = [
            i for i in range(seq_len) 
            if input_ids[i] not in self.special_tokens
        ]
        
        if len(maskable_positions) == 0:
            return masked_ids, labels
        
        # 计算要mask的token总数
        num_to_mask = max(1, int(len(maskable_positions) * self.mask_ratio))
        
        # 50% span mask, 50% single token mask
        num_span_mask = int(num_to_mask * self.span_ratio)
        num_single_mask = num_to_mask - num_span_mask
        
        masked_positions = set()
        
        # 1. Span masking
        span_attempts = 0
        while len(masked_positions) < num_span_mask and span_attempts < 100:
            span_attempts += 1
            
            # 随机选择span起点
            if len(maskable_positions) == 0:
                break
            
            start_idx = random.choice(maskable_positions)
            span_len = random.randint(*self.span_length)
            
            # 检查span是否有效
            span_positions = []
            for offset in range(span_len):
                pos = start_idx + offset
                if pos < seq_len and pos in maskable_positions and pos not in masked_positions:
                    span_positions.append(pos)
                else:
                    break
            
            # 添加span
            if len(span_positions) >= 2:  # 至少2个token才算span
                for pos in span_positions:
                    masked_positions.add(pos)
                    if len(masked_positions) >= num_span_mask:
                        break
        
        # 2. Single token masking
        remaining_positions = [p for p in maskable_positions if p not in masked_positions]
        random.shuffle(remaining_positions)
        
        for pos in remaining_positions[:num_single_mask]:
            masked_positions.add(pos)
        
        # 3. 应用mask
        for pos in masked_positions:
            labels[pos] = input_ids[pos]  # 保存原始token用于计算loss
            
            # 80% [MASK], 10% 随机, 10% 保持不变
            rand = random.random()
            if rand < 0.8:
                masked_ids[pos] = self.tokenizer.mask_token_id
            elif rand < 0.9:
                masked_ids[pos] = random.randint(0, self.tokenizer.vocab_size - 1)
            # else: 保持原样
        
        return masked_ids, labels
