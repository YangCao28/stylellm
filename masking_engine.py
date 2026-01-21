"""
动态掩码引擎 (Dynamic Masking Engine)
实现混合粒度掩码策略：Random Span 60%, Token-level 30%, No Masking 10%
"""

import random
import torch
from typing import List, Tuple, Dict
import numpy as np


class DynamicMaskingEngine:
    """武侠风格的动态掩码引擎"""
    
    def __init__(
        self,
        tokenizer,
        span_mask_ratio: float = 0.6,
        token_mask_ratio: float = 0.3,
        no_mask_ratio: float = 0.1,
        span_length_range: Tuple[int, int] = (3, 8),
        mask_token_id: int = None,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            span_mask_ratio: Random Span Masking 比例 (默认60%)
            token_mask_ratio: Token-level Masking 比例 (默认30%)
            no_mask_ratio: 不掩码比例 (默认10%)
            span_length_range: Span 长度范围 (默认3-8个token)
            mask_token_id: [MASK] token的ID，如果为None则使用tokenizer的mask_token_id
        """
        self.tokenizer = tokenizer
        self.span_mask_ratio = span_mask_ratio
        self.token_mask_ratio = token_mask_ratio
        self.no_mask_ratio = no_mask_ratio
        self.span_length_range = span_length_range
        
        # 获取[MASK] token ID
        if mask_token_id is None:
            if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
                self.mask_token_id = tokenizer.mask_token_id
            else:
                # 如果tokenizer没有mask_token，使用<unk>或添加特殊token
                self.mask_token_id = tokenizer.unk_token_id
                print(f"警告: tokenizer没有mask_token，使用unk_token_id: {self.mask_token_id}")
        else:
            self.mask_token_id = mask_token_id
        
        # 需要保护的特殊token（不应该被掩码）
        self.special_token_ids = set([
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
            tokenizer.bos_token_id,
        ])
        if hasattr(tokenizer, 'sep_token_id'):
            self.special_token_ids.add(tokenizer.sep_token_id)
        self.special_token_ids.discard(None)
    
    def apply_masking(
        self,
        input_ids: torch.Tensor,
        return_labels: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        对输入序列应用动态掩码
        
        Args:
            input_ids: 输入token序列 [batch_size, seq_len]
            return_labels: 是否返回标签（用于计算损失）
            
        Returns:
            masked_input_ids: 掩码后的输入 [batch_size, seq_len]
            labels: 原始token标签，未掩码位置为-100 [batch_size, seq_len]
            mask_positions: 掩码位置的布尔张量 [batch_size, seq_len]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 创建副本
        masked_input_ids = input_ids.clone()
        labels = torch.full_like(input_ids, -100)  # -100会被忽略
        mask_positions = torch.zeros_like(input_ids, dtype=torch.bool)
        
        # 对每个样本单独处理
        for i in range(batch_size):
            # 获取可掩码的位置（排除特殊token）
            valid_positions = []
            for j in range(seq_len):
                if input_ids[i, j].item() not in self.special_token_ids:
                    valid_positions.append(j)
            
            if len(valid_positions) == 0:
                continue
            
            # 决定掩码策略
            strategy = random.choices(
                ['span', 'token', 'none'],
                weights=[self.span_mask_ratio, self.token_mask_ratio, self.no_mask_ratio],
                k=1
            )[0]
            
            if strategy == 'none':
                # 不掩码，作为anchor
                continue
            
            elif strategy == 'span':
                # Random Span Masking
                positions_to_mask = self._span_masking(valid_positions)
            
            else:  # strategy == 'token'
                # Token-level Masking
                positions_to_mask = self._token_masking(valid_positions)
            
            # 应用掩码
            for pos in positions_to_mask:
                masked_input_ids[i, pos] = self.mask_token_id
                labels[i, pos] = input_ids[i, pos]
                mask_positions[i, pos] = True
        
        result = {
            'masked_input_ids': masked_input_ids,
            'mask_positions': mask_positions
        }
        
        if return_labels:
            result['labels'] = labels
        
        return result
    
    def _span_masking(self, valid_positions: List[int]) -> List[int]:
        """Random Span Masking: 随机连续遮盖3-8个token"""
        if len(valid_positions) < self.span_length_range[0]:
            return []
        
        # 随机选择span长度
        max_span_len = min(self.span_length_range[1], len(valid_positions))
        span_length = random.randint(self.span_length_range[0], max_span_len)
        
        # 随机选择起始位置
        max_start = len(valid_positions) - span_length
        if max_start < 0:
            span_length = len(valid_positions)
            max_start = 0
        
        start_idx = random.randint(0, max_start)
        
        # 返回连续的位置
        return valid_positions[start_idx:start_idx + span_length]
    
    def _token_masking(self, valid_positions: List[int]) -> List[int]:
        """Token-level Masking: 随机遮盖15-25%的单个token"""
        num_to_mask = max(1, int(len(valid_positions) * random.uniform(0.15, 0.25)))
        return random.sample(valid_positions, num_to_mask)
    
    def mask_batch(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        对整个batch应用掩码（用于DataLoader）
        
        Args:
            batch: 包含'input_ids'的字典
            
        Returns:
            添加了'masked_input_ids', 'labels', 'mask_positions'的batch
        """
        input_ids = batch['input_ids']
        mask_result = self.apply_masking(input_ids)
        
        # 更新batch
        batch.update(mask_result)
        return batch


class MaskingCollator:
    """用于DataLoader的动态掩码collator"""
    
    def __init__(self, masking_engine: DynamicMaskingEngine, tokenizer):
        self.masking_engine = masking_engine
        self.tokenizer = tokenizer
    
    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        将样本列表转换为batch并应用掩码
        
        Args:
            examples: 样本列表，每个样本是包含'text'或'input_ids'的字典
            
        Returns:
            处理好的batch
        """
        # 如果样本包含text，先tokenize
        if 'text' in examples[0]:
            texts = [ex['text'] for ex in examples]
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )
        else:
            # 假设已经tokenized
            input_ids = [torch.tensor(ex['input_ids']) for ex in examples]
            encoded = {'input_ids': torch.stack(input_ids)}
        
        # 应用动态掩码
        masked_batch = self.masking_engine.mask_batch(encoded)
        
        return masked_batch


# 测试代码
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # 加载tokenizer（这里用一个示例，实际使用时换成你的模型）
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 添加[MASK] token（如果没有）
    if tokenizer.mask_token is None:
        special_tokens_dict = {'mask_token': '[MASK]'}
        tokenizer.add_special_tokens(special_tokens_dict)
    
    # 创建掩码引擎
    masking_engine = DynamicMaskingEngine(tokenizer)
    
    # 测试文本（武侠风格）
    test_texts = [
        "剑光一闪，那人已然倒地，血溅三尺，惨叫之声响彻云霄。",
        "他缓缓抬起头，眼中尽是悲凉之色，轻声道：江湖事，江湖了。"
    ]
    
    # Tokenize
    encoded = tokenizer(
        test_texts,
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=64
    )
    
    print("原始input_ids shape:", encoded['input_ids'].shape)
    print("原始文本1:", test_texts[0])
    print("原始tokens1:", tokenizer.convert_ids_to_tokens(encoded['input_ids'][0]))
    
    # 应用掩码
    masked_result = masking_engine.apply_masking(encoded['input_ids'])
    
    print("\n掩码后tokens1:", tokenizer.convert_ids_to_tokens(masked_result['masked_input_ids'][0]))
    print("掩码位置1:", masked_result['mask_positions'][0].tolist())
    print("标签1 (仅显示非-100):", [
        (i, tokenizer.decode([id])) 
        for i, id in enumerate(masked_result['labels'][0].tolist()) 
        if id != -100
    ])
    
    print("\n多次测试掩码策略分布：")
    strategies = {'span': 0, 'token': 0, 'none': 0}
    for _ in range(100):
        result = masking_engine.apply_masking(encoded['input_ids'][:1])
        num_masked = result['mask_positions'][0].sum().item()
        if num_masked == 0:
            strategies['none'] += 1
        elif num_masked >= 3:
            strategies['span'] += 1
        else:
            strategies['token'] += 1
    
    print(f"Span Masking: {strategies['span']}%")
    print(f"Token Masking: {strategies['token']}%")
    print(f"No Masking: {strategies['none']}%")
