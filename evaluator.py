"""
评估模块 (Evaluation Module)
实现困惑度计算、N-gram重合度、虚词分布分析等无对评估指标
"""

import torch
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import math


class WuxiaEvaluator:
    """武侠风格自动化评估器"""
    
    # 武侠虚词
    WUXIA_FUNCTION_WORDS = [
        '却', '竟', '便', '乃', '焉', '耳', '矣', '哉', '者', '也',
        '之', '而', '且', '若', '则', '唯', '尚', '犹', '岂', '其',
        '然', '故', '是以', '何', '安', '但', '惟', '遂', '既'
    ]
    
    def __init__(
        self,
        model=None,
        tokenizer=None,
        reference_texts: List[str] = None
    ):
        """
        Args:
            model: 语言模型（用于计算困惑度）
            tokenizer: Tokenizer
            reference_texts: 武侠原著参考文本列表（用于计算风格距离）
        """
        self.model = model
        self.tokenizer = tokenizer
        self.reference_texts = reference_texts or []
        
        # 预计算参考文本的统计信息
        if self.reference_texts:
            self._compute_reference_stats()
    
    def _compute_reference_stats(self):
        """预计算参考文本的N-gram和虚词分布"""
        print("Computing reference corpus statistics...")
        
        # N-gram统计
        self.reference_ngrams = {
            1: Counter(),
            2: Counter(),
            3: Counter(),
            4: Counter(),
        }
        
        # 虚词统计
        self.reference_function_word_freq = Counter()
        
        for text in self.reference_texts:
            # 更新N-gram
            for n in [1, 2, 3, 4]:
                ngrams = self._extract_ngrams(text, n)
                self.reference_ngrams[n].update(ngrams)
            
            # 更新虚词频率
            for word in self.WUXIA_FUNCTION_WORDS:
                count = text.count(word)
                if count > 0:
                    self.reference_function_word_freq[word] += count
        
        # 归一化
        for n in self.reference_ngrams:
            total = sum(self.reference_ngrams[n].values())
            if total > 0:
                for key in self.reference_ngrams[n]:
                    self.reference_ngrams[n][key] /= total
        
        total_fw = sum(self.reference_function_word_freq.values())
        if total_fw > 0:
            for key in self.reference_function_word_freq:
                self.reference_function_word_freq[key] /= total_fw
        
        print(f"Reference stats computed from {len(self.reference_texts)} texts")
    
    def compute_perplexity(
        self,
        texts: List[str],
        batch_size: int = 4
    ) -> float:
        """
        计算困惑度 (Perplexity)
        
        Args:
            texts: 待评估的文本列表
            batch_size: 批处理大小
            
        Returns:
            平均困惑度
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer required for perplexity computation")
        
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                encodings = self.tokenizer(
                    batch_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                input_ids = encodings['input_ids'].to(self.model.device)
                attention_mask = encodings['attention_mask'].to(self.model.device)
                
                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                # 累计损失
                total_loss += outputs.loss.item() * input_ids.size(0)
                total_tokens += input_ids.size(0)
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return perplexity
    
    def _extract_ngrams(self, text: str, n: int) -> List[str]:
        """提取N-gram"""
        # 简单按字符提取（中文）
        if n == 1:
            return list(text)
        
        ngrams = []
        for i in range(len(text) - n + 1):
            ngrams.append(text[i:i+n])
        return ngrams
    
    def compute_ngram_overlap(
        self,
        generated_texts: List[str],
        n: int = 2
    ) -> float:
        """
        计算N-gram重合度
        
        Args:
            generated_texts: 生成的文本列表
            n: N-gram的n
            
        Returns:
            与参考文本的N-gram重合率
        """
        if not self.reference_texts:
            raise ValueError("Reference texts required for N-gram overlap computation")
        
        # 提取生成文本的N-gram
        generated_ngrams = Counter()
        for text in generated_texts:
            ngrams = self._extract_ngrams(text, n)
            generated_ngrams.update(ngrams)
        
        # 归一化
        total = sum(generated_ngrams.values())
        if total == 0:
            return 0.0
        
        for key in generated_ngrams:
            generated_ngrams[key] /= total
        
        # 计算重合度（使用余弦相似度）
        reference_ngrams = self.reference_ngrams[n]
        
        # 获取所有N-gram
        all_ngrams = set(generated_ngrams.keys()) | set(reference_ngrams.keys())
        
        # 计算余弦相似度
        dot_product = sum(
            generated_ngrams.get(ng, 0) * reference_ngrams.get(ng, 0)
            for ng in all_ngrams
        )
        
        gen_norm = math.sqrt(sum(v**2 for v in generated_ngrams.values()))
        ref_norm = math.sqrt(sum(v**2 for v in reference_ngrams.values()))
        
        if gen_norm == 0 or ref_norm == 0:
            return 0.0
        
        similarity = dot_product / (gen_norm * ref_norm)
        
        return similarity
    
    def compute_function_word_distribution(
        self,
        texts: List[str]
    ) -> Dict[str, float]:
        """
        计算虚词分布并与参考文本比较
        
        Args:
            texts: 待评估的文本列表
            
        Returns:
            包含各项指标的字典
        """
        # 统计生成文本中的虚词
        generated_fw_freq = Counter()
        total_chars = 0
        
        for text in texts:
            total_chars += len(text)
            for word in self.WUXIA_FUNCTION_WORDS:
                count = text.count(word)
                if count > 0:
                    generated_fw_freq[word] += count
        
        # 归一化
        total_fw = sum(generated_fw_freq.values())
        if total_fw > 0:
            for key in generated_fw_freq:
                generated_fw_freq[key] /= total_fw
        
        # 虚词密度
        fw_density = total_fw / total_chars if total_chars > 0 else 0
        
        # 与参考文本的分布差异（KL散度）
        if self.reference_texts and self.reference_function_word_freq:
            kl_div = self._compute_kl_divergence(
                generated_fw_freq,
                self.reference_function_word_freq
            )
        else:
            kl_div = None
        
        return {
            'function_word_density': fw_density,
            'function_word_diversity': len(generated_fw_freq),
            'distribution_kl_divergence': kl_div,
            'top_function_words': generated_fw_freq.most_common(10),
        }
    
    def _compute_kl_divergence(
        self,
        p: Counter,
        q: Counter
    ) -> float:
        """计算KL散度 KL(P||Q)"""
        all_keys = set(p.keys()) | set(q.keys())
        
        kl = 0.0
        epsilon = 1e-10  # 平滑
        
        for key in all_keys:
            p_val = p.get(key, 0) + epsilon
            q_val = q.get(key, 0) + epsilon
            kl += p_val * math.log(p_val / q_val)
        
        return kl
    
    def evaluate(
        self,
        generated_texts: List[str],
        compute_ppl: bool = True
    ) -> Dict[str, any]:
        """
        综合评估
        
        Args:
            generated_texts: 生成的文本列表
            compute_ppl: 是否计算困惑度（需要模型）
            
        Returns:
            包含所有评估指标的字典
        """
        results = {}
        
        # 困惑度
        if compute_ppl and self.model is not None:
            print("Computing perplexity...")
            results['perplexity'] = self.compute_perplexity(generated_texts)
        
        # N-gram重合度
        if self.reference_texts:
            print("Computing N-gram overlap...")
            for n in [2, 3, 4]:
                overlap = self.compute_ngram_overlap(generated_texts, n)
                results[f'{n}gram_overlap'] = overlap
        
        # 虚词分布
        print("Computing function word distribution...")
        fw_stats = self.compute_function_word_distribution(generated_texts)
        results.update(fw_stats)
        
        # 基础统计
        results['num_texts'] = len(generated_texts)
        results['avg_length'] = np.mean([len(t) for t in generated_texts])
        
        return results
    
    def print_results(self, results: Dict[str, any]):
        """打印评估结果"""
        print("\n" + "="*60)
        print("武侠风格评估结果 (Wuxia Style Evaluation Results)")
        print("="*60)
        
        if 'perplexity' in results:
            print(f"困惑度 (Perplexity): {results['perplexity']:.2f}")
        
        print(f"\nN-gram重合度 (N-gram Overlap):")
        for n in [2, 3, 4]:
            key = f'{n}gram_overlap'
            if key in results:
                print(f"  {n}-gram: {results[key]:.4f}")
        
        print(f"\n虚词分析 (Function Words):")
        print(f"  密度: {results.get('function_word_density', 0):.4f}")
        print(f"  多样性: {results.get('function_word_diversity', 0)}")
        
        if results.get('distribution_kl_divergence') is not None:
            print(f"  分布差异 (KL): {results['distribution_kl_divergence']:.4f}")
        
        if 'top_function_words' in results:
            print(f"\n  高频虚词:")
            for word, freq in results['top_function_words'][:5]:
                print(f"    {word}: {freq:.4f}")
        
        print(f"\n基础统计:")
        print(f"  样本数: {results.get('num_texts', 0)}")
        print(f"  平均长度: {results.get('avg_length', 0):.1f} 字符")
        
        print("="*60 + "\n")


# 测试代码
if __name__ == "__main__":
    print("Testing Evaluation Module...")
    
    # 模拟武侠参考文本
    reference_texts = [
        "剑光一闪，那人已然倒地。他竟是江湖第一高手，却败在此处。",
        "少年缓缓抬起头，眼中尽是悲凉之色。江湖事，江湖了，何必再问。",
        "刀光剑影之中，他们却在谈笑风生。这便是武林高手的风范。",
    ]
    
    # 模拟生成文本
    generated_texts = [
        "剑气纵横，那人竟然败北。江湖之大，无奇不有。",
        "他缓缓而来，眼中却带着笑意。此战，他已然胜券在握。",
    ]
    
    # 创建评估器（不使用模型，只测试风格指标）
    evaluator = WuxiaEvaluator(reference_texts=reference_texts)
    
    # 评估
    results = evaluator.evaluate(generated_texts, compute_ppl=False)
    evaluator.print_results(results)
    
    # 测试N-gram提取
    print("\nTesting N-gram extraction:")
    test_text = "剑光一闪"
    for n in [1, 2, 3]:
        ngrams = evaluator._extract_ngrams(test_text, n)
        print(f"{n}-grams: {ngrams}")
    
    print("\n✓ All tests completed!")
