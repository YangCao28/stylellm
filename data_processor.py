"""
数据处理管道 (Data Processing Pipeline)
实现武侠语料清洗、文档切分、动态掩码应用
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
import random


class WuxiaDataProcessor:
    """武侠语料数据处理器"""
    
    # 需要过滤的现代词汇
    MODERN_KEYWORDS = [
        '手机', '电脑', '互联网', '网络', '电话', '汽车', '飞机', '火车',
        '电视', '科学', '逻辑', '系统', '程序', '软件', '硬件', '数据库',
        '手表', '眼镜', '西装', '领带', '总统', '经理', '公司', '企业'
    ]
    
    # 武侠常见虚词（用于风格评估）
    WUXIA_FUNCTION_WORDS = [
        '却', '竟', '便', '乃', '焉', '耳', '矣', '哉', '者', '也',
        '之', '而', '且', '若', '则', '唯', '尚', '犹', '岂', '其'
    ]
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        min_length: int = 64,
        stride: int = 256,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            max_length: 最大序列长度
            min_length: 最小序列长度
            stride: 滑动窗口步长
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        self.stride = stride
    
    def clean_text(self, text: str) -> Optional[str]:
        """
        清洗文本：去除现代词汇、特殊字符等
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本，如果不符合要求返回None
        """
        if not text or len(text.strip()) < 20:
            return None
        
        # 去除空白字符
        text = text.strip()
        
        # 去除标题行和页眉页脚（通常包含"标题"、"<<"等）
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # 跳过太短的行、标题行
            if len(line) < 3:
                continue
            if any(kw in line for kw in ['标题', '<<', '>>', '――', '===', '---', '第一章', '第二章', '第三章']):
                continue
            cleaned_lines.append(line)
        
        if not cleaned_lines:
            return None
        
        text = ' '.join(cleaned_lines)
        
        # 去除多余的空格和换行
        text = re.sub(r'\s+', ' ', text)
        
        # 去除一些明显的垃圾字符，但保留大部分标点
        text = re.sub(r'[�\x00-\x1f\x7f-\x9f]', '', text)
        
        # 去除重复的标点
        text = re.sub(r'([，。！？；：])\1+', r'\1', text)
        
        return text if len(text) >= 20 else None
    
    def split_into_chunks(
        self,
        text: str,
        preserve_sentences: bool = True
    ) -> List[str]:
        """
        将长文本切分成固定长度的块
        
        Args:
            text: 输入文本
            preserve_sentences: 是否保持句子完整性
            
        Returns:
            文本块列表
        """
        if preserve_sentences:
            # 按句子分割
            sentences = re.split(r'([。！？])', text)
            sentences = [''.join(sentences[i:i+2]) for i in range(0, len(sentences)-1, 2)]
            
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                # 估算token数（中文约等于字符数）
                if len(current_chunk) + len(sentence) <= self.max_length:
                    current_chunk += sentence
                else:
                    if len(current_chunk) >= self.min_length:
                        chunks.append(current_chunk)
                    current_chunk = sentence
            
            if len(current_chunk) >= self.min_length:
                chunks.append(current_chunk)
        
        else:
            # 简单滑动窗口切分
            chunks = []
            for i in range(0, len(text), self.stride):
                chunk = text[i:i + self.max_length]
                if len(chunk) >= self.min_length:
                    chunks.append(chunk)
        
        return chunks
    
    def process_file(self, file_path: str) -> List[str]:
        """
        处理单个txt文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            清洗并切分后的文本块列表
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []
        
        # 清洗文本
        cleaned = self.clean_text(content)
        if not cleaned:
            print(f"  ⚠️ 清洗后内容为空: {Path(file_path).name}")
            return []
        
        # 切分文本
        chunks = self.split_into_chunks(cleaned)
        if not chunks:
            print(f"  ⚠️ 切分后无有效块（可能太短，min_length={self.min_length}）: {Path(file_path).name}")
        
        return chunks
    
    def process_directory(
        self,
        data_dir: str,
        output_file: Optional[str] = None,
        max_files: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        处理整个目录中的所有txt文件
        
        Args:
            data_dir: 数据目录路径
            output_file: 如果提供，将结果保存为JSONL
            max_files: 最大处理文件数
            
        Returns:
            处理后的数据列表，每条数据包含{'text': ...}
        """
        data_path = Path(data_dir)
        # 同时匹配 .txt 和 .TXT（大小写不敏感）
        txt_files_lower = list(data_path.rglob("*.txt"))
        txt_files_upper = list(data_path.rglob("*.TXT"))
        txt_files = txt_files_lower + txt_files_upper
        
        if max_files:
            txt_files = txt_files[:max_files]
        
        print(f"Found {len(txt_files)} txt files in {data_dir}")
        
        if len(txt_files) == 0:
            raise FileNotFoundError(
                f"在 {data_dir} 目录下没有找到任何.txt文件\n"
                f"请确保数据文件已上传到该目录"
            )
        
        all_chunks = []
        processed_files = 0
        failed_files = []
        
        for txt_file in txt_files:
            chunks = self.process_file(str(txt_file))
            
            if chunks:
                processed_files += 1
                all_chunks.extend([{'text': chunk, 'source': txt_file.name} for chunk in chunks])
                
                if processed_files % 10 == 0:
                    print(f"Processed {processed_files}/{len(txt_files)} files, got {len(all_chunks)} chunks")
            else:
                failed_files.append(txt_file.name)
                print(f"⚠️ 文件处理失败（内容为空或太短）: {txt_file.name}")
        
        print(f"\nTotal: {len(all_chunks)} chunks from {processed_files} files")
        
        # 保存到文件
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                for chunk in all_chunks:
                    f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
            print(f"Saved to {output_file}")
        
        return all_chunks
    
    def tokenize_dataset(
        self,
        dataset: List[Dict[str, str]]
    ) -> Dataset:
        """
        对数据集进行tokenization
        
        Args:
            dataset: 包含'text'字段的字典列表
            
        Returns:
            HuggingFace Dataset对象
        """
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=self.max_length,
                padding=False,  # 在DataLoader中动态padding
            )
        
        # 转换为Dataset
        hf_dataset = Dataset.from_list(dataset)
        
        # Tokenize
        tokenized = hf_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text', 'source'],
            desc="Tokenizing"
        )
        
        return tokenized
    
    def create_train_val_split(
        self,
        dataset: Dataset,
        val_ratio: float = 0.05,
        seed: int = 42
    ) -> Tuple[Dataset, Dataset]:
        """
        划分训练集和验证集
        
        Args:
            dataset: 完整数据集
            val_ratio: 验证集比例
            seed: 随机种子
            
        Returns:
            train_dataset, val_dataset
        """
        split = dataset.train_test_split(test_size=val_ratio, seed=seed)
        return split['train'], split['test']


class WuxiaStyleAnalyzer:
    """武侠风格分析器（用于评估）"""
    
    def __init__(self):
        self.function_words = WuxiaDataProcessor.WUXIA_FUNCTION_WORDS
    
    def compute_style_score(self, text: str) -> Dict[str, float]:
        """
        计算武侠风格得分
        
        Args:
            text: 输入文本
            
        Returns:
            包含各项指标的字典
        """
        # 虚词密度
        function_word_count = sum(1 for word in self.function_words if word in text)
        function_word_density = function_word_count / len(text) if text else 0
        
        # 句子长度（武侠风格倾向于较长句子）
        sentences = re.split(r'[。！？]', text)
        avg_sentence_length = sum(len(s) for s in sentences) / len(sentences) if sentences else 0
        
        # 成语/四字词密度（简单估计）
        four_char_phrases = len(re.findall(r'[\u4e00-\u9fa5]{4}', text))
        phrase_density = four_char_phrases / (len(text) / 4) if text else 0
        
        return {
            'function_word_density': function_word_density,
            'avg_sentence_length': avg_sentence_length,
            'phrase_density': phrase_density,
        }


# 测试代码
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # 测试tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 创建处理器
    processor = WuxiaDataProcessor(tokenizer, max_length=256)
    
    # 测试文本清洗
    test_texts = [
        "剑光一闪，那人已然倒地，血溅三尺。他竟是天下第一剑客！",
        "他拿出手机拨打了电话，通知同伴准备行动。",  # 应该被过滤
        "少年缓缓抬起头，眼中尽是悲凉之色，轻声道：江湖事，江湖了。",
    ]
    
    print("\nTesting text cleaning:")
    for i, text in enumerate(test_texts):
        cleaned = processor.clean_text(text)
        print(f"Text {i+1}: {'✓ Passed' if cleaned else '✗ Filtered'}")
        if cleaned:
            print(f"  Original: {text}")
            print(f"  Cleaned: {cleaned}")
    
    # 测试文档切分
    long_text = "剑光一闪。那人倒地。" * 50
    chunks = processor.split_into_chunks(long_text)
    print(f"\nText splitting: {len(chunks)} chunks created")
    
    # 测试风格分析
    analyzer = WuxiaStyleAnalyzer()
    style_score = analyzer.compute_style_score(test_texts[0])
    print(f"\nStyle analysis for text 1:")
    for key, value in style_score.items():
        print(f"  {key}: {value:.4f}")
    
    # 测试处理data目录
    data_dir = "data"
    if os.path.exists(data_dir):
        print(f"\nProcessing {data_dir} directory...")
        chunks = processor.process_directory(
            data_dir,
            output_file="processed_wuxia_data.jsonl",
            max_files=5  # 测试时只处理5个文件
        )
        print(f"Total chunks: {len(chunks)}")
        
        if chunks:
            print(f"\nFirst chunk preview:")
            print(chunks[0]['text'][:200] + "...")
    else:
        print(f"\n{data_dir} directory not found. Skipping directory processing test.")
    
    print("\n✓ All tests completed!")
