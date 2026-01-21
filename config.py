"""
配置文件
训练超参数和路径配置
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """模型配置"""
    model_name_or_path: str = "Qwen/Qwen2.5-7B"  # 基座模型
    use_lora: bool = True  # 是否使用LoRA
    lora_r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA alpha
    lora_dropout: float = 0.05  # LoRA dropout
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练参数
    output_dir: str = "./output/wuxia_style_model"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # 损失函数参数
    kl_beta: float = 0.1  # KL散度权重
    kl_schedule: str = "constant"  # constant, linear, cosine
    kl_beta_min: float = 0.05  # KL权重最小值（用于schedule）
    kl_beta_max: float = 0.2  # KL权重最大值（用于schedule）
    
    # 优化器
    optimizer: str = "adamw_torch"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # 学习率调度
    lr_scheduler_type: str = "cosine"
    
    # 保存和日志
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 500
    
    # 硬件优化
    fp16: bool = True  # 混合精度训练
    bf16: bool = False  # BF16（如果硬件支持）
    gradient_checkpointing: bool = True  # 梯度检查点
    
    # 其他
    seed: int = 42
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False


@dataclass
class DataConfig:
    """数据配置"""
    data_dir: str = "./data"  # 原始数据目录
    processed_data_file: str = "./processed_wuxia_data.jsonl"  # 处理后的数据
    
    # 数据处理参数
    max_length: int = 512  # 最大序列长度
    min_length: int = 32  # 最小序列长度（字符数）
    stride: int = 256  # 滑动窗口步长
    val_ratio: float = 0.05  # 验证集比例
    max_files: Optional[int] = None  # 最大处理文件数（None表示全部）
    
    # 掩码策略
    span_mask_ratio: float = 0.6  # Random Span Masking比例
    token_mask_ratio: float = 0.3  # Token-level Masking比例
    no_mask_ratio: float = 0.1  # 不掩码比例
    span_length_range: tuple = (3, 8)  # Span长度范围


@dataclass
class EvalConfig:
    """评估配置"""
    eval_batch_size: int = 8
    compute_perplexity: bool = True
    compute_ngram_overlap: bool = True
    ngram_sizes: List[int] = field(default_factory=lambda: [2, 3, 4])
    
    # 生成参数（用于评估）
    max_new_tokens: int = 100
    temperature: float = 0.9
    top_p: float = 0.9
    top_k: int = 50
    num_samples: int = 100  # 生成样本数用于评估


@dataclass
class Config:
    """总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    
    def __post_init__(self):
        """验证配置"""
        # 确保比例加起来为1
        total_ratio = (
            self.data.span_mask_ratio +
            self.data.token_mask_ratio +
            self.data.no_mask_ratio
        )
        assert abs(total_ratio - 1.0) < 0.01, f"Masking ratios must sum to 1.0, got {total_ratio}"
        
        # 确保KL beta在合理范围
        assert 0 <= self.training.kl_beta <= 1.0, "kl_beta must be in [0, 1]"
