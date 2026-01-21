"""
简化版风格对齐模型
纯 LoRA 微调，没有 KL 散度
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
from transformers import AutoModelForCausalLM


class StyleAlignmentModel(nn.Module):
    """
    简化的风格对齐模型
    
    功能：
    - 加载预训练模型
    - 应用 LoRA 适配器
    - 计算掩码语言模型损失
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        lora_config: Optional[Dict] = None,
        device: str = "cuda:0",
    ):
        super().__init__()
        
        self.model_name_or_path = model_name_or_path
        self.device = device
        
        # 加载模型
        print(f"Loading model from {model_name_or_path} to {device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        
        # 应用 LoRA
        if lora_config is None:
            lora_config = self._default_lora_config()
        self._apply_lora(lora_config)
    
    def _default_lora_config(self) -> Dict:
        """默认LoRA配置"""
        return {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        }
    
    def _apply_lora(self, config: Dict):
        """应用 LoRA"""
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.get("r", 16),
                lora_alpha=config.get("lora_alpha", 32),
                lora_dropout=config.get("lora_dropout", 0.05),
                target_modules=config.get("target_modules", ["q_proj", "v_proj"]),
            )
            
            self.model = get_peft_model(self.model, lora_config)
            print(f"LoRA applied with r={config['r']}")
            self.model.print_trainable_parameters()
            
        except ImportError:
            print("Warning: peft not installed. Running without LoRA.")
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """启用梯度检查点"""
        if hasattr(self.model, 'enable_input_require_grads'):
            self.model.enable_input_require_grads()
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        """禁用梯度检查点"""
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()
    
    def forward(
        self,
        masked_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        mask_positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            masked_input_ids: 掩码后的输入
            attention_mask: 注意力掩码
            labels: 真实标签
            mask_positions: 掩码位置（未使用）
            
        Returns:
            loss: MLM 损失
            logits: 模型输出
        """
        outputs = self.model(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        return {
            'loss': outputs.loss,
            'logits': outputs.logits,
        }
    
    def save_pretrained(self, save_directory: str):
        """保存 LoRA 权重"""
        print(f"Saving LoRA weights to {save_directory}...")
        self.model.save_pretrained(save_directory)
    
    def get_model(self):
        """获取内部模型"""
        return self.model
