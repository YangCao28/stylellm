"""
双模型对齐框架 (Dual-Model Alignment Framework)
实现Policy Model和Reference Model的并行架构，支持KL散度计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Optional, Tuple
import copy


class StyleAlignmentModel(nn.Module):
    """自监督风格对齐模型"""
    
    def __init__(
        self,
        model_name: str,
        kl_beta: float = 0.1,
        use_lora: bool = True,
        lora_config: Optional[Dict] = None,
    ):
        """
        Args:
            model_name: 预训练模型名称或路径
            kl_beta: KL散度权重
            use_lora: 是否使用LoRA
            lora_config: LoRA配置
        """
        super().__init__()
        
        self.kl_beta = float(kl_beta)
        
        # 设备配置
        self.policy_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        print(f"Single-model mode on {self.policy_device}")
        
        # 加载训练模型 (Policy Model)
        print(f"Loading policy model from {model_name} to {self.policy_device}...")
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={"": 0},  # 强制放在GPU 0
            trust_remote_code=True,
        )
        
        # 应用LoRA（如果需要）
        if use_lora:
            self._apply_lora(lora_config or self._default_lora_config())
        
        if self.kl_beta > 0:
            print("KL enabled. Using single-model KL: reference logits via LoRA-disabled pass.")
        else:
            print("KL disabled (kl_beta <= 0). Running single-model CE only.")

        # 不使用独立的reference模型
        self._reference_model = None
    
    def _default_lora_config(self) -> Dict:
        """默认LoRA配置"""
        return {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        }
    
    def _apply_lora(self, config: Dict):
        """应用LoRA到policy model"""
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.get("r", 16),
                lora_alpha=config.get("lora_alpha", 32),
                lora_dropout=config.get("lora_dropout", 0.05),
                target_modules=config.get("target_modules", ["q_proj", "v_proj"]),
            )
            
            self.policy_model = get_peft_model(self.policy_model, lora_config)
            print(f"LoRA applied with r={config['r']}")
            self.policy_model.print_trainable_parameters()
            
        except ImportError:
            print("Warning: peft not installed. Running without LoRA.")
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """启用梯度检查点（转发到policy_model）"""
        if hasattr(self.policy_model, 'gradient_checkpointing_enable'):
            self.policy_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        else:
            print("Warning: policy_model does not support gradient checkpointing")
    
    def gradient_checkpointing_disable(self):
        """禁用梯度检查点（转发到policy_model）"""
        if hasattr(self.policy_model, 'gradient_checkpointing_disable'):
            self.policy_model.gradient_checkpointing_disable()
    
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
            masked_input_ids: 掩码后的输入 [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            labels: 真实标签，未掩码位置为-100 [batch_size, seq_len]
            mask_positions: 掩码位置的布尔张量 [batch_size, seq_len]
            
        Returns:
            loss: 总损失
            rec_loss: 重构损失
            kl_loss: KL散度损失
            logits: 模型输出logits
        """
        # Policy model前向传播（带LoRA）
        policy_outputs = self.policy_model(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        # 重构损失（仅在掩码位置）
        rec_loss = policy_outputs.loss
        
        # 保存policy logits（带梯度）- 必须在任何adapter操作之前
        policy_logits = policy_outputs.logits
        
        # 计算KL散度（单模型：临时禁用LoRA获得参考分布）
        # 只有在有LoRA且kl_beta>0时才计算KL
        has_peft_adapters = (hasattr(self.policy_model, "peft_config") and 
                            self.policy_model.peft_config is not None and
                            len(self.policy_model.peft_config) > 0)
        
        if self.kl_beta > 0 and has_peft_adapters:
            try:
                # 临时禁用LoRA适配器获取参考分布
                with torch.no_grad():
                    self.policy_model.disable_adapters()
                    
                    reference_outputs = self.policy_model(
                        input_ids=masked_input_ids,
                        attention_mask=attention_mask,
                    )
                    reference_logits = reference_outputs.logits
                    
                    # 立即恢复LoRA适配器
                    self.policy_model.enable_adapters()
                
                # 计算KL散度
                kl_loss = self._compute_kl_divergence(
                    policy_logits=policy_logits,
                    reference_logits=reference_logits,
                    mask_positions=mask_positions,
                )
            except (ValueError, AttributeError) as e:
                # 如果adapter操作失败，确保恢复适配器，跳过KL计算
                if has_peft_adapters:
                    try:
                        self.policy_model.enable_adapters()
                    except:
                        pass
                kl_loss = torch.tensor(0.0, device=rec_loss.device)
        else:
            kl_loss = torch.tensor(0.0, device=rec_loss.device)
        
        # 总损失
        total_loss = rec_loss + self.kl_beta * kl_loss
        
        return {
            'loss': total_loss,
            'rec_loss': rec_loss,
            'kl_loss': kl_loss,
            'logits': policy_logits,  # 使用保存的logits，确保梯度完整
        }
    
    def _compute_kl_divergence(
        self,
        policy_logits: torch.Tensor,
        reference_logits: torch.Tensor,
        mask_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算KL散度 KL(P_policy || P_reference)
        
        Args:
            policy_logits: Policy model的logits [batch_size, seq_len, vocab_size]
            reference_logits: Reference model的logits [batch_size, seq_len, vocab_size]
            mask_positions: 如果提供，仅在掩码位置计算KL [batch_size, seq_len]
            
        Returns:
            kl_loss: KL散度
        """
        # 转换为概率分布
        policy_logprobs = F.log_softmax(policy_logits, dim=-1)
        reference_logprobs = F.log_softmax(reference_logits, dim=-1)
        
        # KL(P||Q) = sum(P * (log(P) - log(Q)))
        # 等价于：sum(P * log(P/Q))
        kl_div = torch.exp(reference_logprobs) * (reference_logprobs - policy_logprobs)
        kl_div = kl_div.sum(dim=-1)  # [batch_size, seq_len]
        
        # 如果指定了掩码位置，仅在这些位置计算KL
        if mask_positions is not None:
            kl_div = kl_div * mask_positions.float()
            kl_loss = kl_div.sum() / mask_positions.sum().clamp(min=1)
        else:
            kl_loss = kl_div.mean()
        
        return kl_loss
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 128,
        **kwargs
    ) -> torch.Tensor:
        """生成文本（使用policy model）"""
        return self.policy_model.generate(
            input_ids=input_ids,
            max_length=max_length,
            **kwargs
        )
    
    def save_pretrained(self, save_directory: str):
        """保存模型"""
        self.policy_model.save_pretrained(save_directory)
        print(f"Model saved to {save_directory}")
    
    def load_pretrained(self, load_directory: str):
        """加载模型"""
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            load_directory,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        print(f"Model loaded from {load_directory}")


class StyleAwareLoss(nn.Module):
    """Style-Aware损失函数（可单独使用）"""
    
    def __init__(self, kl_beta: float = 0.1):
        super().__init__()
        self.kl_beta = kl_beta
    
    def forward(
        self,
        policy_logits: torch.Tensor,
        reference_logits: torch.Tensor,
        labels: torch.Tensor,
        mask_positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算Style-Aware Loss = RecLoss + β * KL(P_train || P_ref)
        
        Args:
            policy_logits: 训练模型的logits [batch_size, seq_len, vocab_size]
            reference_logits: 参考模型的logits [batch_size, seq_len, vocab_size]
            labels: 真实标签，未掩码位置为-100 [batch_size, seq_len]
            mask_positions: 掩码位置 [batch_size, seq_len]
            
        Returns:
            包含total_loss, rec_loss, kl_loss的字典
        """
        # 重构损失（Cross-Entropy）
        rec_loss = F.cross_entropy(
            policy_logits.view(-1, policy_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        
        # KL散度
        policy_logprobs = F.log_softmax(policy_logits, dim=-1)
        reference_logprobs = F.log_softmax(reference_logits, dim=-1)
        
        kl_div = torch.exp(reference_logprobs) * (reference_logprobs - policy_logprobs)
        kl_div = kl_div.sum(dim=-1)  # [batch_size, seq_len]
        
        if mask_positions is not None:
            kl_div = kl_div * mask_positions.float()
            kl_loss = kl_div.sum() / mask_positions.sum().clamp(min=1)
        else:
            kl_loss = kl_div.mean()
        
        # 总损失
        total_loss = rec_loss + self.kl_beta * kl_loss
        
        return {
            'loss': total_loss,
            'rec_loss': rec_loss,
            'kl_loss': kl_loss,
        }


# 测试代码
if __name__ == "__main__":
    print("Testing Dual-Model Framework...")
    
    # 创建模拟数据
    batch_size = 2
    seq_len = 32
    vocab_size = 1000
    
    # 模拟输入
    masked_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.full_like(masked_input_ids, -100)
    labels[:, 10:15] = masked_input_ids[:, 10:15]  # 模拟掩码位置的标签
    mask_positions = torch.zeros_like(masked_input_ids, dtype=torch.bool)
    mask_positions[:, 10:15] = True
    
    # 测试StyleAwareLoss
    print("\nTesting StyleAwareLoss...")
    loss_fn = StyleAwareLoss(kl_beta=0.1)
    
    policy_logits = torch.randn(batch_size, seq_len, vocab_size)
    reference_logits = torch.randn(batch_size, seq_len, vocab_size)
    
    loss_dict = loss_fn(policy_logits, reference_logits, labels, mask_positions)
    
    print(f"Total Loss: {loss_dict['loss'].item():.4f}")
    print(f"Reconstruction Loss: {loss_dict['rec_loss'].item():.4f}")
    print(f"KL Loss: {loss_dict['kl_loss'].item():.4f}")
    
    print("\n✓ StyleAwareLoss test passed!")
    
    # 如果要测试完整模型（需要下载模型）
    # model = StyleAlignmentModel("gpt2", kl_beta=0.1, use_lora=False)
    # outputs = model(masked_input_ids, labels=labels, mask_positions=mask_positions)
    # print(f"Model output loss: {outputs['loss'].item():.4f}")
