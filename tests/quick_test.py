"""
快速测试脚本 - 用小数据集快速验证训练流程
"""
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

def quick_test():
    """快速测试训练流程"""
    print("🚀 快速测试训练流程\n")
    
    # 创建简化配置
    config = Config()
    config.model.model_name_or_path = "Qwen/Qwen3-8B"  # 使用Qwen3-8B
    config.training.num_train_epochs = 1
    config.training.per_device_train_batch_size = 1
    config.training.gradient_accumulation_steps = 1  # 减少累积
    config.training.max_steps = 10  # 只训练10步
    config.training.gradient_checkpointing = True  # 启用梯度检查点
    config.training.fp16 = True  # 启用fp16节省显存
    config.data.max_files = 2  # 只使用2个文件（更少数据）
    config.data.min_length = 10  # 测试时降低最小长度
    config.data.max_length = 256  # 减少序列长度（从512降到256）
    
    print("配置:")
    print(f"  模型: {config.model.model_name_or_path}")
    print(f"  训练步数: {config.training.max_steps}")
    print(f"  数据文件: {config.data.max_files}")
    print()
    
    # 导入训练模块
    from train import train
    
    print("开始快速测试...\n")
    model, tokenizer = train(config)
    
    print("\n✓ 快速测试完成！")
    print("如果没有错误，说明环境配置正确。")
    print("可以开始正式训练了。")

if __name__ == "__main__":
    quick_test()
