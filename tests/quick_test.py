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
    config.model.model_name_or_path = "gpt2"  # 使用小模型
    config.training.num_train_epochs = 1
    config.training.per_device_train_batch_size = 1
    config.training.max_steps = 10  # 只训练10步
    config.data.max_files = 5  # 只使用5个文件
    
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
