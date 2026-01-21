"""
快速测试脚本
测试各个模块是否正常工作
"""

import sys
import os
from pathlib import Path

print("="*60)
print("武侠风格对齐框架 - 模块测试")
print("="*60)

# 测试1: 导入模块
print("\n[1/6] 测试模块导入...")
try:
    from masking_engine import DynamicMaskingEngine
    from style_alignment_model import StyleAlignmentModel, StyleAwareLoss
    from data_processor import WuxiaDataProcessor, WuxiaStyleAnalyzer
    from evaluator import WuxiaEvaluator
    from config import Config
    print("✓ 所有模块导入成功")
except Exception as e:
    print(f"✗ 模块导入失败: {e}")
    sys.exit(1)

# 测试2: 配置加载
print("\n[2/6] 测试配置...")
try:
    config = Config()
    print(f"✓ 配置加载成功")
    print(f"  - 模型: {config.model.model_name_or_path}")
    print(f"  - KL Beta: {config.training.kl_beta}")
    print(f"  - 最大长度: {config.data.max_length}")
except Exception as e:
    print(f"✗ 配置加载失败: {e}")
    sys.exit(1)

# 测试3: 数据处理
print("\n[3/6] 测试数据处理...")
try:
    from transformers import AutoTokenizer
    
    # 使用较小的模型进行测试
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    processor = WuxiaDataProcessor(tokenizer, max_length=128)
    
    # 测试文本清洗
    test_text = "剑光一闪，那人已然倒地。他竟是江湖第一高手！"
    cleaned = processor.clean_text(test_text)
    assert cleaned is not None, "文本清洗失败"
    
    # 测试切分
    long_text = test_text * 10
    chunks = processor.split_into_chunks(long_text)
    assert len(chunks) > 0, "文本切分失败"
    
    print(f"✓ 数据处理正常")
    print(f"  - 清洗后长度: {len(cleaned)}")
    print(f"  - 切分块数: {len(chunks)}")
except Exception as e:
    print(f"✗ 数据处理失败: {e}")
    import traceback
    traceback.print_exc()

# 测试4: 掩码引擎
print("\n[4/6] 测试掩码引擎...")
try:
    # 添加mask token
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    
    masking_engine = DynamicMaskingEngine(tokenizer)
    
    # 测试掩码
    test_ids = tokenizer.encode("剑光一闪，那人倒地", return_tensors='pt')
    result = masking_engine.apply_masking(test_ids)
    
    assert 'masked_input_ids' in result, "掩码结果缺少masked_input_ids"
    assert 'labels' in result, "掩码结果缺少labels"
    assert 'mask_positions' in result, "掩码结果缺少mask_positions"
    
    num_masked = result['mask_positions'].sum().item()
    
    print(f"✓ 掩码引擎正常")
    print(f"  - 输入长度: {test_ids.shape[1]}")
    print(f"  - 掩码数量: {num_masked}")
except Exception as e:
    print(f"✗ 掩码引擎失败: {e}")
    import traceback
    traceback.print_exc()

# 测试5: 损失函数
print("\n[5/6] 测试损失函数...")
try:
    import torch
    
    loss_fn = StyleAwareLoss(kl_beta=0.1)
    
    # 创建模拟数据
    batch_size = 2
    seq_len = 10
    vocab_size = 100
    
    policy_logits = torch.randn(batch_size, seq_len, vocab_size)
    reference_logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels[:, :5] = -100  # 前5个位置忽略
    
    loss_dict = loss_fn(policy_logits, reference_logits, labels)
    
    assert 'loss' in loss_dict, "损失计算缺少loss"
    assert 'rec_loss' in loss_dict, "损失计算缺少rec_loss"
    assert 'kl_loss' in loss_dict, "损失计算缺少kl_loss"
    
    print(f"✓ 损失函数正常")
    print(f"  - 总损失: {loss_dict['loss'].item():.4f}")
    print(f"  - 重构损失: {loss_dict['rec_loss'].item():.4f}")
    print(f"  - KL损失: {loss_dict['kl_loss'].item():.4f}")
except Exception as e:
    print(f"✗ 损失函数失败: {e}")
    import traceback
    traceback.print_exc()

# 测试6: 评估器
print("\n[6/6] 测试评估器...")
try:
    reference_texts = [
        "剑光一闪，那人已然倒地。他竟是江湖第一高手！",
        "少年缓缓抬起头，眼中尽是悲凉之色。",
    ]
    
    generated_texts = [
        "剑气纵横，那人竟然败北。",
        "他缓缓而来，眼中却带着笑意。",
    ]
    
    evaluator = WuxiaEvaluator(reference_texts=reference_texts)
    results = evaluator.evaluate(generated_texts, compute_ppl=False)
    
    assert '2gram_overlap' in results, "评估结果缺少2gram_overlap"
    assert 'function_word_density' in results, "评估结果缺少function_word_density"
    
    print(f"✓ 评估器正常")
    print(f"  - 2-gram重合度: {results['2gram_overlap']:.4f}")
    print(f"  - 虚词密度: {results['function_word_density']:.4f}")
except Exception as e:
    print(f"✗ 评估器失败: {e}")
    import traceback
    traceback.print_exc()

# 测试7: 检查数据目录
print("\n[额外] 检查数据目录...")
data_dir = "data"
if os.path.exists(data_dir):
    txt_files = list(Path(data_dir).rglob("*.txt"))
    print(f"✓ 数据目录存在")
    print(f"  - txt文件数: {len(txt_files)}")
    if txt_files:
        print(f"  - 示例文件: {txt_files[0].name}")
else:
    print(f"⚠ 数据目录不存在: {data_dir}")
    print(f"  请运行: python copy_txt_files.py")

# 总结
print("\n" + "="*60)
print("测试完成!")
print("="*60)
print("\n下一步:")
print("1. 确保数据目录存在并包含txt文件")
print("2. 运行训练: python train.py")
print("3. 查看README.md了解详细使用方法")
print("\n祝训练顺利！⚔️")
