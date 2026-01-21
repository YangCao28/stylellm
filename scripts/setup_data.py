"""
数据准备工具
复制txt文件到data目录
"""
import os
import shutil
from pathlib import Path

def copy_txt_files(source_dir=".", target_dir="data"):
    """复制所有txt文件到data目录"""
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 创建目标目录
    target_path.mkdir(exist_ok=True)
    
    # 查找所有txt文件
    txt_files = list(source_path.rglob("*.txt"))
    txt_files = [f for f in txt_files if "data" not in str(f)]
    
    print(f"找到 {len(txt_files)} 个txt文件")
    
    # 复制文件
    copied = 0
    for txt_file in txt_files:
        try:
            # 使用原始文件名（扁平化）
            target_file = target_path / txt_file.name
            if not target_file.exists():
                shutil.copy2(txt_file, target_file)
                copied += 1
                print(f"✓ 复制: {txt_file.name}")
        except Exception as e:
            print(f"✗ 错误: {txt_file.name} - {e}")
    
    print(f"\n完成！复制了 {copied} 个文件到 {target_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="复制txt文件到data目录")
    parser.add_argument("--source", default=".", help="源目录")
    parser.add_argument("--target", default="data", help="目标目录")
    args = parser.parse_args()
    
    copy_txt_files(args.source, args.target)
