"""
批量转换文件编码为UTF-8
解决武侠小说文件GB2312/GBK乱码问题
"""
import sys
from pathlib import Path
import chardet

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def detect_encoding(file_path):
    """检测文件编码"""
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding'], result['confidence']


def convert_to_utf8(file_path, backup=True, delete_on_fail=False):
    """
    将文件转换为UTF-8编码
    
    Args:
        file_path: 文件路径
        backup: 是否备份原文件
        delete_on_fail: 转换失败时是否删除文件
    """
    try:
        # 检测原始编码
        encoding, confidence = detect_encoding(file_path)
        
        if encoding is None:
            print(f"⚠️ 无法检测编码: {file_path.name}")
            if delete_on_fail:
                file_path.unlink()
                print(f"🗑️ 已删除无法检测编码的文件: {file_path.name}")
            return False
        
        # 如果已经是UTF-8，跳过
        if encoding.lower() in ['utf-8', 'ascii']:
            print(f"✓ 已是UTF-8: {file_path.name}")
            return True
        
        print(f"🔄 转换 {file_path.name}: {encoding} (置信度: {confidence:.2f}) -> UTF-8")
        
        # 读取原始内容
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            content = f.read()
        
        # 检查转换后内容是否有效（不是全乱码）
        if len(content.strip()) < 50 or content.count('�') > len(content) * 0.3:
            print(f"⚠️ 转换后内容无效（乱码过多）: {file_path.name}")
            if delete_on_fail:
                file_path.unlink()
                print(f"🗑️ 已删除: {file_path.name}")
            return False
        
        # 备份原文件
        if backup:
            backup_path = file_path.with_suffix(file_path.suffix + '.bak')
            with open(backup_path, 'wb') as f:
                with open(file_path, 'rb') as src:
                    f.write(src.read())
        
        # 写入UTF-8
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ 转换成功: {file_path.name}")
        return True
        
    except Exception as e:
        print(f"❌ 转换失败 {file_path.name}: {e}")
        if delete_on_fail:
            try:
                file_path.unlink()
                print(f"🗑️ 已删除转换失败的文件: {file_path.name}")
            except:
                pass
        return False


def convert_directory(data_dir, backup=True, delete_on_fail=False):
    """
    转换目录下所有txt文件
    
    Args:
        data_dir: 数据目录
        backup: 是否备份
        delete_on_fail: 转换失败时是否删除文件
    """
    data_path = Path(data_dir)
    
    # 查找所有txt文件（包括.txt和.TXT）
    txt_files_lower = list(data_path.rglob("*.txt"))
    txt_files_upper = list(data_path.rglob("*.TXT"))
    txt_files = txt_files_lower + txt_files_upper
    
    if not txt_files:
        print(f"⚠️ 在 {data_dir} 中没有找到txt文件")
        return
    
    print(f"找到 {len(txt_files)} 个txt文件")
    print("="*60)
    
    success_count = 0
    fail_count = 0
    deleted_count = 0
    
    for txt_file in txt_files:
        file_existed = txt_file.exists()
        if convert_to_utf8(txt_file, backup=backup, delete_on_fail=delete_on_fail):
            success_count += 1
        else:
            fail_count += 1
            if delete_on_fail and not txt_file.exists():
                deleted_count += 1
    
    print("="*60)
    print(f"\n转换完成:")
    print(f"  ✅ 成功: {success_count}")
    print(f"  ❌ 失败: {fail_count}")
    if delete_on_fail:
        print(f"  🗑️ 已删除: {deleted_count}")
    
    if backup:
        print(f"\n原文件已备份为 .bak 文件")
        print(f"确认无误后可删除备份: find {data_dir} -name '*.bak' -delete")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='批量转换文件编码为UTF-8')
    parser.add_argument('--data-dir', type=str, default='./data', 
                        help='数据目录路径')
    parser.add_argument('--no-backup', action='store_true',
                        help='不备份原文件（谨慎使用）')
    parser.add_argument('--delete-on-fail', action='store_true',
                        help='删除无法转换的文件（谨慎使用）')
    
    args = parser.parse_args()
    
    print("="*60)
    print("文件编码转换工具")
    print("="*60)
    print(f"数据目录: {args.data_dir}")
    print(f"备份原文件: {'否' if args.no_backup else '是'}")
    print(f"删除失败文件: {'是' if args.delete_on_fail else '否'}")
    print("="*60)
    print()
    
    convert_directory(args.data_dir, backup=not args.no_backup, delete_on_fail=args.delete_on_fail)
