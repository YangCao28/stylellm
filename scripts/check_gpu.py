"""
GPU训练配置和检测脚本
检测GPU状态，优化训练配置
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import subprocess
import json
from typing import Dict, List
import platform


def check_gpu_availability() -> Dict:
    """检测GPU可用性和详细信息"""
    
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "pytorch_version": torch.__version__,
        "device_count": 0,
        "devices": [],
        "total_memory_gb": 0,
        "recommended_batch_size": 1,
    }
    
    if info["cuda_available"]:
        info["device_count"] = torch.cuda.device_count()
        
        for i in range(info["device_count"]):
            props = torch.cuda.get_device_properties(i)
            device_info = {
                "id": i,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": props.total_memory / 1024**3,
                "multi_processor_count": props.multi_processor_count,
            }
            
            # 获取当前显存使用情况
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                mem_free = device_info["total_memory_gb"] - mem_reserved
                
                device_info["memory_allocated_gb"] = mem_allocated
                device_info["memory_reserved_gb"] = mem_reserved
                device_info["memory_free_gb"] = mem_free
            
            info["devices"].append(device_info)
            info["total_memory_gb"] += device_info["total_memory_gb"]
        
        # 根据显存推荐batch size
        if info["total_memory_gb"] < 12:
            info["recommended_batch_size"] = 1
        elif info["total_memory_gb"] < 24:
            info["recommended_batch_size"] = 2
        elif info["total_memory_gb"] < 48:
            info["recommended_batch_size"] = 4
        else:
            info["recommended_batch_size"] = 8
    
    return info


def check_cudnn():
    """检查cuDNN"""
    cudnn_available = torch.backends.cudnn.is_available()
    cudnn_version = torch.backends.cudnn.version() if cudnn_available else None
    
    return {
        "available": cudnn_available,
        "version": cudnn_version,
        "enabled": torch.backends.cudnn.enabled,
    }


def check_nvidia_smi():
    """使用nvidia-smi获取详细GPU信息"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu", 
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        
        devices = []
        for line in result.stdout.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 8:
                devices.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "driver_version": parts[2],
                    "memory_total_mb": float(parts[3]),
                    "memory_free_mb": float(parts[4]),
                    "memory_used_mb": float(parts[5]),
                    "temperature_c": int(parts[6]) if parts[6] else 0,
                    "utilization_percent": int(parts[7]) if parts[7] else 0,
                })
        
        return {"available": True, "devices": devices}
    
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"available": False, "devices": []}


def recommend_training_config(gpu_info: Dict) -> Dict:
    """根据GPU配置推荐训练参数"""
    
    if not gpu_info["cuda_available"]:
        return {
            "device": "cpu",
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "fp16": False,
            "bf16": False,
            "gradient_checkpointing": True,
            "use_lora": True,
            "lora_r": 8,
            "warning": "⚠️  未检测到GPU，训练速度会非常慢！",
        }
    
    total_memory = gpu_info["total_memory_gb"]
    device_count = gpu_info["device_count"]
    
    # 根据显存推荐配置
    if total_memory < 12:  # <12GB
        config = {
            "device": "cuda",
            "batch_size": 1,
            "gradient_accumulation_steps": 16,
            "fp16": True,
            "bf16": False,
            "gradient_checkpointing": True,
            "use_lora": True,
            "lora_r": 8,
            "use_4bit": True,
            "max_length": 256,
            "warning": "⚠️  显存较小，建议使用4-bit量化",
        }
    
    elif total_memory < 24:  # 12-24GB
        config = {
            "device": "cuda",
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "fp16": True,
            "bf16": False,
            "gradient_checkpointing": True,
            "use_lora": True,
            "lora_r": 16,
            "use_4bit": False,
            "max_length": 512,
            "note": "✓ 配置良好，可正常训练7B模型",
        }
    
    elif total_memory < 48:  # 24-48GB
        config = {
            "device": "cuda",
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "fp16": True,
            "bf16": True,  # A100支持BF16
            "gradient_checkpointing": False,
            "use_lora": True,
            "lora_r": 32,
            "use_4bit": False,
            "max_length": 1024,
            "note": "✓ 配置优秀，可训练更大模型",
        }
    
    else:  # >48GB
        config = {
            "device": "cuda",
            "batch_size": 8,
            "gradient_accumulation_steps": 2,
            "fp16": False,
            "bf16": True,
            "gradient_checkpointing": False,
            "use_lora": True,
            "lora_r": 64,
            "use_4bit": False,
            "max_length": 2048,
            "note": "✓ 配置顶级，可进行全参数微调",
        }
    
    # 多GPU配置
    if device_count > 1:
        config["use_ddp"] = True
        config["device_count"] = device_count
        config["note"] = f"✓ 检测到{device_count}块GPU，将使用分布式训练"
    
    return config


def test_gpu_performance():
    """测试GPU性能"""
    if not torch.cuda.is_available():
        print("❌ 没有可用的GPU")
        return
    
    print("\n" + "="*70)
    print("🔥 GPU性能测试")
    print("="*70)
    
    device = torch.device("cuda:0")
    
    # 测试1: 矩阵乘法
    print("\n[1/3] 测试矩阵乘法...")
    size = 8192
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    import time
    torch.cuda.synchronize()
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    tflops = (2 * size ** 3) / elapsed / 1e12
    print(f"  矩阵大小: {size}x{size}")
    print(f"  耗时: {elapsed:.4f}秒")
    print(f"  性能: {tflops:.2f} TFLOPS")
    
    # 测试2: 显存带宽
    print("\n[2/3] 测试显存带宽...")
    size = 100 * 1024 * 1024  # 100M elements
    data = torch.randn(size, device=device)
    
    torch.cuda.synchronize()
    start = time.time()
    result = data * 2
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    bandwidth = (size * 4 * 2) / elapsed / 1e9  # 4 bytes per float, read+write
    print(f"  数据量: {size / 1024 / 1024:.2f} M elements")
    print(f"  耗时: {elapsed:.4f}秒")
    print(f"  带宽: {bandwidth:.2f} GB/s")
    
    # 测试3: 混合精度
    print("\n[3/3] 测试混合精度...")
    model = torch.nn.Linear(4096, 4096).to(device)
    x = torch.randn(128, 4096, device=device)
    
    # FP32
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            y = model(x)
    torch.cuda.synchronize()
    fp32_time = time.time() - start
    
    # FP16
    model_fp16 = model.half()
    x_fp16 = x.half()
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            y = model_fp16(x_fp16)
    torch.cuda.synchronize()
    fp16_time = time.time() - start
    
    print(f"  FP32: {fp32_time:.4f}秒")
    print(f"  FP16: {fp16_time:.4f}秒")
    print(f"  加速比: {fp32_time/fp16_time:.2f}x")
    
    print("\n✓ 性能测试完成")


def print_gpu_info():
    """打印详细的GPU信息"""
    
    print("\n" + "="*70)
    print("🖥️  GPU配置检测")
    print("="*70)
    
    # 系统信息
    print(f"\n系统信息:")
    print(f"  操作系统: {platform.system()} {platform.release()}")
    print(f"  Python版本: {platform.python_version()}")
    print(f"  PyTorch版本: {torch.__version__}")
    
    # GPU信息
    gpu_info = check_gpu_availability()
    
    print(f"\nCUDA信息:")
    print(f"  CUDA可用: {'是' if gpu_info['cuda_available'] else '否'}")
    if gpu_info["cuda_available"]:
        print(f"  CUDA版本: {gpu_info['cuda_version']}")
        print(f"  GPU数量: {gpu_info['device_count']}")
        print(f"  总显存: {gpu_info['total_memory_gb']:.2f} GB")
    
    # cuDNN信息
    cudnn_info = check_cudnn()
    print(f"\ncuDNN信息:")
    print(f"  可用: {'是' if cudnn_info['available'] else '否'}")
    if cudnn_info["available"]:
        print(f"  版本: {cudnn_info['version']}")
    
    # 详细GPU信息
    if gpu_info["cuda_available"]:
        print(f"\nGPU设备:")
        for device in gpu_info["devices"]:
            print(f"\n  GPU {device['id']}: {device['name']}")
            print(f"    计算能力: {device['compute_capability']}")
            print(f"    总显存: {device['total_memory_gb']:.2f} GB")
            if "memory_free_gb" in device:
                print(f"    已分配: {device['memory_allocated_gb']:.2f} GB")
                print(f"    可用: {device['memory_free_gb']:.2f} GB")
    
    # nvidia-smi信息
    nvidia_info = check_nvidia_smi()
    if nvidia_info["available"]:
        print(f"\nnvidia-smi信息:")
        for device in nvidia_info["devices"]:
            print(f"\n  GPU {device['index']}: {device['name']}")
            print(f"    驱动版本: {device['driver_version']}")
            print(f"    显存: {device['memory_used_mb']:.0f}/{device['memory_total_mb']:.0f} MB")
            print(f"    温度: {device['temperature_c']}°C")
            print(f"    利用率: {device['utilization_percent']}%")
    
    # 推荐配置
    print("\n" + "="*70)
    print("💡 推荐训练配置")
    print("="*70)
    
    recommended = recommend_training_config(gpu_info)
    
    print("\n基础配置:")
    print(f"  设备: {recommended.get('device', 'cpu')}")
    print(f"  Batch Size: {recommended.get('batch_size', 1)}")
    print(f"  梯度累积: {recommended.get('gradient_accumulation_steps', 1)}")
    print(f"  有效Batch Size: {recommended['batch_size'] * recommended['gradient_accumulation_steps']}")
    
    print("\n优化配置:")
    print(f"  FP16: {'是' if recommended.get('fp16', False) else '否'}")
    print(f"  BF16: {'是' if recommended.get('bf16', False) else '否'}")
    print(f"  梯度检查点: {'是' if recommended.get('gradient_checkpointing', False) else '否'}")
    print(f"  4-bit量化: {'是' if recommended.get('use_4bit', False) else '否'}")
    
    print("\nLoRA配置:")
    print(f"  使用LoRA: {'是' if recommended.get('use_lora', True) else '否'}")
    print(f"  LoRA Rank: {recommended.get('lora_r', 16)}")
    print(f"  最大长度: {recommended.get('max_length', 512)}")
    
    if "warning" in recommended:
        print(f"\n{recommended['warning']}")
    if "note" in recommended:
        print(f"\n{recommended['note']}")
    
    # 保存配置到文件
    output_file = "gpu_config.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "gpu_info": gpu_info,
            "cudnn_info": cudnn_info,
            "recommended_config": recommended
        }, f, indent=2)
    
    print(f"\n配置已保存到: {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU配置检测和优化")
    parser.add_argument("--test", action="store_true", help="运行性能测试")
    parser.add_argument("--export", type=str, help="导出配置到文件")
    
    args = parser.parse_args()
    
    # 打印GPU信息
    print_gpu_info()
    
    # 性能测试
    if args.test:
        test_gpu_performance()
    
    # 导出配置
    if args.export:
        gpu_info = check_gpu_availability()
        recommended = recommend_training_config(gpu_info)
        
        with open(args.export, 'w', encoding='utf-8') as f:
            json.dump(recommended, f, indent=2)
        
        print(f"\n配置已导出到: {args.export}")


if __name__ == "__main__":
    main()
