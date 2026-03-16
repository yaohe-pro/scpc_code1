#!/usr/bin/env python3
"""检查环境是否准备好"""
import sys

print("=" * 60)
print("环境检查")
print("=" * 60)

# 检查Python版本
print(f"\n✅ Python版本: {sys.version}")

# 检查datasets库
try:
    import datasets
    print(f"✅ datasets库已安装 (版本: {datasets.__version__})")
    datasets_available = True
except ImportError:
    print("❌ datasets库未安装")
    print("   请运行: pip install datasets")
    datasets_available = False

# 检查numpy
try:
    import numpy as np
    print(f"✅ numpy已安装 (版本: {np.__version__})")
except ImportError:
    print("❌ numpy未安装")
    print("   请运行: pip install numpy")

# 检查其他依赖
dependencies = {
    "json": "标准库",
    "pathlib": "标准库",
    "typing": "标准库"
}

for dep, note in dependencies.items():
    try:
        __import__(dep)
        print(f"✅ {dep} 可用 ({note})")
    except ImportError:
        print(f"❌ {dep} 不可用")

print("\n" + "=" * 60)
if datasets_available:
    print("✅ 环境检查通过！可以下载ECF数据集")
    print("\n下一步:")
    print("  python3 src/ecf_huggingface_loader.py")
else:
    print("⚠️  请先安装datasets库")
    print("\n运行:")
    print("  pip install datasets")
print("=" * 60)

