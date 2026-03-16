#!/usr/bin/env python3
"""
ECF数据集手动下载指南和辅助工具
如果自动下载失败，可以使用此脚本获取详细的手动下载步骤
"""
import os
from pathlib import Path

def print_manual_download_guide():
    """打印手动下载指南"""
    print("=" * 70)
    print("ECF数据集手动下载指南")
    print("=" * 70)
    
    print("\n由于网络问题，自动下载失败。请按照以下步骤手动下载：\n")
    
    print("方法1: 使用Hugging Face CLI（推荐）")
    print("-" * 70)
    print("1. 安装Hugging Face CLI:")
    print("   pip install huggingface_hub")
    print("\n2. 登录Hugging Face（可选，但推荐）:")
    print("   huggingface-cli login")
    print("\n3. 下载数据集:")
    print("   huggingface-cli download NUSTM/ECF --local-dir data/raw/ECF")
    
    print("\n\n方法2: 使用Python脚本（需要网络连接）")
    print("-" * 70)
    print("在能够访问Hugging Face的环境中运行：")
    print("""
from datasets import load_dataset

# 下载数据集
dataset = load_dataset("NUSTM/ECF")

# 保存为JSON
import json
for split in ["train", "validation", "test"]:
    data = dataset[split]
    output_dir = f"data/raw/ECF/{split}"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, item in enumerate(data):
        with open(f"{output_dir}/conv_{i+1}.json", "w") as f:
            json.dump(item, f, indent=2)
""")
    
    print("\n\n方法3: 从Hugging Face网站直接下载")
    print("-" * 70)
    print("1. 访问: https://huggingface.co/datasets/NUSTM/ECF")
    print("2. 点击 'Files and versions' 标签")
    print("3. 下载以下文件:")
    print("   - train/ 目录下的所有文件")
    print("   - validation/ 目录下的所有文件")
    print("   - test/ 目录下的所有文件")
    print("4. 解压并放在 data/raw/ECF/ 目录下")
    
    print("\n\n多模态特征文件（可选但推荐）")
    print("-" * 70)
    print("从数据集页面下载以下文件到 data/raw/ECF/ 目录：")
    print("  - audio_embedding_6373.npy")
    print("  - video_embedding_4096.npy")
    
    print("\n\n数据格式说明")
    print("-" * 70)
    print("下载的数据格式应该是JSON，包含以下字段：")
    print("""
{
    "conversation_ID": 1,
    "conversation": [
        {
            "emotion": "neutral",
            "speaker": "Chandler",
            "text": "...",
            "utterance_ID": 1,
            "video_name": "...",
            "video_source": [...]
        }
    ],
    "emotion-cause_pairs": [
        ["3_surprise", "原因文本"],
        ...
    ]
}
""")
    
    print("\n\n转换数据格式")
    print("-" * 70)
    print("下载后，运行转换脚本：")
    print("  python3 src/ecf_huggingface_loader.py")
    print("\n或者使用我们的数据加载器直接加载原始格式。")
    
    print("\n" + "=" * 70)
    print("提示：如果网络问题持续，可以：")
    print("1. 检查网络连接")
    print("2. 使用VPN或代理")
    print("3. 在能够访问Hugging Face的环境中下载后传输")
    print("=" * 70)


def create_sample_data_structure():
    """创建示例数据目录结构"""
    base_dir = Path("data/raw/ECF")
    
    # 创建目录结构
    for split in ["train", "validation", "test"]:
        (base_dir / split).mkdir(parents=True, exist_ok=True)
    
    # 创建README说明文件
    readme_content = """# ECF数据集目录

请将下载的ECF数据集文件放在此目录下。

目录结构应该是：
```
ECF/
├── train/          # 训练集对话文件
├── validation/     # 验证集对话文件
├── test/           # 测试集对话文件
├── audio_embedding_6373.npy   # 音频特征（可选）
└── video_embedding_4096.npy   # 视频特征（可选）
```

下载链接: https://huggingface.co/datasets/NUSTM/ECF
"""
    
    readme_file = base_dir / "README.txt"
    with open(readme_file, "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print(f"\n✅ 已创建数据目录结构: {base_dir}")
    print(f"✅ 已创建说明文件: {readme_file}")


if __name__ == "__main__":
    print_manual_download_guide()
    print("\n\n正在创建数据目录结构...")
    create_sample_data_structure()
    print("\n✅ 完成！")

