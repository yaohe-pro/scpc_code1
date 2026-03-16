#!/usr/bin/env python3
"""
直接下载ECF数据集的JSON文件
绕过Hugging Face的数据集格式问题
"""
import os
import json
import requests
from pathlib import Path
from tqdm import tqdm

# ECF数据集的文件URL
BASE_URL = "https://huggingface.co/datasets/NUSTM/ECF/resolve/main"
FILES = {
    "train": [
        "span/train.json",
        "utterance/train.json"
    ],
    "validation": [
        "span/dev.json",
        "utterance/dev.json"
    ],
    "test": [
        "span/test.json",
        "utterance/test.json"
    ]
}

def download_file(url: str, output_path: Path) -> bool:
    """下载单个文件"""
    try:
        print(f"下载: {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        
        print(f"✅ 已保存: {output_path}")
        return True
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def load_and_convert_json(file_path: Path) -> list:
    """加载并转换JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    except Exception as e:
        print(f"❌ 加载JSON失败 {file_path}: {e}")
        return []

def main():
    """主函数"""
    print("=" * 70)
    print("直接下载ECF数据集")
    print("=" * 70)
    
    output_dir = Path("data/raw/ECF")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载文件
    downloaded_files = {}
    
    for split, files in FILES.items():
        print(f"\n处理 {split} 分割...")
        split_dir = output_dir / split
        split_dir.mkdir(exist_ok=True)
        
        for file_path in files:
            url = f"{BASE_URL}/{file_path}"
            local_path = split_dir / Path(file_path).name
            
            if local_path.exists():
                print(f"⏭️  文件已存在: {local_path}")
            else:
                if download_file(url, local_path):
                    downloaded_files.setdefault(split, []).append(local_path)
                else:
                    print(f"⚠️  下载失败，但继续...")
    
    print("\n" + "=" * 70)
    print("下载完成！")
    print("=" * 70)
    
    # 尝试加载和转换数据
    print("\n检查下载的文件...")
    for split, files in downloaded_files.items():
        print(f"\n{split} 分割:")
        for file_path in files:
            data = load_and_convert_json(file_path)
            if data:
                print(f"  ✅ {file_path.name}: {len(data)} 条记录")
            else:
                print(f"  ⚠️  {file_path.name}: 无法加载")
    
    print("\n提示:")
    print("1. 如果下载成功，数据文件在 data/raw/ECF/ 目录下")
    print("2. 可以运行 python3 src/ecf_huggingface_loader.py 进行转换")
    print("3. 或者直接使用这些JSON文件")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断下载")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

