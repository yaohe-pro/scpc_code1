#!/usr/bin/env python3
"""
使用Hugging Face Hub API下载多模态特征文件
"""
import os
from pathlib import Path

def download_with_hf_hub():
    """使用huggingface_hub库下载"""
    try:
        from huggingface_hub import hf_hub_download
        import numpy as np
        
        print("=" * 70)
        print("使用Hugging Face Hub下载多模态特征")
        print("=" * 70)
        
        repo_id = "NUSTM/ECF"
        output_dir = Path("data/raw/ECF")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = {
            "audio_embedding_6373.npy": "音频特征 (6373维)",
            "video_embedding_4096.npy": "视频特征 (4096维)"
        }
        
        downloaded = []
        
        for filename, description in files.items():
            print(f"\n{'='*70}")
            print(f"下载: {filename}")
            print(f"描述: {description}")
            print(f"{'='*70}")
            
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=str(output_dir),
                    local_dir=str(output_dir),
                    local_dir_use_symlinks=False
                )
                
                print(f"✅ 下载成功!")
                print(f"   保存位置: {local_path}")
                
                # 验证文件
                try:
                    data = np.load(local_path)
                    print(f"   文件形状: {data.shape}")
                    print(f"   数据类型: {data.dtype}")
                    
                    if "audio" in filename:
                        if len(data.shape) == 2 and data.shape[1] == 6373:
                            print(f"   ✅ 音频特征验证通过")
                        else:
                            print(f"   ⚠️  形状不符合预期")
                    elif "video" in filename:
                        if len(data.shape) == 2 and data.shape[1] == 4096:
                            print(f"   ✅ 视频特征验证通过")
                        else:
                            print(f"   ⚠️  形状不符合预期")
                    
                    downloaded.append(filename)
                except Exception as e:
                    print(f"   ⚠️  验证失败: {e}")
                
            except Exception as e:
                print(f"❌ 下载失败: {e}")
                print(f"   可能原因:")
                print(f"   1. 文件不存在于仓库中")
                print(f"   2. 网络连接问题")
                print(f"   3. 需要登录Hugging Face")
        
        print("\n" + "=" * 70)
        if downloaded:
            print(f"✅ 成功下载 {len(downloaded)} 个文件")
            for f in downloaded:
                print(f"   - {f}")
        else:
            print("❌ 未能下载任何文件")
            print("\n建议:")
            print("1. 检查文件是否存在于: https://huggingface.co/datasets/NUSTM/ECF/tree/main")
            print("2. 手动从网站下载")
            print("3. 使用VPN或代理")
        print("=" * 70)
        
        return len(downloaded) > 0
        
    except ImportError:
        print("❌ 需要安装huggingface_hub库")
        print("运行: pip install huggingface_hub")
        return False

def list_repo_files():
    """列出仓库中的所有文件"""
    try:
        from huggingface_hub import list_repo_files
        
        print("\n列出仓库文件...")
        repo_id = "NUSTM/ECF"
        files = list_repo_files(repo_id=repo_id, repo_type="dataset")
        
        print(f"\n找到 {len(files)} 个文件:")
        embedding_files = [f for f in files if 'embedding' in f.lower() or '.npy' in f.lower()]
        
        if embedding_files:
            print("\n可能的嵌入文件:")
            for f in embedding_files:
                print(f"  - {f}")
        else:
            print("\n未找到嵌入文件，列出所有文件:")
            for f in files[:20]:  # 只显示前20个
                print(f"  - {f}")
            if len(files) > 20:
                print(f"  ... 还有 {len(files) - 20} 个文件")
        
        return embedding_files
        
    except Exception as e:
        print(f"❌ 列出文件失败: {e}")
        return []

if __name__ == "__main__":
    print("尝试方法1: 使用huggingface_hub下载...")
    success = download_with_hf_hub()
    
    if not success:
        print("\n尝试方法2: 列出仓库文件...")
        files = list_repo_files()
        
        if files:
            print("\n找到嵌入文件，请手动下载:")
            for f in files:
                print(f"  https://huggingface.co/datasets/NUSTM/ECF/resolve/main/{f}")

