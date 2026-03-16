#!/usr/bin/env python3
"""
下载ECF数据集的多模态特征文件
- audio_embedding_6373.npy (6373维音频特征)
- video_embedding_4096.npy (4096维视觉特征)
"""
import requests
from pathlib import Path
from tqdm import tqdm
import os


def _parse_bool_env(var_name: str, default: bool) -> bool:
    raw = os.getenv(var_name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def create_mock_feature_files(output_dir: Path, num_rows: int = 128) -> bool:
    """测试模式：生成轻量 mock 多模态特征文件。"""
    try:
        import numpy as np

        output_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(42)
        audio = rng.normal(loc=0.0, scale=1.0, size=(num_rows, 6373)).astype("float32")
        video = rng.normal(loc=-1.9, scale=0.5, size=(num_rows, 4096)).astype("float32")

        audio_path = output_dir / "audio_embedding_6373.npy"
        video_path = output_dir / "video_embedding_4096.npy"

        np.save(audio_path, audio)
        np.save(video_path, video)

        print("\n🧪 TEST MODE: 已生成 mock 特征文件")
        print(f"  - {audio_path} shape={audio.shape}")
        print(f"  - {video_path} shape={video.shape}")
        return True
    except Exception as e:
        print(f"❌ 生成 mock 特征失败: {e}")
        return False

# Hugging Face数据集文件URL
BASE_URL = "https://huggingface.co/datasets/NUSTM/ECF/resolve/main"
FILES = {
    "audio_embedding_6373.npy": "audio_embedding_6373.npy",
    "video_embedding_4096.npy": "video_embedding_4096.npy"
}

def download_file(url: str, output_path: Path) -> bool:
    """下载单个文件"""
    try:
        print(f"\n下载: {url}")
        print(f"保存到: {output_path}")
        
        # 检查文件是否已存在
        if output_path.exists():
            file_size = output_path.stat().st_size
            print(f"⚠️  文件已存在 ({file_size / (1024*1024):.2f} MB)")
            response = requests.head(url, timeout=10)
            remote_size = int(response.headers.get('content-length', 0))
            
            if file_size == remote_size:
                print("✅ 文件完整，跳过下载")
                return True
            else:
                print(f"⚠️  文件大小不匹配，重新下载...")
        
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            if total_size == 0:
                # 没有Content-Length头，直接写入
                f.write(response.content)
                print(f"✅ 下载完成")
            else:
                # 使用进度条
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                print(f"✅ 下载完成 ({total_size / (1024*1024):.2f} MB)")
        
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ 下载失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

def verify_file(file_path: Path) -> bool:
    """验证文件"""
    try:
        import numpy as np
        
        if not file_path.exists():
            return False
        
        print(f"\n验证文件: {file_path.name}")
        data = np.load(file_path)
        
        if "audio" in file_path.name:
            expected_shape = (None, 6373)  # 可变长度，6373维
            print(f"  形状: {data.shape}")
            print(f"  数据类型: {data.dtype}")
            if len(data.shape) == 2 and data.shape[1] == 6373:
                print(f"  ✅ 音频特征验证通过 (6373维)")
                return True
            else:
                print(f"  ⚠️  形状不符合预期")
                return False
        elif "video" in file_path.name:
            expected_shape = (None, 4096)  # 可变长度，4096维
            print(f"  形状: {data.shape}")
            print(f"  数据类型: {data.dtype}")
            if len(data.shape) == 2 and data.shape[1] == 4096:
                print(f"  ✅ 视频特征验证通过 (4096维)")
                return True
            else:
                print(f"  ⚠️  形状不符合预期")
                return False
        
        return True
    except Exception as e:
        print(f"  ❌ 验证失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 70)
    print("下载ECF数据集多模态特征文件")
    print("=" * 70)
    
    output_dir = Path("data/raw/ECF")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_mode = _parse_bool_env("SCPC_TEST_MODE", False)
    if test_mode:
        return create_mock_feature_files(output_dir)
    
    print("\n要下载的文件:")
    print("  1. audio_embedding_6373.npy - 音频特征 (6373维)")
    print("  2. video_embedding_4096.npy - 视频特征 (4096维)")
    print("\n注意: 这些文件可能较大，下载需要一些时间...")
    
    downloaded = []
    failed = []
    
    for filename, filepath in FILES.items():
        url = f"{BASE_URL}/{filepath}"
        local_path = output_dir / filename
        
        print(f"\n{'='*70}")
        print(f"处理: {filename}")
        print(f"{'='*70}")
        
        if download_file(url, local_path):
            if verify_file(local_path):
                downloaded.append(filename)
            else:
                print(f"⚠️  文件下载但验证失败: {filename}")
                failed.append(filename)
        else:
            failed.append(filename)
    
    print("\n" + "=" * 70)
    print("下载总结")
    print("=" * 70)
    
    if downloaded:
        print(f"\n✅ 成功下载 ({len(downloaded)} 个文件):")
        for f in downloaded:
            file_path = output_dir / f
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  - {f} ({size_mb:.2f} MB)")
    
    if failed:
        print(f"\n❌ 下载失败 ({len(failed)} 个文件):")
        for f in failed:
            print(f"  - {f}")
        print("\n建议:")
        print("  1. 检查网络连接")
        print("  2. 手动从以下链接下载:")
        print("     https://huggingface.co/datasets/NUSTM/ECF/tree/main")
        print("  3. 将文件放在 data/raw/ECF/ 目录下")
    
    if downloaded:
        print("\n" + "=" * 70)
        print("✅ 多模态特征文件已就绪！")
        print("=" * 70)
        print("\n使用方法:")
        print("  代码会自动检测并加载这些特征文件")
        print("  运行: python3 main.py")
        print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断下载")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

