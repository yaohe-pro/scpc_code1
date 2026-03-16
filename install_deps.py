#!/usr/bin/env python3
"""
使用 pip 安装依赖（处理权限问题）
"""
import subprocess
import sys
import os
from pathlib import Path

def install_package(package):
    """安装单个包"""
    try:
        # 尝试使用 --user
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "--user", "--no-warn-script-location",
            package
        ])
        return True
    except:
        try:
            # 尝试使用 --target
            target_dir = Path(".local_python/lib/python3.10/site-packages")
            target_dir.mkdir(parents=True, exist_ok=True)
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "--target", str(target_dir),
                "--no-cache-dir",
                package
            ])
            return True
        except Exception as e:
            print(f"❌ 无法安装 {package}: {e}")
            return False

def main():
    """主函数"""
    print("=" * 60)
    print("安装项目依赖")
    print("=" * 60)
    
    # 读取 requirements.txt
    req_file = Path("requirements.txt")
    if not req_file.exists():
        print("❌ 未找到 requirements.txt")
        return
    
    packages = []
    with open(req_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # 提取包名（去掉版本号）
                pkg = line.split(">=")[0].split("==")[0].split("<")[0].strip()
                if pkg:
                    packages.append(pkg)
    
    print(f"\n需要安装 {len(packages)} 个包...")
    print("\n注意：由于权限限制，将尝试多种安装方法")
    print("如果安装失败，请手动运行: sudo apt install python3-pip")
    print("然后运行: pip3 install -r requirements.txt\n")
    
    success = 0
    failed = []
    
    for pkg in packages:
        print(f"\n安装 {pkg}...")
        if install_package(pkg):
            print(f"✅ {pkg} 安装成功")
            success += 1
        else:
            print(f"❌ {pkg} 安装失败")
            failed.append(pkg)
    
    print("\n" + "=" * 60)
    print(f"安装完成: {success}/{len(packages)} 成功")
    if failed:
        print(f"失败的包: {', '.join(failed)}")
        print("\n请手动安装失败的包:")
        for pkg in failed:
            print(f"  pip3 install {pkg}")
    print("=" * 60)

if __name__ == "__main__":
    main()
