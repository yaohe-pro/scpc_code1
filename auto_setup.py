#!/usr/bin/env python3
"""
自动配置环境并运行项目
"""
import os
import sys
import subprocess
import urllib.request
from pathlib import Path

def run_command(cmd, check=True):
    """运行命令"""
    print(f"执行: {cmd}")
    try:
        result = subprocess.run(
            cmd, shell=True, check=check,
            capture_output=True, text=True
        )
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        if e.stderr:
            print(f"错误: {e.stderr}")
        return False

def install_pip():
    """尝试安装 pip"""
    print("\n[1/6] 检查 pip...")
    
    # 检查是否已有 pip
    if run_command("python3 -m pip --version", check=False):
        print("✅ pip 已可用")
        return True
    
    if run_command("pip3 --version", check=False):
        print("✅ pip3 已可用")
        return True
    
    # 检查用户本地 pip
    local_pip = Path.home() / ".local" / "bin" / "pip3"
    if local_pip.exists():
        print(f"✅ 找到本地 pip: {local_pip}")
        os.environ["PATH"] = str(local_pip.parent) + ":" + os.environ.get("PATH", "")
        return True
    
    # 尝试下载并安装 get-pip.py
    print("尝试安装 pip...")
    try:
        print("下载 get-pip.py...")
        urllib.request.urlretrieve(
            "https://bootstrap.pypa.io/get-pip.py",
            "get-pip.py"
        )
        print("安装 pip...")
        if run_command("python3 get-pip.py --user", check=False):
            # 更新 PATH
            local_bin = Path.home() / ".local" / "bin"
            os.environ["PATH"] = str(local_bin) + ":" + os.environ.get("PATH", "")
            if (local_bin / "pip3").exists():
                print("✅ pip 安装成功")
                return True
    except Exception as e:
        print(f"⚠️  无法自动安装 pip: {e}")
    
    print("❌ 无法找到或安装 pip")
    print("请手动运行以下命令之一：")
    print("  sudo apt install python3-pip")
    print("  或")
    print("  wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py --user")
    return False

def get_pip_cmd():
    """获取 pip 命令"""
    if run_command("python3 -m pip --version", check=False):
        return "python3 -m pip"
    if run_command("pip3 --version", check=False):
        return "pip3"
    local_pip = Path.home() / ".local" / "bin" / "pip3"
    if local_pip.exists():
        return str(local_pip)
    return None

def install_dependencies():
    """安装项目依赖"""
    print("\n[2/6] 安装项目依赖...")
    
    pip_cmd = get_pip_cmd()
    if not pip_cmd:
        print("❌ 无法找到 pip 命令")
        return False
    
    print(f"使用: {pip_cmd}")
    
    # 升级 pip
    print("升级 pip...")
    run_command(f"{pip_cmd} install --upgrade pip -q", check=False)
    
    # 安装依赖
    print("安装 requirements.txt 中的依赖...")
    if run_command(f"{pip_cmd} install -r requirements.txt"):
        print("✅ 依赖安装完成")
        return True
    else:
        print("⚠️  依赖安装可能有问题，但继续尝试运行...")
        return True  # 继续尝试

def setup_env_file():
    """设置 .env 文件"""
    print("\n[3/6] 检查环境配置...")
    
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env 文件已存在")
        return
    
    print("创建默认 .env 文件...")
    env_content = """# OpenAI API配置
OPENAI_API_KEY=YOUR_API_KEY_HERE
OPENAI_BASE_URL=https://api.openai.com/v1

# Google Gemini API配置（可选）
# GOOGLE_API_KEY=your_google_api_key_here

# 模型选择: 'gpt-4o' 或 'gemini-1.5-pro'
MODEL_NAME=gpt-4o
"""
    env_file.write_text(env_content)
    print("✅ 已创建 .env 文件")
    print("⚠️  请编辑 .env 文件添加您的 API 密钥（如果需要使用真实 API）")

def check_dependencies():
    """检查关键依赖"""
    print("\n[4/6] 检查关键依赖...")
    
    critical_deps = [
        "numpy", "pandas", "openai", "python-dotenv", 
        "pyyaml", "tqdm"
    ]
    
    missing = []
    for dep in critical_deps:
        try:
            __import__(dep.replace("-", "_"))
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} 未安装")
            missing.append(dep)
    
    if missing:
        print(f"\n⚠️  缺少依赖: {', '.join(missing)}")
        return False
    return True

def run_project():
    """运行项目"""
    print("\n[5/6] 运行项目...")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            [sys.executable, "main.py"],
            cwd=Path.cwd()
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("自动配置环境并运行项目")
    print("=" * 60)
    
    # 切换到项目目录
    os.chdir(Path(__file__).parent)
    print(f"工作目录: {os.getcwd()}")
    
    # 安装 pip
    if not install_pip():
        print("\n⚠️  pip 未安装，尝试继续...")
    
    # 安装依赖
    install_dependencies()
    
    # 设置环境文件
    setup_env_file()
    
    # 检查依赖
    check_dependencies()
    
    # 运行项目
    print("\n" + "=" * 60)
    success = run_project()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ 项目运行完成！")
    else:
        print("⚠️  项目运行可能有问题，请检查上面的输出")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
