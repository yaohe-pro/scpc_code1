#!/bin/bash
# 自动配置环境并运行项目

set -e

echo "============================================================"
echo "开始自动配置环境..."
echo "============================================================"

# 检查并安装系统依赖
echo ""
echo "[1/5] 检查系统依赖..."
if ! command -v pip3 &> /dev/null; then
    echo "需要安装 pip3，请运行以下命令（需要sudo权限）："
    echo "  sudo apt update && sudo apt install -y python3-pip python3-venv"
    echo ""
    echo "或者使用以下命令安装 pip（不需要sudo）："
    echo "  wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py --user"
    echo ""
    read -p "是否已安装 pip3？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "请先安装 pip3，然后重新运行此脚本"
        exit 1
    fi
fi

# 确定 pip 命令
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
elif python3 -m pip --version &> /dev/null; then
    PIP_CMD="python3 -m pip"
elif [ -f ~/.local/bin/pip3 ]; then
    PIP_CMD="$HOME/.local/bin/pip3"
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "❌ 无法找到 pip，请先安装"
    exit 1
fi

echo "✅ 使用 pip: $PIP_CMD"

# 创建虚拟环境（如果可能）
echo ""
echo "[2/5] 设置虚拟环境..."
if python3 -m venv --help &> /dev/null; then
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        echo "✅ 虚拟环境已创建"
    else
        echo "✅ 虚拟环境已存在"
    fi
    source venv/bin/activate
    PIP_CMD="pip"
else
    echo "⚠️  无法创建虚拟环境，使用系统 Python"
fi

# 升级 pip
echo ""
echo "[3/5] 升级 pip..."
$PIP_CMD install --upgrade pip -q

# 安装项目依赖
echo ""
echo "[4/5] 安装项目依赖..."
$PIP_CMD install -r requirements.txt

echo "✅ 依赖安装完成"

# 检查 .env 文件
echo ""
echo "[5/5] 检查环境配置..."
if [ ! -f ".env" ]; then
    echo "⚠️  未找到 .env 文件，创建默认配置..."
    cat > .env << 'EOF'
# OpenAI API配置
OPENAI_API_KEY=YOUR_API_KEY_HERE
OPENAI_BASE_URL=https://api.openai.com/v1

# Google Gemini API配置（可选）
# GOOGLE_API_KEY=your_google_api_key_here

# 模型选择: 'gpt-4o' 或 'gemini-1.5-pro'
MODEL_NAME=gpt-4o
EOF
    echo "✅ 已创建 .env 文件，请编辑它并添加您的 API 密钥"
    echo "   文件位置: $(pwd)/.env"
else
    echo "✅ .env 文件已存在"
fi

echo ""
echo "============================================================"
echo "✅ 环境配置完成！"
echo "============================================================"
echo ""
echo "下一步："
echo "1. 编辑 .env 文件，添加您的 API 密钥（如果需要使用真实 API）"
echo "2. 运行项目: python3 main.py"
echo ""
echo "现在运行项目..."
echo "============================================================"

# 运行项目
python3 main.py
