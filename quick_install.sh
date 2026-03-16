#!/bin/bash
# 快速安装脚本 - 需要 sudo 权限

set -e

echo "============================================================"
echo "快速安装和配置项目环境"
echo "============================================================"

# 检查是否在项目目录
if [ ! -f "requirements.txt" ]; then
    echo "❌ 请在项目根目录运行此脚本"
    exit 1
fi

# 安装系统依赖
echo ""
echo "[1/4] 安装系统依赖..."
sudo apt update -qq
sudo apt install -y python3-pip python3-venv python3-dev 2>&1 | grep -E "(正在|Setting|Unpacking|正在设置)" || true

# 创建虚拟环境
echo ""
echo "[2/4] 创建虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ 虚拟环境已创建"
else
    echo "✅ 虚拟环境已存在"
fi

# 激活虚拟环境并安装依赖
echo ""
echo "[3/4] 安装 Python 依赖..."
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt

echo "✅ 依赖安装完成"

# 检查 .env 文件
echo ""
echo "[4/4] 检查环境配置..."
if [ ! -f ".env" ]; then
    echo "创建默认 .env 文件..."
    cat > .env << 'EOF'
# OpenAI API配置
OPENAI_API_KEY=YOUR_API_KEY_HERE
OPENAI_BASE_URL=https://api.openai.com/v1

# Google Gemini API配置（可选）
# GOOGLE_API_KEY=your_google_api_key_here

# 模型选择: 'gpt-4o' 或 'gemini-1.5-pro'
MODEL_NAME=gpt-4o
EOF
    echo "✅ 已创建 .env 文件"
    echo "⚠️  请编辑 .env 文件添加您的 API 密钥（如果需要使用真实 API）"
else
    echo "✅ .env 文件已存在"
fi

echo ""
echo "============================================================"
echo "✅ 环境配置完成！"
echo "============================================================"
echo ""
echo "下一步："
echo "1. 如果需要使用真实 API，请编辑 .env 文件添加 API 密钥"
echo "2. 运行项目: source venv/bin/activate && python3 main.py"
echo ""
echo "现在运行项目..."
echo "============================================================"

# 运行项目
python3 main.py
