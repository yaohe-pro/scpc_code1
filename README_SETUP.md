# 项目环境配置和运行指南

## 快速开始

由于系统权限限制，需要先安装系统级 Python 包管理器。

### 方法 1: 使用系统包管理器（推荐）

```bash
# 1. 安装 pip 和 venv
sudo apt update
sudo apt install -y python3-pip python3-venv

# 2. 创建虚拟环境
cd /path/to/scpc_code
python3 -m venv venv

# 3. 激活虚拟环境
source venv/bin/activate

# 4. 安装依赖
pip install -r requirements.txt

# 5. 运行项目
python3 main.py
```

### 方法 2: 使用用户级安装

如果无法使用 sudo，可以尝试：

```bash
# 1. 下载并安装 pip（如果还没有）
wget https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py --user

# 2. 将用户 bin 目录添加到 PATH
export PATH="$HOME/.local/bin:$PATH"

# 3. 安装依赖
pip3 install --user -r requirements.txt

# 4. 运行项目
python3 main.py
```

## 环境配置

### 1. API 密钥配置

项目需要 OpenAI API 密钥（可选，如果没有会使用模拟模式）。

编辑 `.env` 文件：

```bash
# 如果 .env 文件不存在，运行：
python3 setup_api.py

# 然后编辑 .env 文件，添加您的 API 密钥
nano .env
```

`.env` 文件内容示例：

```
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o
```

### 2. 配置文件

项目配置文件位于 `config/config.yaml`，可以根据需要修改。

## 运行项目

### 测试模式（GitHub 默认推荐）

测试模式用于快速联调和演示：
- 强制使用 Mock LLM（不依赖真实 API）
- 仅处理少量测试对话（默认 5 个）
- 结果写入 `results/test_mode/`

```bash
# 方式 1：直接运行（推荐）
./run.sh

# 方式 2：手动指定测试参数
SCPC_TEST_MODE=1 SCPC_FORCE_MOCK=1 SCPC_TEST_MAX_CONVS=5 python3 main.py
```

如果需要切回完整模式：

```bash
SCPC_TEST_MODE=0 SCPC_FORCE_MOCK=0 python3 main.py
```

### 直接运行

```bash
python3 main.py
```

### 使用启动脚本

```bash
./run.sh
```

## 项目结构

- `main.py` - 主程序入口
- `src/` - 源代码目录
- `data/` - 数据目录
- `config/` - 配置文件
- `scripts/` - 工具脚本
- `requirements.txt` - Python 依赖列表

## 依赖说明

主要依赖包括：
- openai - OpenAI API 客户端
- numpy, pandas - 数据处理
- pillow, opencv-python - 图像处理
- librosa, soundfile - 音频处理
- scikit-learn - 评估指标
- datasets - Hugging Face 数据集
- python-dotenv - 环境变量管理
- pyyaml - YAML 配置文件解析

## 故障排除

### 问题 1: 权限错误

如果遇到权限错误，请使用虚拟环境或用户级安装。

### 问题 2: 缺少依赖

确保所有依赖都已安装：
```bash
pip3 install -r requirements.txt
```

### 问题 3: API 密钥错误

如果没有 API 密钥，项目会自动使用模拟模式运行。

## 自动安装脚本

如果系统已配置好 pip，可以运行：

```bash
python3 auto_setup.py
```

这个脚本会尝试自动安装所有依赖并运行项目。
