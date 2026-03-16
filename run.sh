#!/bin/bash
# 运行项目的启动脚本

cd "$(dirname "$0")"

# 设置本地 Python 环境
export PYTHONUSERBASE="$(pwd)/.local_python"
export PATH="$(pwd)/.local_python/bin:$PATH"
export PYTHONPATH="$(pwd)/.local_python/lib/python3.10/site-packages:$(pwd):$PYTHONPATH"

echo "============================================================"
echo "运行项目"
echo "============================================================"

# 默认以测试模式运行（可通过环境变量覆盖）
export SCPC_TEST_MODE="${SCPC_TEST_MODE:-1}"
export SCPC_FORCE_MOCK="${SCPC_FORCE_MOCK:-1}"
export SCPC_TEST_MAX_CONVS="${SCPC_TEST_MAX_CONVS:-5}"

echo "PYTHONPATH: $PYTHONPATH"
echo "SCPC_TEST_MODE: $SCPC_TEST_MODE"
echo "SCPC_FORCE_MOCK: $SCPC_FORCE_MOCK"
echo "SCPC_TEST_MAX_CONVS: $SCPC_TEST_MAX_CONVS"
echo ""

# 运行项目
python3 main.py
