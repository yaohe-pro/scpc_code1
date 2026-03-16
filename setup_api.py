#!/usr/bin/env python3
"""
配置API密钥和端点
"""
import os

def setup_api_config():
    """设置API配置"""
    api_key = "YOUR_API_KEY"
    base_url = "https://api.openai.com/v1"
    
    # 创建.env文件
    env_content = f"""# OpenAI API配置
OPENAI_API_KEY={api_key}
OPENAI_BASE_URL={base_url}

# Google Gemini API配置（可选）
# GOOGLE_API_KEY=your_google_api_key_here

# 模型选择: 'gpt-4o' 或 'gemini-1.5-pro'
MODEL_NAME=gpt-4o
"""
    
    env_file = ".env"
    with open(env_file, "w") as f:
        f.write(env_content)
    
    print("=" * 60)
    print("API配置已设置")
    print("=" * 60)
    print(f"✅ API密钥: {api_key[:20]}...")
    print(f"✅ API端点: {base_url}")
    print(f"✅ 配置文件: {env_file}")
    print("\n现在可以使用真实API运行项目了！")
    print("运行: python3 main.py")

if __name__ == "__main__":
    setup_api_config()


