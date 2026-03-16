#!/usr/bin/env python3
"""
测试API连接
"""
import os
import sys
from pathlib import Path

# 添加src到路径
sys.path.append(str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

def test_api():
    """测试API连接"""
    print("=" * 60)
    print("测试API连接")
    print("=" * 60)
    
    # 清除缓存，强制重新加载
    if 'OPENAI_API_KEY' in os.environ:
        del os.environ['OPENAI_API_KEY']
    load_dotenv(override=True)  # 强制重新加载
    
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    if not api_key:
        print("❌ 错误: 未找到OPENAI_API_KEY")
        return False
    
    print(f"\n✅ API密钥: {api_key[:20]}...{api_key[-10:]}")
    print(f"✅ API端点: {base_url}")
    
    try:
        from src.llm_client import LLMClient
        
        print("\n初始化LLM客户端...")
        client = LLMClient(model_name="gpt-4o", temperature=0.0)
        
        print("发送测试请求...")
        test_prompt = "请用一句话回答：1+1等于几？"
        response = client.generate(test_prompt)
        
        print(f"\n✅ API连接成功!")
        print(f"测试提示: {test_prompt}")
        print(f"模型响应: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"\n❌ API连接失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_api()
    sys.exit(0 if success else 1)

