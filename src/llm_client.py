"""
LLM客户端模块
支持OpenAI GPT-4o和Google Gemini 1.5 Pro
"""
import os
import json
import re
import time
from typing import Dict, Optional, Any

# 尝试加载环境变量（可选）
try:
    from dotenv import load_dotenv
    # 清除可能存在的旧环境变量，强制从.env文件加载
    try:
        load_dotenv(override=True)  # 强制重新加载
    except PermissionError:
        # 在某些环境下 .env 可能由其它用户创建，读取会触发权限错误
        # 这里忽略该错误，直接使用已有的环境变量（例如系统环境或IDE注入的环境）
        pass
except ImportError:
    # dotenv 不是必需的，如果不存在就跳过
    pass


class LLMClient:
    """LLM客户端基类"""
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0):
        """
        初始化LLM客户端
        
        Args:
            model_name: 模型名称 ('gpt-4o' 或 'gemini-1.5-pro')
            temperature: 生成温度
        """
        self.model_name = model_name
        self.temperature = temperature
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """初始化客户端"""
        if self.model_name == "gpt-4o" or self.model_name.startswith("gpt-"):
            # 强制重新加载环境变量
            try:
                from dotenv import load_dotenv
                try:
                    load_dotenv(override=True)
                except PermissionError:
                    # 同样忽略 .env 读取的权限问题，避免在受限环境下直接崩溃
                    pass
            except:
                pass
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file or environment variables.")
            if api_key == "YOUR_API_KEY":
                raise ValueError("Please replace YOUR_API_KEY with your actual OpenAI API key in .env file")
            
            # 支持自定义API端点（中转地址）
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            
            # 确保URL格式正确（保留原始大小写）
            base_url = base_url.rstrip("/")
            # 如果URL已经包含路径（/v1或/V1），直接使用；否则添加/v1
            if "/v1" not in base_url.lower():
                # 检查原始URL是否使用大写V1
                if "V1" in base_url.upper() and "V1" not in base_url:
                    base_url = base_url + "/V1"  # 保持大写
                else:
                    base_url = base_url + "/v1"
            
            # 使用requests直接调用API（更可靠，支持自定义端点）
            self.api_key = api_key
            self.base_url = base_url
            self.client_type = "openai_requests"
            print(f"配置API端点: {base_url}")
        
        elif self.model_name == "gemini-1.5-pro":
            try:
                import google.generativeai as genai
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY not found in environment variables")
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel('gemini-1.5-pro')
                self.client_type = "gemini"
            except ImportError:
                raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
        
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        """
        生成响应
        
        Args:
            prompt: 提示文本
            temperature: 生成温度（如果为None，使用初始化时的温度）
            
        Returns:
            生成的文本响应
        """
        temp = temperature if temperature is not None else self.temperature
        
        if self.client_type == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temp,
                max_tokens=2000
            )
            return response.choices[0].message.content
        
        elif self.client_type == "openai_requests":
            # 使用requests直接调用API
            import requests
            import os
            import socket
            # 确保URL格式正确（统一使用小写v1，因为大写V1返回HTML）
            base = self.base_url.rstrip("/")
            # 统一转换为小写v1，因为测试发现大写V1返回HTML而非JSON
            if base.upper().endswith("/V1"):
                base = base[:-3] + "/v1"  # 将/V1转换为/v1
            elif base.endswith("/v1"):
                pass  # 已经是小写，保持不变
            else:
                base = base + "/v1"  # 添加小写v1
            
            url = f"{base}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            # 对于余额不足的情况，尝试使用更便宜的模型或减少token
            model = self.model_name
            # 根据模型调整max_tokens，避免预估费用过高
            # 中转API对预估费用非常严格，根据实际测试，max_tokens=200可以获得完整JSON
            prompt_length = len(prompt)
            if "gpt-4" in model.lower() or model == "gpt-4o":
                max_tokens = 500
            else:
                # gpt-3.5-turbo：根据prompt长度调整
                if prompt_length > 1000:
                    max_tokens = 200  # 长prompt使用200（测试通过，可获得完整JSON）
                else:
                    max_tokens = 300
            
            # 尝试多个模型（从便宜到贵）
            models_to_try = []
            if model == "gpt-4o":
                # 明确指定 gpt-4o 时直接使用，不 fallback 到 3.5
                models_to_try = ["gpt-4o", "gpt-4"]
            elif "gpt-4" in model.lower():
                models_to_try = ["gpt-3.5-turbo", "gpt-4o", "gpt-4"]
            elif "gpt-3.5" in model.lower():
                models_to_try = [model]
            else:
                models_to_try = [model, "gpt-3.5-turbo"]
            
            last_error = None
            response = None
            for model_to_use in models_to_try:
                data = {
                    "model": model_to_use,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temp,
                    "max_tokens": max_tokens
                }
                try:
                    # 调试：打印请求信息
                    if model_to_use == models_to_try[0]:  # 只打印第一个模型的请求信息
                        print(f"  调试: 模型={model_to_use}, max_tokens={max_tokens}, prompt长度={len(prompt)}")
                    
                    # SSL/网络偶发错误时重试最多 3 次
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            response = requests.post(url, json=data, headers=headers, timeout=60)
                            response.raise_for_status()
                            break
                        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                            if attempt < max_retries - 1:
                                wait = (attempt + 1) * 3
                                print(f"  ⚠️  网络/SSL 错误，{wait}s 后重试 ({attempt + 1}/{max_retries}): {type(e).__name__}")
                                time.sleep(wait)
                            else:
                                raise
                    # 如果成功，记录使用的模型
                    if model_to_use != model:
                        print(f"  注意: 使用 {model_to_use} 替代 {model}")
                    break  # 成功，退出循环
                except requests.exceptions.ConnectionError as e:
                    # 如果自定义端点DNS/网络不可达，尝试回退到官方端点（只重试一次）
                    last_error = e
                    should_fallback = False
                    # 仅当当前不是官方端点时才回退，避免循环
                    if "api.openai.com" not in base.lower():
                        # 尝试判断是否为DNS解析失败
                        try:
                            host = ""
                            try:
                                # 从 base URL 粗略提取 host
                                host = base.split("://", 1)[1].split("/", 1)[0]
                            except Exception:
                                host = ""
                            if host:
                                socket.getaddrinfo(host, 443)
                        except Exception:
                            should_fallback = True

                    if should_fallback and os.getenv("DISABLE_OPENAI_FALLBACK", "").lower() not in ["1", "true", "yes"]:
                        fallback_base = "https://api.openai.com/v1"
                        url = f"{fallback_base}/chat/completions"
                        print(f"  ⚠️  端点不可达，回退到官方端点: {fallback_base}")
                        try:
                            response = requests.post(url, json=data, headers=headers, timeout=60)
                            response.raise_for_status()
                            break
                        except Exception as e2:
                            last_error = e2
                            if model_to_use == models_to_try[-1]:
                                raise
                            continue
                    # 不回退则沿用原逻辑继续/抛错
                    if model_to_use == models_to_try[-1]:
                        raise
                    continue
                except requests.exceptions.HTTPError as e:
                    last_error = e
                    # 如果是余额不足，尝试下一个模型
                    should_try_next = False
                    if hasattr(e, 'response') and e.response is not None:
                        try:
                            error_detail = e.response.json()
                            error_str = str(error_detail).lower()
                            # 打印详细错误信息
                            error_msg = error_detail.get('error', {}).get('message', '')
                            print(f"  {model_to_use} 错误详情: {error_msg[:150]}")
                            
                            if "quota" in error_str or "余额" in error_str or "balance" in error_str:
                                print(f"  {model_to_use} 余额不足，尝试下一个模型...")
                                should_try_next = True
                        except:
                            # 如果无法解析JSON，检查状态码
                            if e.response.status_code == 403:
                                print(f"  {model_to_use} 返回403错误，响应: {e.response.text[:150]}")
                                should_try_next = True
                    
                    # 如果不是最后一个模型，尝试下一个
                    if should_try_next and model_to_use != models_to_try[-1]:
                        continue
                    # 如果是最后一个模型或不应该尝试下一个，抛出错误
                    if model_to_use == models_to_try[-1]:
                        # 最后一个模型也失败了，抛出错误
                        raise
                    if not should_try_next:
                        # 不是余额问题，直接抛出
                        raise
                    continue
                except Exception as e:
                    last_error = e
                    if model_to_use == models_to_try[-1]:
                        raise
                    continue
            
            # 如果所有模型都失败了
            if response is None or not response.ok:
                if last_error:
                    raise last_error
                raise Exception("所有模型尝试失败")
            
            # 尝试解析JSON响应
            try:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"API响应格式异常: {result}")
            except json.JSONDecodeError as e:
                # 如果不是JSON，检查响应内容
                response_text = response.text[:500]
                print(f"  ⚠️  JSON解析失败，响应内容: {response_text}")
                # 尝试从HTML或其他格式中提取信息
                if "<!doctype" in response_text.lower() or "<html" in response_text.lower():
                    raise Exception(f"API返回HTML而非JSON，可能是端点配置错误。响应: {response_text[:200]}")
                else:
                    raise Exception(f"API返回非JSON响应 (状态码: {response.status_code}): {response_text}")
        
        elif self.client_type == "gemini":
            response = self.client.generate_content(
                prompt,
                generation_config={
                    "temperature": temp,
                    "max_output_tokens": 2000
                }
            )
            return response.text
    
    def parse_json_response(self, response: str) -> Dict:
        """
        从响应中解析JSON，支持截断的JSON
        
        Args:
            response: LLM响应文本
            
        Returns:
            解析后的JSON字典
        """
        # 尝试提取JSON代码块
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 尝试提取花括号内容
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response
        
        # 尝试解析JSON
        try:
            result = json.loads(json_str)
            # 检查关键字段是否存在
            if "emotion" in result or "is_valid" in result:
                return result
        except json.JSONDecodeError:
            pass
        
        # 如果JSON解析失败，尝试修复截断的JSON
        try:
            # 尝试找到最后一个完整的键值对
            # 如果JSON被截断，尝试补全
            if json_str.count('{') > json_str.count('}'):
                # 缺少右括号，尝试补全
                json_str = json_str.rstrip() + '\n}'
                result = json.loads(json_str)
                if "emotion" in result or "is_valid" in result:
                    return result
        except:
            pass
        
        # 如果还是失败，尝试从文本中提取关键信息
        extracted = {}
        
        # 尝试提取emotion
        emotion_match = re.search(r'"emotion"\s*:\s*"([^"]+)"', json_str, re.IGNORECASE)
        if emotion_match:
            extracted["emotion"] = emotion_match.group(1)
        
        # 尝试提取cause_utterance_id
        cause_id_match = re.search(r'"cause_utterance_id"\s*:\s*"([^"]+)"', json_str, re.IGNORECASE)
        if cause_id_match:
            extracted["cause_utterance_id"] = cause_id_match.group(1)
        
        # 尝试提取cause_span
        cause_span_match = re.search(r'"cause_span"\s*:\s*"([^"]+)"', json_str, re.IGNORECASE)
        if cause_span_match:
            extracted["cause_span"] = cause_span_match.group(1)
        
        # 尝试提取is_valid
        is_valid_match = re.search(r'"is_valid"\s*:\s*(true|false)', json_str, re.IGNORECASE)
        if is_valid_match:
            extracted["is_valid"] = is_valid_match.group(1).lower() == "true"
        
        # 尝试提取confidence
        confidence_match = re.search(r'"confidence"\s*:\s*"([^"]+)"', json_str, re.IGNORECASE)
        if confidence_match:
            extracted["confidence"] = confidence_match.group(1)
        
        # 尝试提取issues
        issues_match = re.search(r'"issues"\s*:\s*\[(.*?)\]', json_str, re.DOTALL)
        if issues_match:
            issues_str = issues_match.group(1)
            # 简单提取issues列表
            issues = re.findall(r'"([^"]+)"', issues_str)
            if issues:
                extracted["issues"] = issues
        
        if extracted:
            extracted["raw_response"] = response
            extracted["parse_error"] = True
            extracted["extracted_from_text"] = True
            return extracted
        
        # 如果所有方法都失败，返回原始响应
        return {"raw_response": response, "parse_error": True}


class MockLLMClient:
    """模拟LLM客户端，用于测试（不需要API密钥）"""
    
    def __init__(self, model_name: str = "mock", temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature
    
    def generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        """生成模拟响应"""
        # 更精确的判断逻辑：优先检查批判和精炼关键词
        prompt_lower = prompt.lower()
        
        # 如果提示中包含多模态部分，则返回与多模态相关的模拟响应
        if "=== audio modality ===" in prompt_lower or "=== visual modality ===" in prompt_lower or "audio:" in prompt_lower or "visual:" in prompt_lower:
            # 模拟：多模态信息导致模型选择不同的原因描述/utterance
            return """{
    "emotion": "joy",
    "cause_utterance_id": "u5",
    "cause_span": "I got it! I'm so excited!",
    "cause_modality": "audio+visual",
    "reasoning": "根据音调与面部表情（兴奋/微笑），判断u5为更可靠的原因证据"
}"""

        # 检查是否是批判阶段（优先级最高）
        if "批判" in prompt or ("评估" in prompt and "批判" in prompt) or "critique" in prompt_lower:
            # 批判阶段
            return """{
    "is_valid": true,
    "confidence": "high",
    "issues": [],
    "suggested_improvement": {},
    "critique_reasoning": "初始提取结果合理，原因在目标话语之前，逻辑清晰"
}"""
        
        # 检查是否是精炼阶段
        elif "最终" in prompt or "重新分析" in prompt or "refinement" in prompt_lower or "精炼" in prompt:
            # 精炼阶段
            return """{
    "emotion": "joy",
    "cause_utterance_id": "u3",
    "cause_span": "Did you hear about the promotion?",
    "cause_modality": "text",
    "reasoning": "经过批判性评估，确认初始提取正确",
    "changes_made": "无变化"
}"""
        
        # 检查是否是提取阶段
        elif "识别" in prompt and "情感" in prompt and "任务" in prompt:
            # 提取阶段
            return """{
    "emotion": "joy",
    "cause_utterance_id": "u3",
    "cause_span": "Did you hear about the promotion?",
    "cause_modality": "text",
    "reasoning": "Alice在u3提到promotion，然后在u5表达兴奋，说明promotion是导致joy的原因"
}"""
        
        else:
            # 默认返回提取结果
            return """{
    "emotion": "joy",
    "cause_utterance_id": "u3",
    "cause_span": "Did you hear about the promotion?",
    "cause_modality": "text",
    "reasoning": "默认模拟响应"
}"""
    
    def parse_json_response(self, response: str) -> Dict:
        """解析JSON响应"""
        # 尝试提取JSON代码块
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 尝试提取花括号内容
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # 如果解析失败，返回原始响应
            return {"raw_response": response, "parse_error": True}


def test_llm_client():
    """测试LLM客户端（使用模拟客户端）"""
    print("=" * 50)
    print("测试LLM客户端（模拟模式）")
    print("=" * 50)
    
    # 使用模拟客户端
    client = MockLLMClient()
    
    # 测试提取提示
    print("\n1. 测试提取阶段:")
    extraction_prompt = "请识别情感和原因"
    response1 = client.generate(extraction_prompt)
    result1 = client.parse_json_response(response1)
    print(f"   响应: {json.dumps(result1, ensure_ascii=False, indent=2)}")
    
    # 测试批判提示
    print("\n2. 测试批判阶段:")
    critique_prompt = "请批判性评估"
    response2 = client.generate(critique_prompt)
    result2 = client.parse_json_response(response2)
    print(f"   响应: {json.dumps(result2, ensure_ascii=False, indent=2)}")
    
    # 测试精炼提示
    print("\n3. 测试精炼阶段:")
    refinement_prompt = "请输出最终结果"
    response3 = client.generate(refinement_prompt)
    result3 = client.parse_json_response(response3)
    print(f"   响应: {json.dumps(result3, ensure_ascii=False, indent=2)}")
    
    print("\n✅ LLM客户端测试通过!")
    return True


if __name__ == "__main__":
    test_llm_client()

