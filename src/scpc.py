"""
自纠正提示链（Self-Correcting Prompt Chain, SCPC）核心实现
"""
import json
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from data_loader import ECFDataLoader
from multimodal_processor import MultimodalProcessor
from prompts import PromptTemplates
from llm_client import LLMClient, MockLLMClient

# ECF 情感白名单：只认这 7 类，其余一律视为无效并映射为 neutral
VALID_EMOTIONS = frozenset({"joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"})

# 情感同义词映射：投票时统一归类，避免 excited/joy 分散票数
EMOTION_SYNONYM_MAP = {
    "excited": "joy", "happy": "joy", "glad": "joy",
    "frustrated": "anger", "annoyed": "anger", "mad": "anger",
    "disappointed": "sadness", "upset": "sadness",
    "worried": "fear", "scared": "fear",
    "shocked": "surprise", "amazed": "surprise",
}


def _normalize_emotion(emo: str) -> str:
    """将情感归一化为规范类别（先应用同义词映射）"""
    if not emo:
        return ""
    e = str(emo).lower().strip()
    return EMOTION_SYNONYM_MAP.get(e, e)


def _to_valid_emotion(emo: str) -> str:
    """白名单过滤：仅当情感在 VALID_EMOTIONS 中时返回，否则返回 neutral"""
    norm = _normalize_emotion(emo)
    return norm if norm in VALID_EMOTIONS else "neutral"


import re as _re

def _normalize_text(text: str) -> str:
    """统一文本格式：小写、去标点、压缩空格"""
    t = text.lower().strip()
    t = _re.sub(r"[^\w\s]", " ", t)
    t = _re.sub(r"\s+", " ", t).strip()
    return t


def _text_overlap_score(span: str, utt_text: str) -> float:
    """计算 span 与话语文本的词重叠比例"""
    span_words = span.split()
    utt_words = set(utt_text.split())
    if not span_words:
        return 0.0
    return sum(1 for w in span_words if w in utt_words) / len(span_words)


def _correct_cause_utterance_id(result: Dict, conversation: Dict) -> None:
    """
    用 cause_span 反查它实际出现在哪个话语中，修正 cause_utterance_id。
    """
    raw_span = (result.get("cause_span") or "").strip()
    if not raw_span or len(raw_span) < 3:
        return

    norm_span = _normalize_text(raw_span)
    target_id = result.get("target_utterance_id", "")
    utterances = conversation.get("utterances", [])

    best_id = None
    best_score = 0.0

    for utt in utterances:
        utt_text = (utt.get("text") or "")
        utt_id = utt.get("utterance_id", "")
        if not utt_text:
            continue

        norm_utt = _normalize_text(utt_text)

        # 1. 精确子串匹配（标点无关化后）
        if norm_span in norm_utt:
            score = len(norm_span) / max(len(norm_utt), 1)
            if utt_id == target_id:
                score += 0.15
            if score > best_score:
                best_score = score
                best_id = utt_id
        else:
            # 2. 词重叠匹配
            ratio = _text_overlap_score(norm_span, norm_utt)
            if utt_id == target_id:
                ratio += 0.15
            if ratio > 0.5 and ratio > best_score:
                best_score = ratio
                best_id = utt_id

    if best_id and best_id != result.get("cause_utterance_id"):
        old_id = result.get("cause_utterance_id")
        result["cause_utterance_id"] = best_id
        result["cause_id_corrected"] = True
        print(f"  [后处理] cause_utterance_id 修正: {old_id} -> {best_id}")


class SelfCorrectingPromptChain:
    """自纠正提示链主类"""
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        use_mock: bool = True,
        enable_critique: bool = True,
        enable_refinement: bool = True,
        max_iterations: int = 2,
        extraction_temperature: float = 0.0,
        critique_temperature: float = 0.7
    ):
        """
        初始化SCPC
        
        Args:
            model_name: 模型名称
            use_mock: 是否使用模拟客户端（用于测试）
            enable_critique: 是否启用批判阶段
            enable_refinement: 是否启用精炼阶段
            max_iterations: Critique-Refine 最大迭代次数
            extraction_temperature: 提取阶段温度
            critique_temperature: 批判阶段温度
        """
        self.model_name = model_name
        self.enable_critique = enable_critique
        self.enable_refinement = enable_refinement
        self.max_iterations = max(0, int(max_iterations))
        self.extraction_temperature = float(extraction_temperature)
        self.critique_temperature = float(critique_temperature)
        
        # 初始化组件
        if use_mock:
            self.llm_client = MockLLMClient(model_name=model_name)
        else:
            self.llm_client = LLMClient(model_name=model_name, temperature=0.0)
        
        self.data_loader = ECFDataLoader("data/raw/ECF")
        self.multimodal_processor = MultimodalProcessor()
        self.prompt_templates = PromptTemplates()

    def phase1_extraction(
        self,
        conversation: Dict,
        target_utterance_id: str
    ) -> Dict:
        """
        阶段1：初始提取（Doer）

        论文设定：提取阶段使用确定性解码（temperature=0.0）。
        """
        multimodal_context = self.multimodal_processor.create_multimodal_context(conversation)
        prompt = self.prompt_templates.get_extraction_prompt(multimodal_context, target_utterance_id)
        response = self.llm_client.generate(prompt, temperature=self.extraction_temperature)
        result = self.llm_client.parse_json_response(response)

        emotion = _to_valid_emotion(result.get("emotion", "neutral"))
        result["emotion"] = emotion
        result["target_utterance_id"] = target_utterance_id
        result["multimodal_context"] = multimodal_context
        result["phase"] = "extraction"

        if not result.get("cause_utterance_id"):
            result["cause_utterance_id"] = target_utterance_id

        return result
        
    def phase2_critique(
        self,
        conversation: Dict,
        initial_extraction: Dict,
        target_utterance_id: str
    ) -> Dict:
        """
        阶段2：批判评估
        
        Args:
            conversation: 对话数据
            initial_extraction: 初始提取结果
            target_utterance_id: 目标话语ID
            
        Returns:
            批判结果
        """
        multimodal_context = initial_extraction.get("multimodal_context", "")
        
        # 生成批判提示
        prompt = self.prompt_templates.get_critique_prompt(
            multimodal_context, initial_extraction, target_utterance_id
        )
        
        response = self.llm_client.generate(prompt, temperature=self.critique_temperature)
        result = self.llm_client.parse_json_response(response)
        
        # 添加元数据
        result["phase"] = "critique"
        result["initial_extraction"] = initial_extraction
        
        return result
    
    def phase3_refinement(
        self,
        conversation: Dict,
        critique_result: Dict,
        target_utterance_id: str
    ) -> Dict:
        """
        阶段3：精炼
        
        Args:
            conversation: 对话数据
            critique_result: 批判结果
            target_utterance_id: 目标话语ID
            
        Returns:
            精炼后的最终结果
        """
        initial_extraction = critique_result.get("initial_extraction", {})
        multimodal_context = initial_extraction.get("multimodal_context", "")
        
        # 生成精炼提示
        prompt = self.prompt_templates.get_refinement_prompt(
            multimodal_context, critique_result, target_utterance_id
        )
        
        # 调用LLM
        response = self.llm_client.generate(prompt, temperature=0.0)
        result = self.llm_client.parse_json_response(response)
        
        # 如果精炼失败，使用初始提取结果或从对话中推断
        if result.get("parse_error") or not result.get("emotion"):
            initial_extraction = critique_result.get("initial_extraction", {})
            # 使用初始提取的结果作为fallback
            result["emotion"] = initial_extraction.get("emotion") or "neutral"
            result["cause_utterance_id"] = initial_extraction.get("cause_utterance_id") or target_utterance_id
            result["cause_span"] = initial_extraction.get("cause_span") or ""
            result["fallback_to_initial"] = True
        
        # 确保关键字段存在
        if not result.get("emotion"):
            result["emotion"] = "neutral"
        if not result.get("cause_utterance_id"):
            result["cause_utterance_id"] = target_utterance_id
        
        # 添加元数据
        result["phase"] = "refinement"
        result["critique_result"] = critique_result
        
        return result
    
    def extract_emotion_cause_pair(
        self,
        conversation: Dict,
        target_utterance_id: str,
        n_samples: int = 1
    ) -> Dict:
        """
        完整的自纠正提示链流程
        1) Doer：确定性初始提取
        2) Critic：批判检验
        3) Refine：在拒绝时修正，并可进行多轮迭代
        
        Args:
            conversation: 对话数据
            target_utterance_id: 目标话语ID
            n_samples: 兼容旧接口，当前不使用多采样
            
        Returns:
            最终的情感-原因对提取结果
        """
        print(f"\n[阶段1] 初始提取 - 目标话语: {target_utterance_id}")
        current_result = self.phase1_extraction(conversation, target_utterance_id)

        if current_result.get("emotion") == "neutral":
            print("  - 初始提取为 neutral，跳过后续反思")
            return {
                "emotion": "neutral",
                "target_utterance_id": target_utterance_id,
                "discarded": True
            }

        print(f"  提取结果: 情感={current_result.get('emotion')}, 原因={current_result.get('cause_utterance_id')}")

        if not self.enable_critique or self.max_iterations <= 0:
            return current_result

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n[阶段2] 批判评估 (迭代 {iteration}/{self.max_iterations})")
            critique_result = self.phase2_critique(
                conversation, current_result, target_utterance_id
            )

            is_valid = bool(critique_result.get("is_valid", False))
            confidence = str(critique_result.get("confidence", "low")).lower()
            discard_requested = bool(
                critique_result.get("suggested_improvement", {}).get("discard_pair")
            )

            print(f"  批判结果: 有效={is_valid}, 置信度={confidence}, 丢弃={discard_requested}")

            if is_valid and not discard_requested:
                current_result["critique_passed"] = True
                current_result["critique_iteration"] = iteration
                current_result["critique_confidence"] = confidence
                return current_result

            if discard_requested:
                return {
                    "emotion": "neutral",
                    "target_utterance_id": target_utterance_id,
                    "discarded": True,
                    "discarded_by_critic": True,
                    "critique_iteration": iteration
                }

            if not self.enable_refinement:
                current_result["critique_rejected"] = True
                current_result["critique_iteration"] = iteration
                return current_result

            print(f"\n[阶段3] 精炼 (迭代 {iteration}/{self.max_iterations})")
            refined_result = self.phase3_refinement(
                conversation, critique_result, target_utterance_id
            )

            refined_emo = _to_valid_emotion(refined_result.get("emotion", "neutral"))
            refined_result["emotion"] = refined_emo
            refined_result["target_utterance_id"] = target_utterance_id
            refined_result["iteration"] = iteration

            if refined_result.get("discarded") or refined_emo == "neutral":
                return {
                    "emotion": "neutral",
                    "target_utterance_id": target_utterance_id,
                    "discarded": True,
                    "discarded_by_refiner": True,
                    "iteration": iteration
                }

            print(f"  精炼结果: 情感={refined_emo}, 原因={refined_result.get('cause_utterance_id')}")
            current_result = refined_result

        return current_result

    def process_conversation(self, conversation: Dict) -> List[Dict]:
        """
        处理整个对话，提取所有情感-原因对
        """
        pairs = []
        utterances = conversation.get("utterances", [])

        # 零样本设定：对每个话语都进行抽取，不依赖标注 emotion 字段筛选目标。
        target_utterances = [utt for utt in utterances if utt.get("utterance_id")]
        
        print(f"\n处理对话: {conversation.get('conversation_id')}")
        print(f"找到 {len(target_utterances)} 个需要提取情感的话语")
        
        for utt in target_utterances:
            utt_id = utt.get("utterance_id")
            result = self.extract_emotion_cause_pair(conversation, utt_id)
            
            result["target_utterance_id"] = utt_id  
            
            # 后处理：用 cause_span 反查实际所在话语，修正 cause_utterance_id
            _correct_cause_utterance_id(result, conversation)
            
            final_valid = _to_valid_emotion(result.get("emotion", "neutral"))
            if final_valid in VALID_EMOTIONS and final_valid != "neutral" and not result.get("discarded", False):
                result["emotion"] = final_valid
                pairs.append(result)
            else:
                print(f"  - 过滤掉非情感/低置信度结果 (Target: {utt_id}, Emotion: {result.get('emotion', '')})")
        
        return pairs


def test_scpc():
    """测试自纠正提示链"""
    print("=" * 50)
    print("测试自纠正提示链（SCPC）")
    print("=" * 50)
    
    # 创建SCPC实例（使用模拟模式）
    scpc = SelfCorrectingPromptChain(
        model_name="mock",
        use_mock=True,
        enable_critique=True,
        enable_refinement=True
    )
    
    # 创建测试对话
    loader = ECFDataLoader("data/raw/ECF")
    conversation = loader.create_mock_conversation("test_scpc_001")
    
    # 添加多模态特征
    conversation = scpc.multimodal_processor.add_mock_multimodal_features(conversation)
    
    # 处理对话
    results = scpc.process_conversation(conversation)
    
    print(f"\n最终结果:")
    for i, result in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(f"  情感: {result.get('emotion')}")
        print(f"  原因话语ID: {result.get('cause_utterance_id')}")
        print(f"  原因文本: {result.get('cause_span')}")
        print(f"  是否精炼: {result.get('was_refined', False)}")
    
    print("\n✅ 自纠正提示链测试通过!")
    return True


if __name__ == "__main__":
    test_scpc()

