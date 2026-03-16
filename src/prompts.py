"""
提示模板模块
定义三个阶段使用的提示模板：提取、批判、精炼
"""
from typing import Dict, Optional


class PromptTemplates:
    """提示模板类"""
    
    @staticmethod
    def get_extraction_prompt(multimodal_context: str, target_utterance_id: str) -> str:
        prompt = f"""你是一个顶尖的情感因果标注专家。请对目标话语 [{target_utterance_id}] 进行情感标注并定位原因文本。

{multimodal_context}

**【多模态线索使用指南】**

上方提供了三种模态的信息：TEXT（文本）、AUDIO（语音特征）和 VISUAL（面部表情特征）。
请综合利用它们来判断情感：

- **文本为主，音频/视频为辅**：以文本语义为基础判断，但当文本情感模糊时，利用音频和视觉线索消歧。
- **音频线索解读**：
  · loud voice + dynamic pitch + very expressive → 强烈情感（joy / anger / surprise）
  · soft voice + flat pitch + restrained → 消沉或平静（sadness / neutral）
  · upward tonal tendency → 可能是兴奋或惊讶
  · downward tonal tendency → 可能是悲伤或疲惫
- **视觉线索解读**：
  · strong/complex facial expression → 有明显情绪
  · neutral/blank face + few active features → 平静或抑制情绪
  · 表情强度与文本情感应大体一致，矛盾时需谨慎判断
- 当音频和视觉都显示强烈情绪信号，但文本看似平淡，说话者可能在用语气和表情传达情绪，仍应标注相应情感。

**情感类别（必须且只能从以下 7 类中选择）：**

1. **joy** — 快乐、高兴、满足、兴奋、感激、得意、欣赏、期待。
   注意：感叹句（"Oh!", "Wow!", "Great!"）如果表达的是**正面情绪**，应标为 joy 而非 surprise。
   例："Oh that's wonderful!" → joy（正面感受）

2. **surprise** — **仅用于意料之外的事件**引发的震惊、难以置信、困惑。
   关键判断：说话者是否在对一个**出乎预料**的信息做反应？如果只是正面感慨，不算 surprise。
   例："You're getting married?! I had no idea!" → surprise（意外信息）

3. **sadness** — 悲伤、失望、沮丧、遗憾、无奈、心痛。

4. **anger** — 愤怒、不满、恼火、指责、烦躁。
   注意：不满和恼火应标为 anger 而非 disgust。disgust 仅用于反感、恶心。

5. **fear** — 恐惧、担心、焦虑、紧张不安。

6. **disgust** — 厌恶、反感、恶心、鄙视。仅用于对事物本身的生理或道德上的排斥。

7. **neutral** — 平静、无明显情绪波动。**仅在话语中完全没有情感表达时才使用**。
   如果你能感受到任何情感倾向（哪怕是微弱的），请标注具体情感而非 neutral。

禁止输出 positive、negative、confused、anxious 等任何其他词汇。

**【cause_utterance_id 的定义】**

cause_utterance_id = **原因文本片段 (cause_span) 实际出现在哪个话语里**。

规则：
- 情况 A：原因文本在目标话语自身中 → cause_utterance_id = 目标话语 ID。
    例：u5 说 "I got promoted! I'm thrilled!" → cause_utterance_id="u5", cause_span="I got promoted"

- 情况 B：原因文本在前面的话语中 → cause_utterance_id = 该话语 ID。
    例：u3 说 "You're fired" → u4 说 "What?!" → cause_utterance_id="u3", cause_span="You're fired"

- 请根据上下文证据选择最合理来源，不要使用固定比例先验。
- **cause_span 必须是 cause_utterance_id 指向的话语中的原文**。
- **时序约束**：cause_utterance_id <= 目标话语 ID。

**输出格式（严格 JSON，无额外文本）：**
{{
    "emotion": "7类之一",
    "cause_utterance_id": "包含原因文本的话语ID",
    "cause_span": "从该话语中摘取的原因文本",
    "reasoning": "简短判定依据（含所用多模态线索）"
}}"""
        return prompt
    
    @staticmethod
    def get_critique_prompt(
        multimodal_context: str,
        initial_extraction: Dict,
        target_utterance_id: str
    ) -> str:
        emotion = initial_extraction.get("emotion", "unknown")
        cause_id = initial_extraction.get("cause_utterance_id", "unknown")
        cause_span = initial_extraction.get("cause_span", "")
        reasoning = initial_extraction.get("reasoning", "")
        
        prompt = f"""你是情感因果审查专家。请审查以下标注结果。

**对话上下文：**
{multimodal_context}

**待审查标注：**
- 目标话语: [{target_utterance_id}]
- 情感: {emotion}
- 原因话语 ID: {cause_id}
- 原因文本: {cause_span}
- 推理: {reasoning}

**【重点审查】**

1. **多模态一致性**：标注的情感是否与该话语的音频特征（语调、响度）和视觉特征（面部表情）一致？
   - 如果标为 neutral 但音频显示 loud + expressive，视觉显示 strong expression，可能漏标了情感。
   - 如果标为 anger 但音频显示 soft + flat，视觉显示 neutral face，需要重新审视。

2. **joy vs surprise 区分**：如果标为 surprise，检查是否真的是"意外信息"引起的。如果只是正面感慨（"Oh great!", "Wow cool!"），应建议改为 joy。

3. **cause_utterance_id 验证**：cause_span 文本是否真的出现在 {cause_id} 指向的话语中？如果 cause_span 实际在目标话语 [{target_utterance_id}] 中，应建议修正。

4. **时序合法性**：cause_utterance_id 必须 <= 目标话语 ID。

5. **宽容原则（最重要）**：标注大概率是正确的。仅在有**确凿证据**证明标注错误时才建议修改。不确定时应设 is_valid=true。不要因为"可能有其他解释"就否定标注。discard_pair=true 仅在话语完全无情感时使用。

**输出格式（严格 JSON）：**
{{
    "is_valid": true/false,
    "confidence": "high/medium/low",
    "issues": ["具体问题"],
    "suggested_improvement": {{
        "emotion": "建议情感（7类之一）",
        "cause_utterance_id": "建议的原因话语ID",
        "discard_pair": false
    }},
    "critique_reasoning": "审查分析（不少于50字）"
}}"""
        return prompt
    
    @staticmethod
    def get_refinement_prompt(
        multimodal_context: str,
        critique_result: Dict,
        target_utterance_id: str
    ) -> str:
        issues = critique_result.get("issues", [])
        suggested = critique_result.get("suggested_improvement", {})
        
        prompt = f"""基于评审意见，给出最终情感-原因标注。

**对话上下文：**
{multimodal_context}

**评审发现的问题：**
{chr(10).join(f"- {issue}" for issue in issues) if issues else "- 无显著问题"}

**评审建议：**
- 建议情感: {suggested.get('emotion', '保持原样')}
- 建议原因话语 ID: {suggested.get('cause_utterance_id', '保持原样')}
- 是否丢弃: {suggested.get('discard_pair', False)}

**任务：**
综合评审意见和多模态信息，输出修正后的最终结果。

**重要原则：**
- 你的首要目标是**修正**而非**丢弃**。大多数情况下话语确实有情感，只是需要修正具体类别或原因定位。
- 仅当话语明确无任何情感表达时才设为 neutral（极少见）。
- 如果评审建议和原始标注有分歧，优先参考对话原文和多模态线索做独立判断。

**提醒：**
- emotion 必须是 joy/sadness/anger/fear/surprise/disgust/neutral 之一
- joy = 正面情绪（包括感叹），surprise = 仅限意外事件
- cause_utterance_id 是包含 cause_span 文本的那个话语的 ID
- 参考目标话语 [{target_utterance_id}] 对应的音频/视觉描述辅助最终判断

输出格式：
{{
    "emotion": "7类之一",
    "cause_utterance_id": "包含原因文本的话语ID",
    "cause_span": "原因文本片段",
    "cause_modality": "text/audio/visual",
    "reasoning": "最终判定依据",
    "discarded": true/false
}}"""
        
        return prompt
