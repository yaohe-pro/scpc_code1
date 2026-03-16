"""
多模态处理器模块
处理文本、音频、视频数据，为LLM准备输入
"""
import os
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from multimodal_classifier import MultimodalClassifier


class MultimodalProcessor:
    """多模态数据处理器"""
    
    def __init__(self, video_fps: int = 1, enable_audio: bool = True, enable_video: bool = True):
        """
        初始化多模态处理器
        
        Args:
            video_fps: 视频采样帧率（每秒帧数）
            enable_audio: 是否启用音频处理
            enable_video: 是否启用视频处理
        """
        self.video_fps = video_fps
        self.enable_audio = enable_audio
        self.enable_video = enable_video
        self.classifier = MultimodalClassifier()

    @staticmethod
    def _describe_audio_from_file(audio_path: str) -> Optional[str]:
        """从原始音频提取简要韵律描述（音量/音调/语速）。"""
        try:
            import numpy as np
            import librosa
        except Exception:
            return None

        try:
            y, sr = librosa.load(audio_path, sr=None)
            if y is None or len(y) == 0:
                return None

            rms = librosa.feature.rms(y=y)[0]
            energy = float(np.mean(rms))

            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            valid = pitches[magnitudes > np.median(magnitudes)]
            pitch_hz = float(np.mean(valid)) if valid.size > 0 else 0.0

            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo = float(tempo) if tempo is not None else 0.0

            if energy >= 0.08:
                volume = "high"
            elif energy >= 0.03:
                volume = "medium"
            else:
                volume = "low"

            if pitch_hz >= 240:
                pitch = "high"
            elif pitch_hz >= 140:
                pitch = "medium"
            elif pitch_hz > 0:
                pitch = "low"
            else:
                pitch = "normal"

            if tempo >= 130 or (pitch_hz >= 240 and energy >= 0.06):
                tone = "excited"
            elif tempo <= 75 and energy <= 0.03:
                tone = "sad"
            elif energy >= 0.1 and pitch_hz >= 170:
                tone = "tense"
            else:
                tone = "neutral"

            return (
                f"tone={tone}, volume={volume}, pitch={pitch}, "
                f"tempo={tempo:.1f}bpm"
            )
        except Exception:
            return None

    def _extract_video_keyframe_descriptions(self, video_path: str) -> List[str]:
        """按 video_fps 对视频做均匀采样，输出关键帧外观描述。"""
        try:
            import cv2
            import numpy as np
        except Exception:
            return []

        if not video_path or not Path(video_path).exists():
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_step = max(1, int(round(src_fps / max(self.video_fps, 1))))
        max_keyframes = 24

        descriptions = []
        frame_idx = 0
        sampled = 0
        try:
            while sampled < max_keyframes:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_idx % frame_step != 0:
                    frame_idx += 1
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = float(np.mean(gray))
                contrast = float(np.std(gray))

                if brightness < 70:
                    light_desc = "dim"
                elif brightness > 170:
                    light_desc = "bright"
                else:
                    light_desc = "normal"

                if contrast > 55:
                    expr_desc = "high facial detail/variation"
                elif contrast > 35:
                    expr_desc = "moderate facial detail"
                else:
                    expr_desc = "flat or subtle expression cues"

                t_sec = frame_idx / max(src_fps, 1.0)
                descriptions.append(
                    f"t={t_sec:.1f}s, lighting={light_desc}, {expr_desc}"
                )

                sampled += 1
                frame_idx += 1
        finally:
            cap.release()

        return descriptions
        
    def process_text(self, conversation: Dict) -> str:
        """
        处理文本模态：提取对话文本
        
        Args:
            conversation: 对话数据
            
        Returns:
            格式化后的文本
        """
        utterances = conversation.get("utterances", [])
        text_lines = []
        
        for utt in utterances:
            speaker = utt.get("speaker", "Unknown")
            text = utt.get("text", "")
            utt_id = utt.get("utterance_id", "")
            timestamp = utt.get("timestamp", 0.0)
            
            line = f"[{utt_id}] {speaker} (t={timestamp:.1f}s): {text}"
            text_lines.append(line)
        
        return "\n".join(text_lines)
    
    def process_audio(self, conversation: Dict) -> Optional[str]:
        """
        处理音频模态：提取音频特征描述
        
        Args:
            conversation: 对话数据
            
        Returns:
            音频特征描述文本，如果不可用则返回None
        """
        if not self.enable_audio:
            return None
        
        audio_descriptions = []
        utterances = conversation.get("utterances", [])
        
        for utt in utterances:
            utt_id = utt.get("utterance_id", "")
            
            if utt.get("audio_features"):
                features = utt["audio_features"]
                if "tone" in features or "volume" in features or "pitch" in features:
                    desc = f"[{utt_id}] Audio: tone={features.get('tone', 'neutral')}, "
                    desc += f"volume={features.get('volume', 'medium')}, "
                    desc += f"pitch={features.get('pitch', 'normal')}"
                else:
                    stats = features.get("stats")
                    emb = features.get("embedding", [])
                    desc_text = self.classifier.describe_audio(emb if emb else None, stats)
                    desc = f"[{utt_id}] Audio: {desc_text}"
                audio_descriptions.append(desc)

        # 如果没有逐话语音频特征，尝试从原始音频文件提取韵律描述
        if not audio_descriptions and conversation.get("audio_path"):
            audio_path = str(conversation.get("audio_path"))
            audio_summary = self._describe_audio_from_file(audio_path)
            if audio_summary:
                for utt in utterances:
                    utt_id = utt.get("utterance_id", "")
                    audio_descriptions.append(f"[{utt_id}] Audio: {audio_summary}")
        
        if audio_descriptions:
            return "\n".join(audio_descriptions)
        return None
    
    def process_video(self, conversation: Dict) -> Optional[str]:
        """
        处理视频模态：提取关键帧描述
        
        Args:
            conversation: 对话数据
            
        Returns:
            视频帧描述文本，如果不可用则返回None
        """
        if not self.enable_video:
            return None
        
        video_descriptions = []
        utterances = conversation.get("utterances", [])
        
        for utt in utterances:
            utt_id = utt.get("utterance_id", "")
            
            # 如果有视觉描述，直接使用
            if utt.get("visual_description"):
                desc = f"[{utt_id}] Visual: {utt['visual_description']}"
                video_descriptions.append(desc)
            elif utt.get("visual_features"):
                vf = utt["visual_features"]
                stats = vf.get("stats")
                emb = vf.get("embedding", [])
                desc_text = self.classifier.describe_video(emb if emb else None, stats)
                desc = f"[{utt_id}] Visual: {desc_text}"
                video_descriptions.append(desc)
            # 如果有面部表情信息
            elif utt.get("facial_expression"):
                expr = utt["facial_expression"]
                desc = f"[{utt_id}] Facial Expression: {expr}"
                video_descriptions.append(desc)

        # 如果没有逐话语视觉特征，尝试从原始视频按 1fps 抽取关键帧描述
        if not video_descriptions and conversation.get("video_path"):
            video_path = str(conversation.get("video_path"))
            keyframe_desc = self._extract_video_keyframe_descriptions(video_path)
            for i, desc in enumerate(keyframe_desc, 1):
                video_descriptions.append(f"[kf{i}] Visual: {desc}")
        
        if video_descriptions:
            return "\n".join(video_descriptions)
        return None
    
    def create_multimodal_context(self, conversation: Dict) -> str:
        """
        创建多模态上下文，整合文本、音频、视频信息
        
        Args:
            conversation: 对话数据
            
        Returns:
            完整的多模态上下文字符串
        """
        parts = []
        
        # 1. 文本模态（必需）
        text_content = self.process_text(conversation)
        parts.append("=== TEXT MODALITY ===")
        parts.append(text_content)
        
        # 2. 音频模态（可选）
        audio_content = self.process_audio(conversation)
        if audio_content:
            parts.append("\n=== AUDIO MODALITY ===")
            parts.append(audio_content)
        
        # 3. 视频模态（可选）
        video_content = self.process_video(conversation)
        if video_content:
            parts.append("\n=== VISUAL MODALITY ===")
            parts.append(video_content)
        
        return "\n".join(parts)
    
    def add_mock_multimodal_features(self, conversation: Dict) -> Dict:
        """
        为模拟数据添加多模态特征（用于测试）
        
        Args:
            conversation: 对话数据
            
        Returns:
            添加了多模态特征的对话数据
        """
        utterances = conversation.get("utterances", [])
        
        # 为每个话语添加模拟的音频和视觉特征
        for utt in utterances:
            text = utt.get("text", "").lower()
            utt_id = utt.get("utterance_id", "")
            
            # 根据文本内容推断音频特征
            if "excited" in text or "!" in text:
                utt["audio_features"] = {
                    "tone": "excited",
                    "volume": "high",
                    "pitch": "high"
                }
                utt["visual_description"] = "smiling broadly, eyes wide open"
            elif "sad" in text or "disappointed" in text:
                utt["audio_features"] = {
                    "tone": "sad",
                    "volume": "low",
                    "pitch": "low"
                }
                utt["visual_description"] = "frowning, eyes downcast"
            elif "angry" in text:
                utt["audio_features"] = {
                    "tone": "angry",
                    "volume": "high",
                    "pitch": "medium"
                }
                utt["visual_description"] = "brows furrowed, mouth tight"
            else:
                utt["audio_features"] = {
                    "tone": "neutral",
                    "volume": "medium",
                    "pitch": "normal"
                }
                utt["visual_description"] = "neutral expression"
        
        return conversation


def test_multimodal_processor():
    """测试多模态处理器"""
    print("=" * 50)
    print("测试多模态处理器")
    print("=" * 50)
    
    # 创建处理器
    processor = MultimodalProcessor(video_fps=1, enable_audio=True, enable_video=True)
    
    # 创建模拟对话数据
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from data_loader import ECFDataLoader
    loader = ECFDataLoader("data/raw/ECF")
    mock_conv = loader.create_mock_conversation("test_002")
    
    # 添加多模态特征
    mock_conv = processor.add_mock_multimodal_features(mock_conv)
    
    print("\n1. 处理文本模态:")
    text = processor.process_text(mock_conv)
    print(text[:200] + "..." if len(text) > 200 else text)
    
    print("\n2. 处理音频模态:")
    audio = processor.process_audio(mock_conv)
    print(audio if audio else "无音频数据")
    
    print("\n3. 处理视频模态:")
    video = processor.process_video(mock_conv)
    print(video if video else "无视频数据")
    
    print("\n4. 创建完整的多模态上下文:")
    context = processor.create_multimodal_context(mock_conv)
    print(context[:300] + "..." if len(context) > 300 else context)
    
    print("\n✅ 多模态处理器测试通过!")
    return True


if __name__ == "__main__":
    test_multimodal_processor()

