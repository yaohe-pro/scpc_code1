"""
数据加载器模块
支持加载ECF数据集或模拟数据用于测试
"""
import os
import re
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None


def _safe_skew(a) -> float:
    """计算偏度，避免除零。"""
    std = float(np.std(a))
    if std < 1e-10:
        return 0.0
    mean = float(np.mean(a))
    return float(np.mean(((a - mean) / std) ** 3))


@dataclass
class _Embeddings:
    audio: "Optional[object]" = None  # numpy.ndarray
    video: "Optional[object]" = None  # numpy.ndarray


class ECFDataLoader:
    """ECF数据集加载器"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.conversations = []
        self._offset_map: Optional[Dict[int, int]] = None
        
    def load_conversation(self, conversation_id: str) -> Optional[Dict]:
        conv_file = self.data_path / f"{conversation_id}.json"
        if conv_file.exists():
            with open(conv_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def load_all_conversations(self, split: str = "test") -> List[Dict]:
        split_path = self.data_path / split
        if not split_path.exists():
            print(f"警告: {split_path} 不存在，返回空列表")
            return []
        
        conversations = []
        for conv_file in split_path.glob("*.json"):
            with open(conv_file, 'r', encoding='utf-8') as f:
                obj = json.load(f)
                if isinstance(obj, list):
                    conversations.extend([c for c in obj if isinstance(c, dict)])
                elif isinstance(obj, dict):
                    conversations.append(obj)
                else:
                    continue

        conversations = self._maybe_attach_embeddings(conversations, split=split)
        return conversations

    # ------------------------------------------------------------------
    # Embedding 对齐挂载（修正版）
    # ------------------------------------------------------------------

    def _maybe_attach_embeddings(self, conversations: List[Dict], split: str) -> List[Dict]:
        """
        按 conversation_ID 的全局偏移正确地把 embedding 挂载到对应话语上。

        embedding 文件按 conversation_ID 升序（1,2,...,1374）排列，覆盖 train+val+test 全量。
        必须先读取所有 split 的原始 JSON 构建全局偏移表，
        才能为任意单个 split 的对话定位到正确的 embedding 行。
        """
        if np is None:
            return conversations

        emb = self._load_local_embeddings()
        if emb.audio is None and emb.video is None:
            return conversations

        offset_map = self._build_embedding_offset_map()
        if not offset_map:
            print("⚠️  无法构建 embedding 全局偏移映射，跳过 embedding 挂载")
            return conversations

        audio_arr = emb.audio
        video_arr = emb.video
        audio_len = int(getattr(audio_arr, "shape", [0])[0]) if audio_arr is not None else 0
        video_len = int(getattr(video_arr, "shape", [0])[0]) if video_arr is not None else 0

        attached = 0
        skipped = 0
        for conv in conversations:
            cid_num = self._get_numeric_conv_id(conv)
            if cid_num is None or cid_num not in offset_map:
                skipped += 1
                continue

            start_idx = offset_map[cid_num]
            utterances = conv.get("utterances", []) or []
            for i, utt in enumerate(utterances):
                idx = start_idx + i

                if audio_arr is not None and idx < audio_len:
                    a = audio_arr[idx]
                    utt["audio_features"] = self._compute_audio_stats(a)

                if video_arr is not None and idx < video_len:
                    v = video_arr[idx]
                    utt["visual_features"] = self._compute_video_stats(v)

            attached += 1

        print(f"✅ Embedding 对齐挂载: {attached} 对话成功, {skipped} 对话跳过 (split={split})")
        return conversations

    def _build_embedding_offset_map(self) -> Dict[int, int]:
        """
        读取 train.json / dev.json / test.json 三个原始文件，
        按 conversation_ID 升序排列，计算每个对话在 embedding 数组中的起始行号。
        结果会缓存，不重复读取。
        """
        if self._offset_map is not None:
            return self._offset_map

        bulk_files = [
            self.data_path / "train" / "train.json",
            self.data_path / "validation" / "dev.json",
            self.data_path / "test" / "test.json",
        ]

        id_to_n_utts: Dict[int, int] = {}
        for bf in bulk_files:
            if not bf.exists():
                continue
            try:
                with open(bf, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            if not isinstance(data, list):
                continue
            for conv in data:
                cid = conv.get("conversation_ID")
                if cid is None:
                    continue
                try:
                    cid = int(cid)
                except (ValueError, TypeError):
                    continue
                id_to_n_utts[cid] = len(conv.get("conversation", []))

        if not id_to_n_utts:
            self._offset_map = {}
            return self._offset_map

        offset_map: Dict[int, int] = {}
        cursor = 0
        for cid in sorted(id_to_n_utts.keys()):
            offset_map[cid] = cursor
            cursor += id_to_n_utts[cid]

        print(f"📊 Embedding 偏移映射: {len(offset_map)} 对话, 总行数={cursor}")
        self._offset_map = offset_map
        return offset_map

    def _get_numeric_conv_id(self, conv: Dict) -> Optional[int]:
        """从对话的 metadata.original_id 或 conversation_id 后缀提取数字 ID。"""
        md = conv.get("metadata") or {}
        oid = md.get("original_id")
        if oid is not None:
            try:
                val = int(oid)
                if val < 10**6:
                    return val
            except (ValueError, TypeError):
                pass

        cid = str(conv.get("conversation_id", ""))
        m = re.search(r"(\d+)$", cid)
        if m:
            return int(m.group(1))

        return None

    @staticmethod
    def _compute_audio_stats(a) -> Dict:
        """从完整的 OpenSmile IS13 音频 embedding (6373-dim) 提取统计特征。"""
        try:
            abs_a = np.abs(a)
            nz = np.count_nonzero(a)
            dim = int(a.shape[0])
            return {
                "dimension": dim,
                "embedding": a.tolist()[:10],
                "stats": {
                    "energy": float(np.mean(abs_a)),
                    "energy_std": float(np.std(abs_a)),
                    "max_abs": float(np.max(abs_a)),
                    "variance": float(np.var(a)),
                    "nonzero_ratio": float(nz / dim),
                    "positive_ratio": float(np.sum(a > 0) / dim),
                    "q25": float(np.percentile(a, 25)),
                    "q75": float(np.percentile(a, 75)),
                    "skew": float(_safe_skew(a)),
                }
            }
        except Exception:
            return {"dimension": 0, "embedding": [], "stats": {}}

    @staticmethod
    def _compute_video_stats(v) -> Dict:
        """从完整的 VGG-Face fc6 视频 embedding (4096-dim) 提取统计特征。"""
        try:
            dim = int(v.shape[0])
            nz = np.count_nonzero(v)
            pos = v[v > 0]
            return {
                "dimension": dim,
                "embedding": v.tolist()[:10],
                "stats": {
                    "mean_activation": float(np.mean(v)),
                    "max_activation": float(np.max(v)),
                    "activation_std": float(np.std(v)),
                    "active_ratio": float(nz / dim),
                    "positive_mean": float(np.mean(pos)) if len(pos) > 0 else 0.0,
                    "high_act_ratio": float(np.sum(v > np.mean(v) + np.std(v)) / dim),
                }
            }
        except Exception:
            return {"dimension": 0, "embedding": [], "stats": {}}

    # ------------------------------------------------------------------
    # Embedding 文件查找
    # ------------------------------------------------------------------

    def _load_local_embeddings(self) -> _Embeddings:
        """在常见目录里查找并加载 embedding 文件。"""
        if np is None:
            return _Embeddings()

        candidates = [
            self.data_path,
            self.data_path.parent,
            self.data_path.parent.parent,
            Path.cwd() / "data",
            Path.cwd() / "data" / "raw" / "ECF",
        ]

        audio = None
        video = None
        for d in candidates:
            try:
                d = Path(d)
            except Exception:
                continue
            a_path = d / "audio_embedding_6373.npy"
            v_path = d / "video_embedding_4096.npy"
            if audio is None and a_path.exists():
                try:
                    audio = np.load(a_path)
                    print(f"✅ 已加载音频embedding: {a_path} shape={getattr(audio, 'shape', None)}")
                except Exception as e:
                    print(f"⚠️  读取音频embedding失败: {a_path} ({e})")
            if video is None and v_path.exists():
                try:
                    video = np.load(v_path)
                    print(f"✅ 已加载视频embedding: {v_path} shape={getattr(video, 'shape', None)}")
                except Exception as e:
                    print(f"⚠️  读取视频embedding失败: {v_path} ({e})")
            if audio is not None or video is not None:
                pass

        return _Embeddings(audio=audio, video=video)

    # ------------------------------------------------------------------
    # 其他方法
    # ------------------------------------------------------------------
    
    def create_mock_conversation(self, conversation_id: str = "mock_001") -> Dict:
        mock_data = {
            "conversation_id": conversation_id,
            "utterances": [
                {
                    "utterance_id": "u1",
                    "speaker": "Alice",
                    "text": "Hey, how was your day?",
                    "emotion": None,
                    "timestamp": 0.0
                },
                {
                    "utterance_id": "u2",
                    "speaker": "Bob",
                    "text": "It was okay, nothing special.",
                    "emotion": None,
                    "timestamp": 2.5
                },
                {
                    "utterance_id": "u3",
                    "speaker": "Alice",
                    "text": "Did you hear about the promotion?",
                    "emotion": None,
                    "timestamp": 5.0
                },
                {
                    "utterance_id": "u4",
                    "speaker": "Bob",
                    "text": "No, what happened?",
                    "emotion": None,
                    "timestamp": 7.5
                },
                {
                    "utterance_id": "u5",
                    "speaker": "Alice",
                    "text": "I got it! I'm so excited!",
                    "emotion": "joy",
                    "emotion_cause": {
                        "utterance_id": "u3",
                        "span": "Did you hear about the promotion?",
                        "modality": "text"
                    },
                    "timestamp": 10.0
                }
            ],
            "video_path": None,
            "audio_path": None,
            "metadata": {
                "source": "mock",
                "num_utterances": 5
            }
        }
        return mock_data
    
    def get_emotion_cause_pairs(self, conversation: Dict) -> List[Dict]:
        pairs = []
        utterances = conversation.get("utterances", [])
        
        for utterance in utterances:
            if utterance.get("emotion") and utterance.get("emotion_cause"):
                span = utterance["emotion_cause"].get("span", "")
                m = re.match(r'^(\d+)_', span)
                if m:
                    cause_utterance_id = f"u{m.group(1)}"
                else:
                    cause_utterance_id = utterance["emotion_cause"]["utterance_id"]
                pairs.append({
                    "emotion": utterance["emotion"],
                    "cause_utterance_id": cause_utterance_id,
                    "cause_span": span,
                    "cause_modality": utterance["emotion_cause"].get("modality", "text"),
                    "target_utterance_id": utterance["utterance_id"]
                })
        
        return pairs
    
    def format_conversation_for_llm(self, conversation: Dict, include_visual: bool = False) -> str:
        formatted = []
        utterances = conversation.get("utterances", [])
        
        for utt in utterances:
            speaker = utt.get("speaker", "Unknown")
            text = utt.get("text", "")
            utt_id = utt.get("utterance_id", "")
            
            line = f"[{utt_id}] {speaker}: {text}"
            
            if include_visual and utt.get("visual_description"):
                line += f" [Visual: {utt['visual_description']}]"
            
            formatted.append(line)
        
        return "\n".join(formatted)


def test_data_loader():
    """测试数据加载器"""
    print("=" * 50)
    print("测试数据加载器")
    print("=" * 50)
    
    loader = ECFDataLoader("data/raw/ECF")
    
    mock_conv = loader.create_mock_conversation("test_001")
    print("\n1. 模拟对话数据:")
    print(f"   对话ID: {mock_conv['conversation_id']}")
    print(f"   话语数量: {len(mock_conv['utterances'])}")
    
    pairs = loader.get_emotion_cause_pairs(mock_conv)
    print(f"\n2. 提取的情感-原因对数量: {len(pairs)}")
    for i, pair in enumerate(pairs, 1):
        print(f"   对 {i}:")
        print(f"     情感: {pair['emotion']}")
        print(f"     原因话语ID: {pair['cause_utterance_id']}")
        print(f"     原因文本: {pair['cause_span']}")
    
    formatted = loader.format_conversation_for_llm(mock_conv)
    print(f"\n3. 格式化后的对话:")
    print(formatted)
    
    print("\n✅ 数据加载器测试通过!")
    return True


if __name__ == "__main__":
    test_data_loader()
