"""
ECF数据集Hugging Face加载器
从Hugging Face加载真实的ECF数据集
"""
import os
import json
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


class ECFHuggingFaceLoader:
    """从Hugging Face加载ECF数据集"""
    
    def __init__(self, dataset_name: str = "NUSTM/ECF", cache_dir: Optional[str] = None):
        """
        初始化ECF Hugging Face加载器
        
        Args:
            dataset_name: Hugging Face数据集名称
            cache_dir: 缓存目录
        """
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir or "data/raw/ECF"
        self.dataset = None
        self.audio_embeddings = None
        self.video_embeddings = None
        
    def load_dataset(self):
        """加载数据集"""
        try:
            from datasets import load_dataset
            print(f"正在从Hugging Face加载数据集: {self.dataset_name}")
            print("这可能需要几分钟，请耐心等待...")
            print("注意: 该数据集在Hugging Face上有已知的格式问题，我们使用streaming模式绕过...")
            
            # 使用streaming模式，这样可以绕过格式验证问题
            try:
                print("尝试使用streaming模式加载...")
                self.dataset = load_dataset(
                    self.dataset_name, 
                    cache_dir=self.cache_dir,
                    streaming=True  # 使用streaming模式
                )
                print("✅ 使用streaming模式加载成功!")
                
                # 将streaming数据集转换为普通数据集
                print("正在转换为普通格式...")
                self.dataset = {
                    split: list(ds) for split, ds in self.dataset.items()
                }
                print("✅ 转换完成!")
                
            except Exception as e1:
                print(f"Streaming模式失败: {e1}")
                print("尝试直接从缓存文件读取...")
                
                # 尝试从缓存读取
                cache_path = Path(self.cache_dir) / "NUSTM___ecf" / "default"
                if cache_path.exists():
                    print(f"找到缓存目录: {cache_path}")
                    # 尝试直接读取parquet文件
                    import glob
                    parquet_files = list(cache_path.rglob("*.parquet"))
                    if parquet_files:
                        print(f"找到 {len(parquet_files)} 个parquet文件")
                        from datasets import Dataset
                        datasets_dict = {}
                        for pq_file in parquet_files:
                            split_name = pq_file.parent.name
                            if split_name not in datasets_dict:
                                datasets_dict[split_name] = []
                            # 这里需要手动处理，因为格式问题
                        # 如果缓存不可用，返回False让用户手动下载
                        print("⚠️  缓存文件格式不兼容，建议使用手动下载")
                        return False
                
                # 最后尝试：直接加载，接受可能的错误
                print("尝试直接加载（可能遇到格式警告）...")
                self.dataset = load_dataset(
                    self.dataset_name,
                    cache_dir=self.cache_dir,
                    ignore_verifications=True  # 忽略验证
                )
            
            print("✅ 数据集加载成功!")
            if isinstance(self.dataset, dict):
                print(f"可用分割: {list(self.dataset.keys())}")
            return True
        except ImportError:
            print("❌ 错误: 需要安装datasets库")
            print("请运行: pip install datasets")
            return False
        except Exception as e:
            print(f"❌ 加载数据集时出错: {e}")
            print("\n由于数据集格式问题，建议使用以下方法之一：")
            print("1. 使用Hugging Face CLI: huggingface-cli download NUSTM/ECF")
            print("2. 从网站直接下载: https://huggingface.co/datasets/NUSTM/ECF")
            print("3. 查看 DATA_DOWNLOAD.md 了解详细步骤")
            return False
    
    def load_embeddings(self, embeddings_dir: Optional[str] = None):
        """
        加载音频和视频特征嵌入
        
        Args:
            embeddings_dir: 嵌入文件目录（如果已下载）
        """
        embeddings_dir = embeddings_dir or self.cache_dir
        
        # 尝试加载音频嵌入
        audio_path = Path(embeddings_dir) / "audio_embedding_6373.npy"
        if audio_path.exists():
            print(f"加载音频特征: {audio_path}")
            self.audio_embeddings = np.load(audio_path)
            print(f"  音频特征形状: {self.audio_embeddings.shape}")
        else:
            print(f"⚠️  音频特征文件未找到: {audio_path}")
            print("  提示: 可以从数据集页面下载 audio_embedding_6373.npy")
        
        # 尝试加载视频嵌入
        video_path = Path(embeddings_dir) / "video_embedding_4096.npy"
        if video_path.exists():
            print(f"加载视频特征: {video_path}")
            self.video_embeddings = np.load(video_path)
            print(f"  视频特征形状: {self.video_embeddings.shape}")
        else:
            print(f"⚠️  视频特征文件未找到: {video_path}")
            print("  提示: 可以从数据集页面下载 video_embedding_4096.npy")
    
    def convert_to_our_format(self, split: str = "test") -> List[Dict]:
        """
        将Hugging Face格式转换为我们的数据格式
        
        Args:
            split: 数据分割 ('train', 'validation', 'test')
            
        Returns:
            转换后的对话列表
        """
        if self.dataset is None:
            print("❌ 错误: 请先调用 load_dataset()")
            return []
        
        if split not in self.dataset:
            print(f"❌ 错误: 分割 '{split}' 不存在")
            print(f"可用分割: {list(self.dataset.keys())}")
            return []
        
        conversations = []
        data = self.dataset[split]
        
        print(f"\n转换 {split} 分割...")
        print(f"总对话数: {len(data)}")
        
        for idx, item in enumerate(data):
            conv_id = item.get("conversation_ID", idx + 1)
            conversation_list = item.get("conversation", [])
            emotion_cause_pairs = item.get("emotion-cause_pairs", [])
            
            # 转换话语格式
            utterances = []
            for utt in conversation_list:
                utterances.append({
                    "utterance_id": f"u{utt.get('utterance_ID', len(utterances) + 1)}",
                    "speaker": utt.get("speaker", "Unknown"),
                    "text": utt.get("text", ""),
                    "emotion": utt.get("emotion", None),
                    "timestamp": len(utterances) * 2.5  # 估算时间戳
                })
            
            # 添加情感-原因对信息
            # 创建情感-原因对映射
            pair_map = {}
            for pair in emotion_cause_pairs:
                if len(pair) >= 2:
                    emotion_utt_id = pair[0]  # 格式: "3_fear"
                    cause_text = pair[1]  # 原因文本
                    
                    # 解析情感和话语ID
                    parts = emotion_utt_id.split("_", 1)
                    if len(parts) == 2:
                        utt_id_num = int(parts[0])
                        emotion = parts[1]
                        
                        # 找到对应的话语并添加情感-原因信息
                        for utt in utterances:
                            if utt["utterance_id"] == f"u{utt_id_num}":
                                utt["emotion"] = emotion
                                utt["emotion_cause"] = {
                                    "utterance_id": f"u{utt_id_num}",  # 自因情况
                                    "span": cause_text,
                                    "modality": "text"
                                }
                                break
            
            conversation = {
                "conversation_id": f"ecf_{split}_{conv_id}",
                "utterances": utterances,
                "video_path": None,  # 视频文件需要单独下载
                "audio_path": None,   # 音频文件需要单独下载
                "metadata": {
                    "source": "huggingface",
                    "split": split,
                    "original_id": conv_id,
                    "num_utterances": len(utterances),
                    "num_emotion_pairs": len(emotion_cause_pairs)
                }
            }
            
            conversations.append(conversation)
        
        print(f"✅ 转换完成: {len(conversations)} 个对话")
        return conversations
    
    def add_embedding_features(self, conversation: Dict, utterance_index: int) -> Dict:
        """
        为对话添加嵌入特征
        
        Args:
            conversation: 对话数据
            utterance_index: 对话在数据集中的索引
            
        Returns:
            添加了嵌入特征的对话
        """
        utterances = conversation.get("utterances", [])
        
        for i, utt in enumerate(utterances):
            global_idx = utterance_index * len(utterances) + i
            
            # 添加音频特征
            if self.audio_embeddings is not None and global_idx < len(self.audio_embeddings):
                audio_feat = self.audio_embeddings[global_idx]
                # 将高维特征转换为描述性文本（简化处理）
                utt["audio_features"] = {
                    "embedding": audio_feat.tolist()[:10],  # 只保存前10维用于显示
                    "dimension": len(audio_feat),
                    "description": f"6373维音频特征向量"
                }
            
            # 添加视频特征
            if self.video_embeddings is not None and global_idx < len(self.video_embeddings):
                video_feat = self.video_embeddings[global_idx]
                utt["visual_features"] = {
                    "embedding": video_feat.tolist()[:10],  # 只保存前10维用于显示
                    "dimension": len(video_feat),
                    "description": f"4096维视觉特征向量"
                }
        
        return conversation
    
    def save_conversations(self, conversations: List[Dict], output_dir: str, split: str):
        """
        保存转换后的对话到JSON文件
        
        Args:
            conversations: 对话列表
            output_dir: 输出目录
            split: 数据分割名称
        """
        output_path = Path(output_dir) / split
        output_path.mkdir(parents=True, exist_ok=True)
        
        for conv in conversations:
            conv_id = conv["conversation_id"]
            output_file = output_path / f"{conv_id}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(conv, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 已保存 {len(conversations)} 个对话到 {output_path}")


def download_and_convert_ecf():
    """下载并转换ECF数据集"""
    print("=" * 60)
    print("ECF数据集下载和转换工具")
    print("=" * 60)
    
    loader = ECFHuggingFaceLoader()
    
    # 加载数据集
    if not loader.load_dataset():
        return False
    
    # 加载嵌入特征（如果可用）
    loader.load_embeddings()
    
    # 转换各个分割
    output_dir = "data/raw/ECF"
    for split in ["train", "validation", "test"]:
        conversations = loader.convert_to_our_format(split)
        if conversations:
            loader.save_conversations(conversations, output_dir, split)
    
    print("\n" + "=" * 60)
    print("✅ 数据集下载和转换完成!")
    print("=" * 60)
    print("\n注意:")
    print("1. 音频和视频特征文件需要从数据集页面单独下载")
    print("2. 下载链接: https://huggingface.co/datasets/NUSTM/ECF")
    print("3. 将下载的 .npy 文件放在 data/raw/ECF/ 目录下")
    
    return True


if __name__ == "__main__":
    download_and_convert_ecf()

