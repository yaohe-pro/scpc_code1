#!/usr/bin/env python3
"""
转换ECF数据集为我们的格式
"""
import json
from pathlib import Path
from typing import Dict, List

def convert_ecf_to_our_format(input_file: Path, output_dir: Path, split: str):
    """转换ECF数据格式"""
    print(f"\n转换 {split} 分割: {input_file.name}")
    
    # 加载数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        data = [data]
    
    print(f"  找到 {len(data)} 个对话")
    
    converted = []
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
        for pair in emotion_cause_pairs:
            if len(pair) >= 2:
                emotion_utt_id = pair[0]  # 格式: "3_fear"
                cause_text = pair[1]  # 原因文本
                
                # 解析情感和话语ID
                parts = emotion_utt_id.split("_", 1)
                if len(parts) == 2:
                    try:
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
                    except ValueError:
                        pass
        
        conversation = {
            "conversation_id": f"ecf_{split}_{conv_id}",
            "utterances": utterances,
            "video_path": None,
            "audio_path": None,
            "metadata": {
                "source": "huggingface",
                "split": split,
                "original_id": conv_id,
                "num_utterances": len(utterances),
                "num_emotion_pairs": len(emotion_cause_pairs)
            }
        }
        
        converted.append(conversation)
        
        # 保存单个对话文件
        output_file = output_dir / f"ecf_{split}_{conv_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)
    
    print(f"  ✅ 转换完成: {len(converted)} 个对话")
    return converted

def main():
    """主函数"""
    print("=" * 70)
    print("转换ECF数据集格式")
    print("=" * 70)
    
    base_dir = Path("data/raw/ECF")
    output_base = Path("data/raw/ECF")
    
    splits = {
        "train": "train/train.json",
        "validation": "validation/dev.json",
        "test": "test/test.json"
    }
    
    total_converted = 0
    
    for split, file_path in splits.items():
        input_file = base_dir / file_path
        if not input_file.exists():
            print(f"⚠️  文件不存在: {input_file}")
            continue
        
        output_dir = output_base / split
        output_dir.mkdir(parents=True, exist_ok=True)
        
        converted = convert_ecf_to_our_format(input_file, output_dir, split)
        total_converted += len(converted)
    
    print("\n" + "=" * 70)
    print(f"✅ 转换完成！总共转换了 {total_converted} 个对话")
    print("=" * 70)
    print("\n数据已保存在:")
    print("  - data/raw/ECF/train/")
    print("  - data/raw/ECF/validation/")
    print("  - data/raw/ECF/test/")
    print("\n现在可以使用真实数据运行项目了！")
    print("  python3 main.py")

if __name__ == "__main__":
    main()

