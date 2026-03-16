"""
评估器模块
计算F1、准确率、Pair-F1等指标
"""
from typing import Dict, List, Tuple
import sys
import re
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from data_loader import ECFDataLoader

# 情感同义词映射：LLM 可能输出 excited/happy 等，需统一为规范类别以正确匹配
EMOTION_SYNONYM_MAP = {
    "excited": "joy", "happy": "joy", "glad": "joy",
    "frustrated": "anger", "annoyed": "anger", "mad": "anger",
    "disappointed": "sadness", "upset": "sadness",
    "worried": "fear", "scared": "fear",
    "shocked": "surprise", "amazed": "surprise",
}


class Evaluator:
    """评估器类"""
    
    def __init__(self):
        """初始化评估器"""
        self.data_loader = ECFDataLoader("data/raw/ECF")
    
    def compute_pair_f1(
        self,
        predicted_pairs: List[Dict],
        ground_truth_pairs: List[Dict]
    ) -> float:
        """
        计算Pair-F1（严格匹配情感和原因）
        
        Args:
            predicted_pairs: 预测的情感-原因对列表
            ground_truth_pairs: 真实的情感-原因对列表
            
        Returns:
            Pair-F1分数
        """
        if not ground_truth_pairs:
            return 0.0 if predicted_pairs else 1.0
        
        # 将配对转换为可比较的格式（使用模糊匹配）
        def _normalize_span_text(s: str) -> str:
            if not s:
                return ""
            s = str(s).lower().strip()
            # 去掉开头的数字前缀和下划线（如 "1_" 或 "10_"）
            s = re.sub(r'^\d+[_\s]*', '', s)
            # 移除多余空格
            s = re.sub(r'\s+', ' ', s).strip()
            # 移除首尾标点
            s = s.strip('.,!?;:\"\'')
            return s

        def pair_to_tuple(pair, fuzzy=True):
            emotion = pair.get("emotion", "").lower()
            cause_id = pair.get("cause_utterance_id", "")
            cause_span = pair.get("cause_span", "")
            
            # 情感标准化（处理相似情感）
            emotion_map = {
                "excited": "joy",
                "happy": "joy",
                "glad": "joy",
                "frustrated": "anger",
                "annoyed": "anger",
                "mad": "anger",
                "disappointed": "sadness",
                "upset": "sadness",
                "worried": "fear",
                "scared": "fear",
                "shocked": "surprise",
                "amazed": "surprise"
            }
            if emotion in emotion_map:
                emotion = emotion_map[emotion]

            # 原因文本清理（移除标点、数字前缀等）
            if fuzzy:
                cause_span = _normalize_span_text(cause_span)
            
            return (emotion, cause_id, cause_span)
        
        pred_set = set(pair_to_tuple(p, fuzzy=True) for p in predicted_pairs)
        gt_set = set(pair_to_tuple(p, fuzzy=True) for p in ground_truth_pairs)
        
        # 计算精确率、召回率、F1
        if not pred_set:
            precision = 0.0
            recall = 0.0
        else:
            tp = len(pred_set & gt_set)
            precision = tp / len(pred_set) if pred_set else 0.0
            recall = tp / len(gt_set) if gt_set else 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return f1
    
    def compute_emotion_accuracy(
        self,
        predicted_pairs: List[Dict],
        ground_truth_pairs: List[Dict]
    ) -> float:
        """
        计算情感准确率
        
        Args:
            predicted_pairs: 预测的情感-原因对列表
            ground_truth_pairs: 真实的情感-原因对列表
            
        Returns:
            情感准确率
        """
        if not ground_truth_pairs:
            return 0.0
        
        # 创建目标话语ID到情感的映射
        gt_emotions = {
            pair.get("target_utterance_id", ""): pair.get("emotion", "").lower()
            for pair in ground_truth_pairs
        }
        
        correct = 0
        total = len(gt_emotions)
        
        for pred_pair in predicted_pairs:
            target_id = pred_pair.get("target_utterance_id", "")
            pred_emotion = pred_pair.get("emotion", "").lower()
            
            if target_id in gt_emotions:
                if pred_emotion == gt_emotions[target_id]:
                    correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def compute_weighted_f1(
        self,
        predicted_pairs: List[Dict],
        ground_truth_pairs: List[Dict]
    ) -> float:
        """
        计算加权F1分数（考虑不同情感类别的分布）
        
        Args:
            predicted_pairs: 预测的情感-原因对列表
            ground_truth_pairs: 真实的情感-原因对列表
            
        Returns:
            加权F1分数
        """
        from collections import Counter
        
        # 统计真实标签中的情感分布
        gt_emotions = [pair.get("emotion", "").lower() for pair in ground_truth_pairs]
        emotion_counts = Counter(gt_emotions)
        total = len(gt_emotions)
        
        if total == 0:
            return 0.0
        
        # 为每个情感类别计算F1
        emotion_f1s = {}
        all_emotions = set(gt_emotions) | set(p.get("emotion", "").lower() for p in predicted_pairs)
        
        for emotion in all_emotions:
            # 提取该情感的预测和真实配对
            pred_emotion_pairs = [p for p in predicted_pairs if p.get("emotion", "").lower() == emotion]
            gt_emotion_pairs = [p for p in ground_truth_pairs if p.get("emotion", "").lower() == emotion]
            
            # 计算该情感的F1
            f1 = self.compute_pair_f1(pred_emotion_pairs, gt_emotion_pairs)
            weight = emotion_counts.get(emotion, 0) / total
            emotion_f1s[emotion] = (f1, weight)
        
        # 计算加权平均
        weighted_f1 = sum(f1 * weight for f1, weight in emotion_f1s.values())
        
        return weighted_f1
    
    def evaluate(
        self,
        predicted_pairs: List[Dict],
        ground_truth_pairs: List[Dict]
    ) -> Dict[str, Dict[str, float]]:
        """
        标准 MCECPE 评估 (EE, CE, Pair)
        
        Args:
            predicted_pairs: 预测的情感-原因对列表
            ground_truth_pairs: 真实的情感-原因对列表
            
        Returns:
            包含 EE, CE, Pair 的 P, R, F1 指标
        """
        def _normalize_utterance_id(uid: str) -> str:
            """标准化话语ID格式：如 '3' -> 'u3'，'u_3' -> 'u3'，与数据集一致"""
            if not uid:
                return uid
            uid = str(uid).strip()
            m = re.match(r'^u_?(\d+)$', uid, re.IGNORECASE)
            if m:
                return f"u{m.group(1)}"
            if uid.isdigit():
                return f"u{uid}"
            return uid

        def _normalize_pairs(pairs: List[Dict]) -> List[Dict]:
            """对 pairs 中的 ID、情感做标准化；情感同义词映射以正确匹配"""
            out = []
            for p in pairs:
                q = dict(p)
                q["target_utterance_id"] = _normalize_utterance_id(p.get("target_utterance_id", ""))
                q["cause_utterance_id"] = _normalize_utterance_id(p.get("cause_utterance_id", ""))
                emo = (p.get("emotion") or "").lower().strip()
                q["emotion"] = EMOTION_SYNONYM_MAP.get(emo, emo) if emo else ""
                out.append(q)
            return out

        # 0. 标准化 ID 格式（LLM 可能返回 "3" 而非 "u3"）
        predicted_pairs = _normalize_pairs(predicted_pairs)
        ground_truth_pairs = _normalize_pairs(ground_truth_pairs)

        # 额外：对 predicted_pairs 执行去重（基于 target/cause_id/emotion/normalized_span）
        seen = set()
        deduped_preds = []
        for p in predicted_pairs:
            span = p.get('cause_span', '')
            # 为去重使用规范化后的 span
            norm_span = re.sub(r'^\d+[_\s]*', '', str(span).lower()).strip()
            norm_span = re.sub(r'\s+', ' ', norm_span)
            key = (p.get('target_utterance_id',''), p.get('cause_utterance_id',''), p.get('emotion',''), norm_span)
            if key in seen:
                continue
            seen.add(key)
            deduped_preds.append(p)
        predicted_pairs = deduped_preds

        # 1. 提取集合用于比较（包含 conversation_id 防止跨对话去重）
        def _cid(p):
            return p.get("conversation_id", "")

        gt_ee = set((_cid(p), p["target_utterance_id"], p["emotion"]) for p in ground_truth_pairs)
        pred_ee = set((_cid(p), p["target_utterance_id"], p["emotion"]) for p in predicted_pairs)
        
        gt_ce = set((_cid(p), p["target_utterance_id"], p["cause_utterance_id"]) for p in ground_truth_pairs)
        pred_ce = set((_cid(p), p["target_utterance_id"], p["cause_utterance_id"]) for p in predicted_pairs)
        
        gt_pair = set((_cid(p), p["target_utterance_id"], p["emotion"], p["cause_utterance_id"]) for p in ground_truth_pairs)
        pred_pair = set((_cid(p), p["target_utterance_id"], p["emotion"], p["cause_utterance_id"]) for p in predicted_pairs)
        
        def calculate_prf1(pred_set, gt_set):
            tp = len(pred_set & gt_set)
            p = tp / len(pred_set) if len(pred_set) > 0 else 0.0
            r = tp / len(gt_set) if len(gt_set) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            return {"p": p, "r": r, "f1": f1}

        results = {
            "emotion_extraction": calculate_prf1(pred_ee, gt_ee),
            "cause_extraction": calculate_prf1(pred_ce, gt_ce),
            "pair_extraction": calculate_prf1(pred_pair, gt_pair)
        }
        
        # 2. 计算 w-avg. 6 和 w-avg. 4 (针对 ECPE 任务)
        # 情感类别定义 (根据论文通常使用的分类)
        emotions_6 = {"anger", "disgust", "fear", "joy", "sadness", "surprise"}
        emotions_4 = {"anger", "joy", "sadness", "fear"} # 排除 disgust 和 surprise
        
        def calculate_weighted_f1(emotions_subset):
            subset_gt_pairs = [p for p in ground_truth_pairs if p["emotion"] in emotions_subset]
            if not subset_gt_pairs:
                return 0.0
            
            from collections import Counter
            gt_counts = Counter(p["emotion"] for p in subset_gt_pairs)
            total_subset = len(subset_gt_pairs)
            
            weighted_f1_sum = 0.0
            for emo in emotions_subset:
                emo_gt = set((_cid(p), p["target_utterance_id"], p["emotion"], p["cause_utterance_id"]) 
                            for p in ground_truth_pairs if p["emotion"] == emo)
                emo_pred = set((_cid(p), p["target_utterance_id"], p["emotion"], p["cause_utterance_id"]) 
                              for p in predicted_pairs if p["emotion"] == emo)
                
                if not emo_gt and not emo_pred:
                    continue
                
                # 计算该类别的 F1
                tp = len(emo_pred & emo_gt)
                p = tp / len(emo_pred) if len(emo_pred) > 0 else 0.0
                r = tp / len(emo_gt) if len(emo_gt) > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                
                # 按真实样本量加权
                weight = gt_counts.get(emo, 0) / total_subset
                weighted_f1_sum += f1 * weight
                
            return weighted_f1_sum

        results["w_avg_6"] = calculate_weighted_f1(emotions_6)
        results["w_avg_4"] = calculate_weighted_f1(emotions_4)

        # 额外：比例匹配 Pair-F1（基于 cause_span 的重叠比例），阈值 0.5
        try:
            prop = self.compute_proportional_pair_f1(predicted_pairs, ground_truth_pairs, threshold=0.5)
            # ensure dict with p,r,f1
            if isinstance(prop, dict):
                results["pair_extraction_proportional_0.5"] = prop
            else:
                results["pair_extraction_proportional_0.5"] = {"p": 0.0, "r": 0.0, "f1": float(prop)}
        except Exception:
            results["pair_extraction_proportional_0.5"] = {"p": 0.0, "r": 0.0, "f1": 0.0}

        return results

    def compute_proportional_pair_f1(self, predicted_pairs: List[Dict], ground_truth_pairs: List[Dict], threshold: float = 0.5) -> float:
        """
        基于 cause_span 重叠比例的柔性匹配：
        - 对于每个预测 (pred)，在相同 target_utterance_id 且 emotion 相同的真实对中，计算 cause_span 的词级重叠比例：|intersection|/|gt_tokens|
        - 若重叠比例 >= threshold，则视为 TP。
        最终计算 P/R/F1。
        """
        import re

        def tokenize(s: str):
            if not s:
                return []
            s = str(s).lower().strip()
            # 移除开头数字前缀和下划线
            s = re.sub(r'^\d+[_\s]*', '', s)
            # 把下划线视为分隔符
            s = s.replace('_', ' ')
            s = re.sub(r"[^\w\s]", " ", s)
            return [t for t in s.split() if t]

        gt_index = {}
        for g in ground_truth_pairs:
            key = (g.get("conversation_id", ""), g.get("target_utterance_id", ""), (g.get("emotion") or "").lower())
            gt_index.setdefault(key, []).append(g)

        tp = 0
        matched_gt = set()
        for p in predicted_pairs:
            key = (p.get("conversation_id", ""), p.get("target_utterance_id", ""), (p.get("emotion") or "").lower())
            if key not in gt_index:
                continue
            p_tokens = set(tokenize(p.get("cause_span", "")))
            for i, g in enumerate(gt_index[key]):
                gt_tokens = tokenize(g.get("cause_span", ""))
                if not gt_tokens:
                    continue
                inter = p_tokens.intersection(set(gt_tokens))
                overlap = len(inter) / max(len(gt_tokens), 1)
                if overlap >= threshold:
                    tp += 1
                    matched_gt.add((key, i))
                    break

        pred_count = len(predicted_pairs)
        gt_count = len(ground_truth_pairs)

        precision = tp / pred_count if pred_count > 0 else 0.0
        recall = tp / gt_count if gt_count > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"p": precision, "r": recall, "f1": f1}

    def format_pairs_for_evaluation(self, scpc_results: List[Dict]) -> List[Dict]:
        """
        将SCPC结果格式化为评估格式
        
        Args:
            scpc_results: SCPC提取结果列表
            
        Returns:
            格式化的配对列表
        """
        formatted = []
        for result in scpc_results:
            formatted.append({
                "emotion": result.get("emotion", ""),
                "cause_utterance_id": result.get("cause_utterance_id", ""),
                "cause_span": result.get("cause_span", ""),
                "target_utterance_id": result.get("target_utterance_id", "")
            })
        return formatted


def test_evaluator():
    """测试评估器"""
    print("=" * 50)
    print("测试评估器")
    print("=" * 50)
    
    evaluator = Evaluator()
    
    # 创建测试数据
    ground_truth = [
        {
            "emotion": "joy",
            "cause_utterance_id": "u3",
            "cause_span": "Did you hear about the promotion?",
            "target_utterance_id": "u5"
        }
    ]
    
    predicted = [
        {
            "emotion": "joy",
            "cause_utterance_id": "u3",
            "cause_span": "Did you hear about the promotion?",
            "target_utterance_id": "u5"
        }
    ]
    
    # 完全匹配的情况
    print("\n1. 完全匹配测试:")
    results = evaluator.evaluate(predicted, ground_truth)
    print(f"   Pair F1: {results['pair_extraction']['f1']:.4f}")
    print(f"   EE F1: {results['emotion_extraction']['f1']:.4f}")
    print(f"   w-avg. 6: {results.get('w_avg_6', 0):.4f}")
    
    # 部分匹配的情况
    print("\n2. 部分匹配测试（情感正确，原因错误）:")
    predicted_partial = [
        {
            "emotion": "joy",
            "cause_utterance_id": "u2",
            "cause_span": "It was okay",
            "target_utterance_id": "u5"
        }
    ]
    results2 = evaluator.evaluate(predicted_partial, ground_truth)
    print(f"   Pair F1: {results2['pair_extraction']['f1']:.4f}")
    print(f"   EE F1: {results2['emotion_extraction']['f1']:.4f}")
    
    print("\n✅ 评估器测试通过!")
    return True


if __name__ == "__main__":
    test_evaluator()

