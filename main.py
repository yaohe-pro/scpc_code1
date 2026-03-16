import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.append(str(Path(__file__).parent / "src"))

from data_loader import ECFDataLoader
from evaluator import Evaluator
from multimodal_processor import MultimodalProcessor
from scpc import SelfCorrectingPromptChain


def parse_bool_env(var_name: str, default: bool) -> bool:
    raw = os.getenv(var_name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def load_runtime_config(config_path: Path) -> Dict:
    defaults = {
        "model": {
            "name": "gpt-4o",
            "temperature_extraction": 0.0,
            "temperature_critique": 0.7,
        },
        "data": {
            "dataset_path": "data/raw/ECF",
            "video_fps": 1,
            "enable_audio": True,
            "enable_video": True,
        },
        "scpc": {
            "max_iterations": 2,
            "enable_critique": True,
            "enable_refinement": True,
        },
        "runtime": {
            "test_mode": True,
            "force_mock": True,
            "test_max_conversations": 5,
        },
        "evaluation": {
            "output_dir": "results",
        },
    }

    if not config_path.exists():
        print(f"⚠️ 配置文件不存在，使用默认配置: {config_path}")
        return defaults

    try:
        import yaml
    except ImportError:
        print("⚠️ 未安装 pyyaml，使用默认配置")
        return defaults

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"⚠️ 读取配置失败，使用默认配置: {e}")
        return defaults

    if not isinstance(loaded, dict):
        return defaults

    merged = dict(defaults)
    for key in ("model", "data", "scpc", "runtime", "evaluation"):
        section = loaded.get(key, {})
        if isinstance(section, dict):
            merged[key] = {**defaults[key], **section}

    return merged


def main():
    print("=" * 60)
    print("Reflective Extraction: Zero-Shot Multimodal Emotion-Cause Pair Extraction")
    print("=" * 60)

    config = load_runtime_config(Path("config/config.yaml"))
    data_cfg = config["data"]
    model_cfg = config["model"]
    scpc_cfg = config["scpc"]
    runtime_cfg = config["runtime"]
    eval_cfg = config["evaluation"]

    test_mode = parse_bool_env("SCPC_TEST_MODE", bool(runtime_cfg.get("test_mode", True)))
    force_mock = parse_bool_env("SCPC_FORCE_MOCK", bool(runtime_cfg.get("force_mock", True)))

    try:
        test_max_conversations = int(
            os.getenv("SCPC_TEST_MAX_CONVS", str(runtime_cfg.get("test_max_conversations", 5)))
        )
    except ValueError:
        test_max_conversations = int(runtime_cfg.get("test_max_conversations", 5))
    test_max_conversations = max(0, test_max_conversations)

    output_dir = Path(str(eval_cfg.get("output_dir", "results")))
    if test_mode:
        output_dir = output_dir / "test_mode"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_dir / "predictions_checkpoint.jsonl"

    print("\n[初始化] 加载组件...")
    if test_mode:
        print("🧪 运行模式: TEST MODE")
    else:
        print("🚀 运行模式: FULL MODE")

    data_loader = ECFDataLoader(data_cfg.get("dataset_path", "data/raw/ECF"))
    multimodal_processor = MultimodalProcessor(
        video_fps=int(data_cfg.get("video_fps", 1)),
        enable_audio=bool(data_cfg.get("enable_audio", True)),
        enable_video=bool(data_cfg.get("enable_video", True)),
    )

    try:
        from dotenv import load_dotenv

        try:
            load_dotenv(override=True)
        except PermissionError:
            pass
    except ImportError:
        pass

    api_key = os.getenv("OPENAI_API_KEY", "")
    use_mock = api_key == ""
    if test_mode and force_mock:
        use_mock = True

    model_name = os.getenv("MODEL_NAME", model_cfg.get("name", "gpt-4o"))

    if use_mock:
        print("⚠️ 使用模拟模式 (Mock LLM)")
    else:
        print("✅ 使用真实API")
        print(f"   端点: {os.getenv('OPENAI_BASE_URL', '默认')}")
        print(f"   模型: {model_name}")

    print(
        "📌 SCPC参数: "
        f"extract_temp={model_cfg.get('temperature_extraction', 0.0)}, "
        f"critique_temp={model_cfg.get('temperature_critique', 0.7)}, "
        f"max_iterations={scpc_cfg.get('max_iterations', 2)}"
    )
    if test_mode:
        print(
            "📌 TEST参数: "
            f"force_mock={force_mock}, test_max_conversations={test_max_conversations}"
        )

    scpc = SelfCorrectingPromptChain(
        model_name=model_name,
        use_mock=use_mock,
        enable_critique=bool(scpc_cfg.get("enable_critique", True)),
        enable_refinement=bool(scpc_cfg.get("enable_refinement", True)),
        max_iterations=int(scpc_cfg.get("max_iterations", 2)),
        extraction_temperature=float(model_cfg.get("temperature_extraction", 0.0)),
        critique_temperature=float(model_cfg.get("temperature_critique", 0.7)),
    )

    evaluator = Evaluator()
    print("✅ 组件加载完成")

    print("\n[数据准备] 加载测试集...")
    real_conversations = data_loader.load_all_conversations("test")

    # 仅保留可处理对话，避免混入原始聚合文件导致 conversation_id 缺失
    real_conversations = [
        c
        for c in real_conversations
        if isinstance(c, dict)
        and c.get("conversation_id")
        and isinstance(c.get("utterances"), list)
    ]

    if test_mode and test_max_conversations > 0:
        real_conversations = sorted(
            real_conversations, key=lambda c: str(c.get("conversation_id", ""))
        )[:test_max_conversations]
        print(f"🧪 TEST MODE: 仅加载前 {len(real_conversations)} 个对话")

    if not real_conversations:
        print("❌ 未找到测试数据，请检查 data/raw/ECF 路径")
        return {}

    processed_conv_ids = set()
    all_predicted_pairs = []
    all_ground_truth_pairs = []

    if checkpoint_file.exists():
        print(f"检测到断点文件: {checkpoint_file}，正在恢复进度...")
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    cid = record["conversation_id"]
                    if cid in processed_conv_ids:
                        continue
                    processed_conv_ids.add(cid)

                    for p in record.get("predicted_pairs", []):
                        p.setdefault("conversation_id", cid)
                    for g in record.get("ground_truth_pairs", []):
                        g.setdefault("conversation_id", cid)

                    all_predicted_pairs.extend(record.get("predicted_pairs", []))
                    all_ground_truth_pairs.extend(record.get("ground_truth_pairs", []))
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"  ⚠️ 跳过无效断点行: {e}")
                    continue
        print(f"✅ 已加载 {len(processed_conv_ids)} 个对话的缓存结果")

    total_convs = len(real_conversations)
    to_process = [c for c in real_conversations if c.get("conversation_id") not in processed_conv_ids]

    print(f"📊 状态统计: 总数 {total_convs}, 已处理 {len(processed_conv_ids)}, 待处理 {len(to_process)}")
    if not to_process:
        print("💡 所有对话已处理完成。如需重新运行，请删除 checkpoint 文件。")

    print("\n" + "=" * 60)
    print(f"[批量处理] 开始处理剩余 {len(to_process)} 个对话...")
    print("=" * 60)

    start_time = time.time()

    try:
        for conversation in real_conversations:
            if not isinstance(conversation, dict):
                continue

            conv_id = conversation.get("conversation_id")
            if not conv_id or conv_id in processed_conv_ids:
                continue

            utterances = conversation.get("utterances", [])
            print(f"\n[{len(processed_conv_ids) + 1}/{total_convs}] 正在处理: {conv_id} ({len(utterances)} 话语)")

            gt_pairs = data_loader.get_emotion_cause_pairs(conversation)

            try:
                if use_mock and not any(utt.get("audio_features") for utt in utterances):
                    conversation = multimodal_processor.add_mock_multimodal_features(conversation)

                predicted_results = scpc.process_conversation(conversation)

                conv_predicted_pairs = []
                for result in predicted_results:
                    conv_predicted_pairs.append(
                        {
                            "emotion": result.get("emotion", ""),
                            "cause_utterance_id": result.get("cause_utterance_id", ""),
                            "cause_span": result.get("cause_span", ""),
                            "target_utterance_id": result.get("target_utterance_id", ""),
                        }
                    )

                with open(checkpoint_file, "a", encoding="utf-8") as f:
                    checkpoint_record = {
                        "conversation_id": conv_id,
                        "predicted_pairs": conv_predicted_pairs,
                        "ground_truth_pairs": gt_pairs,
                    }
                    f.write(json.dumps(checkpoint_record, ensure_ascii=False) + "\n")

                for p in conv_predicted_pairs:
                    p["conversation_id"] = conv_id
                all_predicted_pairs.extend(conv_predicted_pairs)

                for g in gt_pairs:
                    g["conversation_id"] = conv_id
                all_ground_truth_pairs.extend(gt_pairs)

                processed_conv_ids.add(conv_id)
                print(f"  - 提取完成并保存: 预测 {len(conv_predicted_pairs)} 对, 真实 {len(gt_pairs)} 对")

            except Exception as e:
                print(f"  ❌ 对话 {conv_id} 处理失败: {e}")
                continue

    except KeyboardInterrupt:
        print("\n\n⚠️ 检测到手动中断，正在生成当前进度的评估报告...")
    except Exception as e:
        print(f"\n\n❌ 运行发生致命错误: {e}")
    finally:
        duration = time.time() - start_time

        if not all_ground_truth_pairs:
            print("\n未处理任何有效数据，无法生成评估报告。")
            return {}

        print("\n" + "=" * 60)
        print(f"[评估报告] 已处理对话数: {len(processed_conv_ids)} (耗时: {duration:.2f}s)")
        print("=" * 60)
        if test_mode:
            print("🧪 当前为 TEST MODE 结果，仅用于联调与演示")

        n_pred = len(all_predicted_pairs)
        n_gt = len(all_ground_truth_pairs)
        print(f"\n[诊断] 预测对数: {n_pred}, 真实对数: {n_gt}")

        evaluation_results = evaluator.evaluate(all_predicted_pairs, all_ground_truth_pairs)

        print("\n总体指标 (Global Metrics):")
        for task, metrics in evaluation_results.items():
            if isinstance(metrics, dict):
                task_name = {
                    "emotion_extraction": "1. 情感抽取 (EE)",
                    "cause_extraction": "2. 原因抽取 (CE)",
                    "pair_extraction": "3. 情感-原因对抽取 (ECPE)",
                }.get(task, task)
                print(f"\n  {task_name}:")
                print(f"    Precision: {metrics['p']:.4f}")
                print(f"    Recall:    {metrics['r']:.4f}")
                print(f"    F1-Score:  {metrics['f1']:.4f}")

        print("\n" + "=" * 60)
        print("✅ 结果已汇总!")
        print(f"📁 结果目录: {output_dir}")
        print("=" * 60)

    return evaluation_results


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
