"""
多模态特征分类器
将 data_loader 预计算的音频/视频统计特征转换为自然语言描述。

阈值根据 ECF test set 2566 条话语的实际百分位分布校准：
  Audio energy  : p10=1.36e8  p25=3.82e8  median=8.36e8  p75=1.69e9  p90=2.87e9
  Audio q75     : p10=1.17     p25=1.40     median=1.62    p75=1.83    p90=2.02
  Audio e_std   : p10=7.0e9   p25=2.0e10   median=4.5e10  p75=9.2e10  p90=1.6e11
  Video max_act : p10=-0.473   p25=-0.464   median=-0.453  p75=-0.442  p90=-0.433
"""
from typing import Dict, List, Optional


class MultimodalClassifier:

    # ---- audio ----

    def describe_audio(self, embedding: List[float] = None,
                       stats: Optional[Dict] = None) -> str:
        if stats:
            return self._describe_audio_from_stats(stats)
        if embedding and len(embedding) >= 10:
            return self._audio_fallback(embedding)
        return "audio unavailable"

    def _describe_audio_from_stats(self, s: Dict) -> str:
        energy = s.get("energy", 0)
        e_std = s.get("energy_std", 0)
        q75 = s.get("q75", 0)
        q25 = s.get("q25", 0)
        nz = s.get("nonzero_ratio", 0)

        parts = []

        # 1) 响度 / 语音能量 (基于 energy 百分位)
        if energy < 1e7:
            parts.append("near silence")
        elif energy < 3.8e8:
            parts.append("soft voice")
        elif energy < 8.4e8:
            parts.append("moderate volume")
        elif energy < 1.7e9:
            parts.append("moderately loud")
        elif energy < 2.9e9:
            parts.append("loud voice")
        else:
            parts.append("very loud voice")

        # 2) 语调变化幅度 (基于 q75 - q25 = IQR)
        iqr = q75 - q25
        if iqr < 1.0:
            parts.append("flat pitch")
        elif iqr < 1.35:
            parts.append("slightly varied pitch")
        elif iqr < 1.6:
            parts.append("moderate pitch variation")
        elif iqr < 1.8:
            parts.append("dynamic pitch")
        else:
            parts.append("highly dynamic pitch")

        # 3) 表现力 (基于 energy_std)
        if e_std < 7e9:
            parts.append("restrained")
        elif e_std < 2e10:
            parts.append("calm tone")
        elif e_std < 4.5e10:
            parts.append("moderate expressiveness")
        elif e_std < 9.2e10:
            parts.append("expressive")
        elif e_std < 1.6e11:
            parts.append("quite expressive")
        else:
            parts.append("very expressive")

        # 4) 稀疏/静音检测
        if nz < 0.5:
            parts.append("mostly silence")
        elif nz < 0.99:
            parts.append("contains pauses")

        return ", ".join(parts)

    @staticmethod
    def _audio_fallback(embedding: List[float]) -> str:
        import numpy as np
        arr = np.array(embedding)
        e = float(np.mean(np.abs(arr)))
        if e > 1e9:
            return "loud, expressive"
        elif e > 1e8:
            return "moderate volume"
        return "quiet"

    # ---- video ----

    def describe_video(self, embedding: List[float] = None,
                       stats: Optional[Dict] = None) -> str:
        if stats:
            return self._describe_video_from_stats(stats)
        if embedding and len(embedding) >= 10:
            return self._video_fallback(embedding)
        return "visual unavailable"

    def _describe_video_from_stats(self, s: Dict) -> str:
        """
        VGG-Face fc6 特征全为负值，范围很窄，分辨力有限。
        用 max_activation 和 mean_activation 的相对位置提供尽可能的区分。
        """
        max_act = s.get("max_activation", -0.45)
        mean_act = s.get("mean_activation", -1.91)
        act_std = s.get("activation_std", 1.09)

        parts = []

        # max_activation: p10=-0.473, p25=-0.464, median=-0.453, p75=-0.442, p90=-0.433
        if max_act > -0.41:
            parts.append("relatively prominent facial features")
        elif max_act > -0.442:
            parts.append("moderately active facial features")
        elif max_act > -0.464:
            parts.append("mild facial features")
        else:
            parts.append("subdued facial features")

        # mean_activation: p25=-1.908, median=-1.906, p75=-1.904
        if mean_act > -1.904:
            parts.append("slightly elevated expression level")
        elif mean_act < -1.910:
            parts.append("low expression level")
        else:
            parts.append("baseline expression level")

        return ", ".join(parts)

    @staticmethod
    def _video_fallback(embedding: List[float]) -> str:
        import numpy as np
        m = float(np.mean(np.array(embedding)))
        if m > -1.90:
            return "slightly expressive face"
        return "neutral expression"

    # ---- combined ----

    def get_multimodal_summary(self, audio_emb=None, video_emb=None,
                               audio_stats=None, video_stats=None) -> str:
        a = self.describe_audio(audio_emb, audio_stats)
        v = self.describe_video(video_emb, video_stats)
        return f"Audio: [{a}]; Visual: [{v}]"
