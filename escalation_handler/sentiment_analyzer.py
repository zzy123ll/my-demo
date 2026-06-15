"""中文情感分析器 — 基于规则的轻量实现。"""

from __future__ import annotations


class SentimentAnalyzer:
    """基于规则的中文情感分析器。"""

    POSITIVE_WORDS = {
        "谢谢": 0.8, "感谢": 0.9, "很好": 0.7, "不错": 0.5,
        "满意": 0.8, "喜欢": 0.7, "棒": 0.6, "优秀": 0.8,
        "有用": 0.6, "解决了": 0.8, "明白了": 0.5, "清楚了": 0.5,
        "好的": 0.4, "可以": 0.2, "没问题": 0.5, "完美": 0.9, "赞": 0.6,
    }

    NEGATIVE_WORDS = {
        "不满": -0.7, "失望": -0.8, "差": -0.6, "垃圾": -0.9,
        "没用": -0.7, "废物": -0.9, "滚": -0.8, "烦": -0.5,
        "生气": -0.7, "无语": -0.6, "坑": -0.5, "骗": -0.7,
        "错了": -0.5, "不对": -0.5, "没解决": -0.7, "还是不行": -0.6,
        "没用处": -0.7, "什么玩意": -0.8, "离谱": -0.6,
        "投诉": -0.8, "太差": -0.8, "极差": -0.9, "糟糕": -0.7,
        "很差": -0.7, "极差": -0.9, "不行": -0.5,
    }

    NEGATION_WORDS = {"不", "没", "没有", "别", "不要", "不是", "不会"}

    INTENSIFIERS = {"很": 1.5, "非常": 1.8, "特别": 1.5,
                    "太": 1.6, "极其": 2.0, "十分": 1.5,
                    "超级": 1.5, "相当": 1.3}

    def analyze(self, text: str) -> float:
        if not text or not text.strip():
            return 0.0

        score = 0.0
        word_count = 0

        for word, val in self.POSITIVE_WORDS.items():
            if word in text:
                intensity = self._get_intensity(text, word)
                if self._is_negated(text, word):
                    score -= abs(val) * intensity
                else:
                    score += val * intensity
                word_count += 1

        for word, val in self.NEGATIVE_WORDS.items():
            if word in text:
                intensity = self._get_intensity(text, word)
                if self._is_negated(text, word):
                    score += abs(val) * intensity
                else:
                    score += val * intensity
                word_count += 1

        if word_count == 0:
            return 0.0
        return max(-1.0, min(1.0, score / max(word_count, 1)))

    def _is_negated(self, text: str, word: str) -> bool:
        word_pos = text.find(word)
        if word_pos < 0:
            return False
        for neg in self.NEGATION_WORDS:
            neg_pos = text.find(neg)
            if neg_pos >= 0 and neg_pos < word_pos and (word_pos - neg_pos) < 5:
                return True
        return False

    def _get_intensity(self, text: str, word: str) -> float:
        word_pos = text.find(word)
        if word_pos <= 0:
            return 1.0
        pre = text[max(0, word_pos - 2):word_pos]
        return self.INTENSIFIERS.get(pre.strip(), 1.0)
