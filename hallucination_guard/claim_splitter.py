"""答案拆分器：将 LLM 生成的答案分割为原子声明。

支持两种分割引擎:
1. spaCy 中文模型 (zh_core_web_sm) — 优先使用，更精确的分句
2. 正则引擎 — 降级方案，纯 Python 无依赖

使用 factory 方法自动选择可用引擎。
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AtomicClaim:
    """一条原子声明。"""
    text: str
    index: int
    sentence_index: int
    contains_numbers: bool = False
    contains_dates: bool = False


# ---- Regex Engine (fallback) ----

class RegexSentenceSplitter:
    """基于正则的中文分句器。"""

    @staticmethod
    def split(text: str) -> list[str]:
        if not text or not text.strip():
            return []
        parts = re.split(r'(?<=[。！？\n])', text)
        return [p.strip() for p in parts if p.strip()]


class RegexAtomicSplitter:
    """基于正则的原子声明拆分器。

    按连接词（并且、此外、同时）+ 分号拆分，但保护书名号等内容。
    """

    CONJUNCTION_PATTERNS = [
        (r'(?:，\s*)?并且\s*', "并且"),
        (r'(?:，\s*)?此外\s*', "此外"),
        (r'(?:，\s*)?另外\s*', "另外"),
        (r'(?:，\s*)?同时\s*', "同时"),
        (r'(?:，\s*)?以及\s*', "以及"),
        (r'[；;]\s*', "分号"),
    ]

    PRESERVE_PATTERNS = [
        (r'《[^》]+》', lambda m: f"__PROTECTED_{hash(m.group()) % 10000}__"),
        (r'\d+[\.、]\d+', lambda m: f"__PROTECTED_{hash(m.group()) % 10000 + 10000}__"),
    ]

    @classmethod
    def split_atomic(cls, sentence: str) -> list[str]:
        parts = [sentence]
        for pattern, _ in cls.CONJUNCTION_PATTERNS:
            new_parts = []
            for part in parts:
                new_parts.extend(re.split(pattern, part))
            parts = new_parts
        return [p.strip() for p in parts if p.strip()]


# ---- spaCy Engine (preferred) ----

class SpacySentenceSplitter:
    """基于 spaCy 的中文分句器。"""

    _nlp = None

    @classmethod
    def _get_nlp(cls):
        if cls._nlp is None:
            import spacy
            cls._nlp = spacy.blank("zh")
            cls._nlp.add_pipe("sentencizer")
        return cls._nlp

    @classmethod
    def split(cls, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        nlp = cls._get_nlp()
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


class SpacyAtomicSplitter:
    """基于 spaCy 词性标注的原子声明拆分器。

    利用 spaCy 的依存分析识别并列结构:
    - 并列连接词 (cc): 并且、以及、和
    - 并列连词 (conj): 识别并列的名词短语/动词短语
    """

    _nlp = None

    @classmethod
    def _get_nlp(cls):
        if cls._nlp is None:
            import spacy
            try:
                cls._nlp = spacy.load("zh_core_web_sm")
                logger.info("Loaded spaCy zh_core_web_sm model")
            except Exception:
                cls._nlp = spacy.blank("zh")
                cls._nlp.add_pipe("sentencizer")
                logger.warning("zh_core_web_sm not available, using blank model")
        return cls._nlp

    @classmethod
    def split_atomic(cls, sentence: str) -> list[str]:
        nlp = cls._get_nlp()
        doc = nlp(sentence)

        # 收集所有 token 的依存关系，找到并列连词 (cc) 连接的并列结构
        conj_boundaries = set()
        for token in doc:
            if token.dep_ == "cc":
                # 这个 token 是并列连词 (如 "并且")
                head = token.head
                for child in head.children:
                    if child.dep_ == "conj":
                        conj_boundaries.add(child.i)

        if not conj_boundaries:
            # 无并列结构 → 退化为正则拆分
            return RegexAtomicSplitter.split_atomic(sentence)

        # 在并列边界处拆分
        parts = []
        last_idx = 0
        for boundary in sorted(conj_boundaries):
            part = doc[last_idx:boundary].text.strip()
            if part:
                parts.append(part)
            last_idx = boundary

        tail = doc[last_idx:].text.strip()
        if tail:
            parts.append(tail)

        return parts if parts else [sentence.strip()]


# ---- ClaimSplitter (main interface) ----

class ClaimSplitter:
    """答案拆分器主类。

    自动选择最佳可用引擎:
    - 优先使用 spaCy (如果 zh_core_web_sm 可用)
    - 降级到纯正则引擎

    使用方法:
        splitter = ClaimSplitter(use_spacy=True)
        claims = splitter.split("年假有5天。病假需要证明。")
    """

    def __init__(self, use_spacy: bool = True):
        self.use_spacy = use_spacy
        self._spacy_available = None

    def _check_spacy(self) -> bool:
        """检查 spaCy 中文模型是否可用。"""
        if self._spacy_available is not None:
            return self._spacy_available
        try:
            import spacy
            nlp = spacy.load("zh_core_web_sm")
            _ = nlp("测试")
            self._spacy_available = True
            logger.info("spaCy zh_core_web_sm is available")
        except Exception as e:
            logger.warning(f"spaCy not available: {e}, using regex fallback")
            self._spacy_available = False
        return self._spacy_available

    def split(self, answer: str) -> list[AtomicClaim]:
        """拆分为原子声明列表。"""
        if not answer or not answer.strip():
            return []

        # 1. 句子分割
        if self.use_spacy and self._check_spacy():
            sentences = SpacySentenceSplitter.split(answer)
        else:
            sentences = RegexSentenceSplitter.split(answer)

        if not sentences:
            return []

        # 2. 原子声明拆分
        claims = []
        claim_idx = 0
        for sent_idx, sentence in enumerate(sentences):
            if self.use_spacy and self._check_spacy():
                sub_claims = SpacyAtomicSplitter.split_atomic(sentence)
            else:
                sub_claims = RegexAtomicSplitter.split_atomic(sentence)

            for sub in sub_claims:
                if sub.strip():
                    claims.append(AtomicClaim(
                        text=sub.strip(),
                        index=claim_idx,
                        sentence_index=sent_idx,
                        contains_numbers=bool(re.search(r'\d+', sub)),
                        contains_dates=bool(re.search(r'\d{4}[-/年]', sub)),
                    ))
                    claim_idx += 1

        return claims
