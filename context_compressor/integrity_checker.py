"""完整性检查：验证压缩后文本是否保留了原文档的关键实体。"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class IntegrityResult:
    """完整性检查结果。"""
    passed: bool
    entity_preservation_rate: float
    missing_entities: list[str]
    original_entity_count: int
    compressed_entity_count: int
    detail: str = ""


class IntegrityChecker:
    """实体保留完整性检查器。

    提取原文档中的关键实体（数字、日期、专有名词、政策编号等），
    验证压缩后文本中是否保留了这些实体。
    不依赖 NLI 模型，用正则 + 规则即可实现高精度。
    """

    # 关键实体模式
    ENTITY_PATTERNS: list[tuple[str, str]] = [
        # (模式, 实体类型名)
        (r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日号]?', "日期"),
        (r'\d+\.?\d*\s*%', "百分比"),
        (r'[¥￥]\s*\d+[万亿]?\d*(?:\.\d+)?', "金额"),
        (r'[A-Z]{2,6}[-_]\d{3,8}', "编号"),
        (r'《[^》]{2,30}》', "文档名"),
        (r'\d{11,19}', "长数字"),
        (r'\d+\s*(?:天|年|月|周|小时|分钟|元|人|次|个|条|项)', "数量单位"),
        (r'(?:第\s*[一二三四五六七八九十\d]+\s*(?:条|章|节|款|项))', "条款"),
    ]

    # 专有名词检测（连续大写/中文专名）
    PROPER_NOUN_PATTERNS = [
        (r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5}', "英文专名"),
        (r'[\u4e00-\u9fff]{2,4}(?:公司|部门|系统|平台|中心|委员会|管理局)', "机构名"),
    ]

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold

    def check(self, original_text: str,
              compressed_text: str) -> IntegrityResult:
        """检查压缩后文本的实体保留率。

        Args:
            original_text: 原始文档
            compressed_text: 压缩后文本

        Returns:
            IntegrityResult: 检查结果
        """
        # 提取原文档中的所有关键实体
        original_entities = self._extract_entities(original_text)

        if not original_entities:
            return IntegrityResult(
                passed=True,
                entity_preservation_rate=1.0,
                missing_entities=[],
                original_entity_count=0,
                compressed_entity_count=0,
                detail="原始文档未检测到关键实体",
            )

        # 检查哪些实体在压缩文本中出现
        missing = []
        preserved = 0
        for entity in original_entities:
            if entity in compressed_text:
                preserved += 1
            else:
                missing.append(entity)

        total = len(original_entities)
        rate = preserved / total if total > 0 else 0.0
        passed = rate >= self.threshold

        return IntegrityResult(
            passed=passed,
            entity_preservation_rate=round(rate, 4),
            missing_entities=missing,
            original_entity_count=total,
            compressed_entity_count=preserved,
            detail=(
                f"保留 {preserved}/{total} 个关键实体 "
                f"({rate:.1%}), 阈值 {self.threshold:.0%}"
            ),
        )

    def _extract_entities(self, text: str) -> list[str]:
        """从文本中提取所有关键实体（去重）。"""
        entities = set()

        # 通用实体模式
        for pattern, _ in self.ENTITY_PATTERNS:
            for match in re.finditer(pattern, text):
                entities.add(match.group())

        # 专有名词
        for pattern, _ in self.PROPER_NOUN_PATTERNS:
            for match in re.finditer(pattern, text):
                entities.add(match.group())

        return list(entities)
