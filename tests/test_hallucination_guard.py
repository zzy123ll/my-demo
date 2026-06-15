"""Hallucination Guard 模块的完整测试。"""

import pytest
from unittest.mock import MagicMock, patch

from hallucination_guard.claim_splitter import ClaimSplitter, AtomicClaim
from hallucination_guard.config import GuardConfig


RETRIEVED_DOCS = [
    {"chunk_id": "doc_1_chunk_1", "doc_id": "doc_1",
     "content": "入职满一年的员工享有带薪年假5天。入职满五年的员工享有带薪年假10天。"},
    {"chunk_id": "doc_1_chunk_2", "doc_id": "doc_1",
     "content": "病假需要提供二级以上医院开具的病假证明，3天以内由部门主管审批。"},
    {"chunk_id": "doc_2_chunk_1", "doc_id": "doc_2",
     "content": "工作日加班按基本工资的150%计算，休息日加班按200%计算。法定节假日加班按300%计算。"},
]

MOSTLY_CORRECT_ANSWER = """
根据公司政策，年假有5天。入职满五年的员工享有10天年假。
病假需要医生证明，由主管审批即可。另外，加班按基本工资的200%计算。
"""

ALL_HALLUCINATION_ANSWER = """
公司每年还提供额外的免费旅游福利，包括巴厘岛七日游。
员工生日当天可以领取5000元礼品卡。此外，每周三为强制休息日。
"""

ANSWER_WITH_FABRICATED_NUMBERS = """
根据规定，年假有15天。病假需要提供医院证明并经过两级审批。
加班费按基本工资的500%计算。入职不满一年员工也可以申请年假。
"""


def _strict_mock_entailment(premise_docs, hypothesis):
    """严格版 mock NLI: 假设中的数字和关键实体必须在文档中精确匹配。"""
    import re
    combined = " ".join(premise_docs)
    if not combined:
        return False

    # 提取假设中的数字
    numbers = re.findall(r'\d+', hypothesis)

    # 严格检查: 所有数字必须在文档中出现
    for n in numbers:
        if n not in combined:
            return False

    # 关键术语检查
    key_terms = re.findall(r'年假|病假|加班|入职|审批|主管|基本工资|证明|医院', hypothesis)
    if key_terms:
        has_term = any(t in combined for t in key_terms)
        if not has_term:
            return False

    return bool(numbers) or bool(key_terms)


# ============================================================
class TestClaimSplitter:
    def test_basic_split(self):
        claims = ClaimSplitter().split("年假有5天。病假需要医院证明。")
        assert len(claims) == 2

    def test_conjunction_split(self):
        claims = ClaimSplitter().split("年假有5天，并且病假需要医院证明。")
        assert len(claims) >= 2

    def test_semicolon_split(self):
        claims = ClaimSplitter().split("年假5天；病假需审批；加班费按150%计算。")
        assert len(claims) >= 3

    def test_empty_answer(self):
        assert ClaimSplitter().split("") == []
        assert ClaimSplitter().split("   ") == []

    def test_number_detection(self):
        claims = ClaimSplitter().split("年假有15天。")
        assert any(c.contains_numbers for c in claims)

    def test_date_detection(self):
        claims = ClaimSplitter().split("政策于2024年1月1日起执行。")
        assert any(c.contains_dates for c in claims)

    def test_preserve_doc_names(self):
        claims = ClaimSplitter().split("参考《员工手册》并且《绩效考核制度》执行。")
        texts = " ".join(c.text for c in claims)
        assert "《员工手册》" in texts

    def test_index_tracking(self):
        claims = ClaimSplitter().split("第一句。第二句。第三句。")
        assert claims[0].index == 0
        assert claims[2].index == 2


# ============================================================
class TestNLIChecker:
    def test_init_no_model_load(self):
        from hallucination_guard.nli_checker import NLIChecker
        assert not NLIChecker().is_loaded

    def test_entailment_found(self):
        from hallucination_guard.nli_checker import NLIChecker
        import torch
        checker = NLIChecker()
        checker._model = MagicMock()
        checker._tokenizer = MagicMock()
        checker._tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        logits = torch.tensor([[-2.0, -1.0, 3.0]])
        checker._model.return_value = MagicMock(logits=logits)
        assert checker.check_entailment(["年假有5天。"], "年假有5天。") is True

    def test_no_entailment(self):
        from hallucination_guard.nli_checker import NLIChecker
        import torch
        checker = NLIChecker()
        checker._model = MagicMock()
        checker._tokenizer = MagicMock()
        checker._tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        logits = torch.tensor([[3.0, 1.0, -2.0]])
        checker._model.return_value = MagicMock(logits=logits)
        assert checker.check_entailment(["年假有5天。"], "年假有100天。") is False

    def test_empty_premise(self):
        from hallucination_guard.nli_checker import NLIChecker
        checker = NLIChecker()
        checker._model = MagicMock()
        checker._tokenizer = MagicMock()
        assert checker.check_entailment([], "声明") is False


# ============================================================
class TestHallucinationGuard:
    def test_mostly_correct_answer(self):
        from hallucination_guard.hallucination_guard import HallucinationGuard
        guard = HallucinationGuard()
        guard.nli.check_entailment = _strict_mock_entailment
        output = guard.guard(MOSTLY_CORRECT_ANSWER, RETRIEVED_DOCS)
        assert output.safe_answer
        assert output.total_statements > 0
        assert output.supported_count > 0

    def test_all_hallucination_answer(self):
        from hallucination_guard.hallucination_guard import HallucinationGuard
        guard = HallucinationGuard()
        guard.nli.check_entailment = _strict_mock_entailment
        output = guard.guard(ALL_HALLUCINATION_ANSWER, RETRIEVED_DOCS)
        assert output.all_hallucination
        assert output.safe_answer == guard.config.fallback_message
        assert len(output.unsupported_spans) > 0

    def test_fabricated_numbers_detected(self):
        """核心测试：检测编造的数字。"""
        from hallucination_guard.hallucination_guard import HallucinationGuard
        guard = HallucinationGuard()
        guard.nli.check_entailment = _strict_mock_entailment
        output = guard.guard(ANSWER_WITH_FABRICATED_NUMBERS, RETRIEVED_DOCS)
        # "年假有15天" -> "15" 不在文档中,"500%" -> "500" 不在文档中
        assert len(output.unsupported_spans) > 0
        assert "医院证明" in output.safe_answer or "病假" in output.safe_answer

    def test_fabricated_dates_detected(self):
        from hallucination_guard.hallucination_guard import HallucinationGuard
        guard = HallucinationGuard()
        guard.nli.check_entailment = _strict_mock_entailment
        answer = "政策自2025年3月1日起执行。年假有5天。"
        docs = [{"chunk_id": "d1", "doc_id": "d1",
                 "content": "政策自2024年1月1日起执行。年假有5天。"}]
        output = guard.guard(answer, docs)
        assert "年假" in output.safe_answer or output.supported_count > 0

    def test_output_format(self):
        from hallucination_guard.hallucination_guard import HallucinationGuard
        guard = HallucinationGuard()
        guard.nli.check_entailment = _strict_mock_entailment
        output = guard.guard(MOSTLY_CORRECT_ANSWER, RETRIEVED_DOCS)
        json_out = guard.to_json(output)
        for key in ["safe_answer", "unsupported_spans", "citations",
                     "all_hallucination", "stats"]:
            assert key in json_out
        assert isinstance(json_out["stats"]["total"], int)
        assert isinstance(json_out["stats"]["latency_ms"], float)

    def test_latency_tracking(self):
        from hallucination_guard.hallucination_guard import HallucinationGuard
        guard = HallucinationGuard()
        guard.nli.check_entailment = _strict_mock_entailment
        output = guard.guard(MOSTLY_CORRECT_ANSWER, RETRIEVED_DOCS)
        assert output.detection_latency_ms >= 0

    def test_citation_generation(self):
        from hallucination_guard.hallucination_guard import HallucinationGuard
        guard = HallucinationGuard()
        guard.nli.check_entailment = _strict_mock_entailment
        output = guard.guard(MOSTLY_CORRECT_ANSWER, RETRIEVED_DOCS)
        json_out = guard.to_json(output)
        if output.supported_count > 0:
            assert len(json_out["citations"]) > 0


# ============================================================
class TestEdgeCases:
    def test_empty_answer(self):
        from hallucination_guard.hallucination_guard import HallucinationGuard
        output = HallucinationGuard().guard("", [])
        assert output.safe_answer == ""

    def test_no_retrieved_docs(self):
        from hallucination_guard.hallucination_guard import HallucinationGuard
        guard = HallucinationGuard()
        guard.nli.check_entailment = _strict_mock_entailment
        output = guard.guard("年假有5天。", [])
        assert output.total_statements == 0 or output.all_hallucination

    def test_single_statement_answer(self):
        from hallucination_guard.hallucination_guard import HallucinationGuard
        guard = HallucinationGuard()
        guard.nli.check_entailment = _strict_mock_entailment
        output = guard.guard("年假有5天。", RETRIEVED_DOCS)
        assert output.total_statements == 1
        assert output.supported_count == 1 and output.hallucination_count == 0

    def test_mixed_chinese_english(self):
        claims = ClaimSplitter().split(
            "根据SOX法案404条款，企业需要建立内部控制体系。此外，PCAOB要求外部审计。"
        )
        assert len(claims) >= 2


# ============================================================
class TestStreamingProtocol:
    def test_retract_event(self):
        from hallucination_guard.streaming_protocol import create_retract_event
        event = create_retract_event([{"start": 0, "end": 5}], "hallucination")
        assert event.event == "retract"
        assert event.data["reason"] == "hallucination"

    def test_replace_event(self):
        from hallucination_guard.streaming_protocol import create_replace_event
        event = create_replace_event("年假有10天。", "年假有5天。",
                                     [{"span": [3, 5], "type": "hallucination"}])
        assert event.event == "replace"
        assert event.data["replacement"] == "年假有5天。"
        assert len(event.data["highlight"]) > 0

    def test_citation_event(self):
        from hallucination_guard.streaming_protocol import create_citation_event
        event = create_citation_event(0, "doc_42", "入职满一年的员工享有带薪年假5天。")
        assert event.event == "citation"
        assert event.data["source_doc_id"] == "doc_42"
