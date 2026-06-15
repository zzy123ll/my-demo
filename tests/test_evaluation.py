"""Evaluation 模块的单元测试。"""

import json
import tempfile
import os
import pytest
from unittest.mock import MagicMock, patch

from evaluation.golden_dataset import GoldenDataset, TestCase
from evaluation.config import EvalConfig


# ============================================================
class TestGoldenDataset:
    def test_load_from_jsonl(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                         delete=False, encoding="utf-8") as f:
            f.write('{"question": "年假几天？", "ground_truth_answer": "5天", "relevant_doc_ids": ["d1"]}\n')
            f.write('{"question": "病假怎么办？", "ground_truth_answer": "医院证明", "relevant_doc_ids": ["d2"]}\n')
            path = f.name

        ds = GoldenDataset(path)
        assert len(ds) == 2
        assert ds[0].question == "年假几天？"
        assert ds[0].relevant_doc_ids == ["d1"]

        os.unlink(path)

    def test_save_and_reload(self):
        ds = GoldenDataset()
        ds.add(TestCase("Q1", "A1", ["d1", "d2"]))
        ds.add(TestCase("Q2", "A2", ["d3"]))

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        ds.save(path)

        ds2 = GoldenDataset(path)
        assert len(ds2) == 2
        assert ds2[0].ground_truth_answer == "A1"

        os.unlink(path)

    def test_filter(self):
        ds = GoldenDataset()
        ds.add(TestCase("Q1", "A1", ["d1"]))
        ds.add(TestCase("Q2", "A2", ["d2", "d3"]))
        filtered = ds.filter(lambda c: len(c.relevant_doc_ids) > 1)
        assert len(filtered) == 1

    def test_empty_dataset(self):
        ds = GoldenDataset()
        assert len(ds) == 0
        assert list(ds) == []


# ============================================================
class TestMetrics:
    def test_hit_at_k(self):
        from evaluation.metrics import compute_hit_at_k

        hit = compute_hit_at_k(["d1", "d2", "d3", "d4"], ["d2", "d5"],
                               [1, 3, 5])
        assert hit[1] == 0.0  # d2 不在 top-1
        assert hit[3] == 1.0  # d2 在 top-3
        assert hit[5] == 1.0

    def test_hit_at_k_miss(self):
        from evaluation.metrics import compute_hit_at_k
        hit = compute_hit_at_k(["d1", "d2"], ["d5", "d6"], [5])
        assert hit[5] == 0.0

    def test_mrr(self):
        from evaluation.metrics import compute_mrr

        mrr = compute_mrr(["d3", "d1", "d2"], ["d1", "d5"], k=10)
        assert mrr == 0.5  # d1 at position 2 → 1/2

    def test_mrr_miss(self):
        from evaluation.metrics import compute_mrr
        mrr = compute_mrr(["d1", "d2"], ["d5"], k=10)
        assert mrr == 0.0

    def test_rouge_l_identical(self):
        from evaluation.metrics import compute_rouge_l

        score = compute_rouge_l("年假有5天", "年假有5天")
        assert score == 1.0

    def test_rouge_l_partial(self):
        from evaluation.metrics import compute_rouge_l

        score = compute_rouge_l("年假有5天病假需要证明",
                                 "年假有5天")
        assert 0 < score < 1.0

    def test_faithfulness_mock(self):
        """核心测试: 用 mock 数据验证 Faithfulness 计算逻辑。"""
        from evaluation.metrics import compute_faithfulness

        # 完全有支撑的回答
        answer = "年假有5天。病假需要医院证明。"
        docs = ["年假有5天。", "病假需要提供二级以上医院证明。"]

        score = compute_faithfulness(answer, docs)
        assert score >= 0.5, f"Expected high faithfulness, got {score}"

    def test_faithfulness_all_hallucination(self):
        from evaluation.metrics import compute_faithfulness

        # 全部幻觉
        answer = "年假有100天。病假不需要证明。"
        docs = ["年假有5天。", "病假需要提供医院证明。"]

        score = compute_faithfulness(answer, docs)
        # 数字 100 不在文档中, "不需要证明" 也不在
        assert score < 0.5, f"Expected low faithfulness, got {score}"

    def test_faithfulness_fabricated_numbers(self):
        from evaluation.metrics import compute_faithfulness

        # 数字不匹配 = 幻觉
        answer = "加班费按基本工资的500%计算。"
        docs = ["加班费按基本工资的150%计算。"]

        score = compute_faithfulness(answer, docs)
        assert score == 0.0, f"Expected 0 faithfulness, got {score}"

    def test_relevance(self):
        from evaluation.metrics import compute_relevance

        rel = compute_relevance("年假有几天", "年假是5天")
        assert rel > 0

    def test_empty_answer(self):
        from evaluation.metrics import compute_faithfulness
        assert compute_faithfulness("", ["doc"]) == 1.0


# ============================================================
class TestOfflineEvaluator:
    def test_evaluate_with_mock(self):
        from evaluation.evaluator import OfflineEvaluator
        from evaluation.golden_dataset import GoldenDataset, TestCase

        ds = GoldenDataset()
        ds.add(TestCase("年假几天？", "5天", ["d1"]))
        ds.add(TestCase("病假怎么办？", "医院证明", ["d2"]))

        # Mock retriever
        mock_retriever = MagicMock()
        mock_retriever.search = MagicMock(return_value=[
            {"chunk_id": "d1", "content": "年假有5天。"},
        ])

        evaluator = OfflineEvaluator(retriever=mock_retriever)
        report = evaluator.evaluate(ds)

        assert report.total_cases == 2
        assert len(report.per_case) == 2
        assert "avg_hit_at_k" in report.to_dict()
        assert report.elapsed_seconds >= 0

    def test_empty_dataset(self):
        from evaluation.evaluator import OfflineEvaluator
        from evaluation.golden_dataset import GoldenDataset

        evaluator = OfflineEvaluator()
        report = evaluator.evaluate(GoldenDataset())
        assert report.total_cases == 0

    def test_report_save(self):
        from evaluation.evaluator import EvaluationReport, EvalMetrics
        import tempfile

        report = EvaluationReport(
            dataset_name="test", total_cases=1,
            avg_hit_at_k={5: 1.0}, avg_mrr=1.0,
            avg_faithfulness=1.0, avg_relevance=0.9, avg_rouge_l=0.8,
            per_case=[], elapsed_seconds=0.5,
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        report.save(path)
        with open(path) as f:
            data = json.load(f)
        assert data["total_cases"] == 1
        assert data["avg_faithfulness"] == 1.0
        os.unlink(path)


# ============================================================
class TestFeedback:
    def test_submit_and_stats(self):
        from evaluation.feedback import FeedbackCollector
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        fc = FeedbackCollector(db_path=path)
        fc.submit("msg_1", rating=5, session_id="s1", user_id="u1")
        fc.submit("msg_2", rating=2, session_id="s2", user_id="u2",
                  comment="不准确")
        fc.submit("msg_3", rating=4, session_id="s3", user_id="u3")

        stats = fc.get_stats()
        assert stats["total"] == 3
        assert stats["avg_rating"] == pytest.approx(11 / 3, abs=0.1)
        assert stats["positive_rate"] == pytest.approx(2 / 3, abs=0.1)
        assert stats["negative_rate"] == pytest.approx(1 / 3, abs=0.1)

        os.unlink(path)

    def test_invalid_rating(self):
        from evaluation.feedback import FeedbackCollector
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        fc = FeedbackCollector(db_path=path)

        with pytest.raises(ValueError):
            fc.submit("msg_1", rating=6)
        with pytest.raises(ValueError):
            fc.submit("msg_1", rating=0)

        os.unlink(path)

    def test_empty_stats(self):
        from evaluation.feedback import FeedbackCollector
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        fc = FeedbackCollector(db_path=path)
        stats = fc.get_stats()
        assert stats["total"] == 0
        os.unlink(path)


# ============================================================
class TestABTest:
    def test_traffic_allocation(self):
        from evaluation.ab_test import ABTestRunner

        runner = ABTestRunner("exp-001")
        # 确定性分配
        a1 = runner.allocate_traffic("user_123", "exp-001", 0.5)
        a2 = runner.allocate_traffic("user_123", "exp-001", 0.5)
        assert a1 == a2  # 同一用户应一致

    def test_compare_reports(self):
        from evaluation.ab_test import ABTestRunner

        report_a = {
            "total_cases": 100,
            "avg_faithfulness": 0.85,
            "avg_relevance": 0.72,
            "avg_hit_at_5": 0.88,
            "avg_rouge_l": 0.65,
            "std_avg_faithfulness": 0.1,
            "std_avg_relevance": 0.12,
            "std_avg_hit_at_5": 0.08,
            "std_avg_rouge_l": 0.15,
        }
        report_b = {
            "total_cases": 100,
            "avg_faithfulness": 0.91,
            "avg_relevance": 0.75,
            "avg_hit_at_5": 0.92,
            "avg_rouge_l": 0.68,
            "std_avg_faithfulness": 0.1,
            "std_avg_relevance": 0.12,
            "std_avg_hit_at_5": 0.08,
            "std_avg_rouge_l": 0.15,
        }

        runner = ABTestRunner("exp-test")
        result = runner.compare(report_a, report_b)

        assert len(result.metrics) > 0
        assert result.winner in ("A", "B", "inconclusive")
        assert len(result.recommendation) > 0

    def test_report_generation(self):
        from evaluation.ab_test import ABTestRunner

        runner = ABTestRunner("exp-1")
        result = runner.compare(
            {"total_cases": 50, "avg_faithfulness": 0.8,
             "std_avg_faithfulness": 0.1},
            {"total_cases": 50, "avg_faithfulness": 0.9,
             "std_avg_faithfulness": 0.1},
        )
        report_text = runner.generate_report(result)
        assert "exp-1" in report_text
        assert "A=" in report_text
        assert "Recommendation" in report_text
