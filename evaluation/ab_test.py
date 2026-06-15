"""A/B 测试比较脚本。"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ABTestResult:
    experiment_name: str
    variant_a: str   # "control"
    variant_b: str   # "treatment"
    metrics: dict[str, tuple[float, float]]  # metric -> (a_mean, b_mean)
    p_values: dict[str, float]              # metric -> simulated p-value
    significant: dict[str, bool]
    winner: str = ""
    recommendation: str = ""


class ABTestRunner:
    """A/B 测试比较器。

    输入两组评估报告，对每组指标做 t 检验近似，输出差异显著性判断。

    流量分配: user_id % 2 == 0 → variant_a, else → variant_b
    比较指标: hit_rate@5, faithfulness, relevance, user_satisfaction
    """

    def __init__(self, experiment_name: str = "exp-001"):
        self.name = experiment_name

    def compare(self, report_a: dict, report_b: dict,
                metrics_to_compare: list[str] = None) -> ABTestResult:
        metrics_to_compare = metrics_to_compare or [
            "avg_hit_at_5", "avg_faithfulness", "avg_relevance",
            "avg_rouge_l", "user_satisfaction",
        ]

        result = ABTestResult(
            experiment_name=self.name,
            variant_a="control",
            variant_b="treatment",
            metrics={},
            p_values={},
            significant={},
        )

        for metric in metrics_to_compare:
            a_val = self._get_metric(report_a, metric)
            b_val = self._get_metric(report_b, metric)

            if a_val is None or b_val is None:
                continue

            result.metrics[metric] = (round(a_val, 4), round(b_val, 4))

            # 模拟 t 检验（简化版：基于样本量和方差做近似）
            n = report_a.get("total_cases", 30)
            p_val, is_sig = self._simulated_ttest(
                a_val, b_val, n,
                report_a.get("std_" + metric, 0.1),
                report_b.get("std_" + metric, 0.1),
            )
            result.p_values[metric] = round(p_val, 4)
            result.significant[metric] = is_sig

        # 判断赢家
        sig_count = sum(result.significant.values())
        if sig_count >= len(result.metrics) // 2 + 1:
            result.winner = "B" if self._count_better(result) > 0 else "A"
            result.recommendation = (
                f"建议全量部署 variant_{result.winner.lower()}"
            )
        else:
            result.winner = "inconclusive"
            result.recommendation = "差异不显著，建议继续收集数据或扩大样本"

        return result

    def _get_metric(self, report: dict, key: str) -> float | None:
        # 支持嵌套 key
        if key in report:
            return report[key]
        # avg_hit_at_5 → avg_hit_at_k["5"] 或 hit_at_k[5]
        if key.startswith("avg_hit_at_"):
            k = int(key.split("_")[-1])
            hit_map = report.get("avg_hit_at_k", {})
            return hit_map.get(k, None)
        return None

    def _simulated_ttest(self, mean_a: float, mean_b: float, n: int,
                         std_a: float, std_b: float) -> tuple[float, bool]:
        """模拟两样本 t 检验。

        使用 Welch's t-test 近似。
        """
        import math

        diff = abs(mean_a - mean_b)
        if diff == 0:
            return 1.0, False

        # Pooled standard error
        se = math.sqrt(std_a**2 / n + std_b**2 / n)
        if se == 0:
            return 1.0, False

        t_stat = diff / se

        # 使用正态近似计算 p-value (Welch-Satterthwaite df ~ n)
        # 简化: 用标准正态的尾部概率
        import scipy.stats as stats

        # 双尾检验
        p_val = 2 * stats.t.sf(t_stat, df=2 * n - 2)

        return float(p_val), p_val < 0.05

    def _count_better(self, result: ABTestResult) -> int:
        better = 0
        for metric, (a_val, b_val) in result.metrics.items():
            if result.significant.get(metric, False):
                # "higher is better" for most metrics
                if b_val > a_val:
                    better += 1
                else:
                    better -= 1
        return better

    def allocate_traffic(self, user_id: str, experiment: str = "exp-001",
                         traffic_split: float = 0.5) -> str:
        """确定性流量分配: user_id hash → variant。"""
        h = hash(user_id + experiment) % 100
        if h < traffic_split * 100:
            return "A"
        return "B"

    def generate_report(self, result: ABTestResult) -> str:
        lines = [
            f"=== A/B Test Report: {self.name} ===",
            f"Control (A): {result.variant_a}",
            f"Treatment (B): {result.variant_b}",
            "",
            "Metrics:",
        ]
        for metric, (a, b) in result.metrics.items():
            sig = "**" if result.significant.get(metric) else ""
            p = result.p_values.get(metric, 0)
            lines.append(
                f"  {metric}: A={a:.4f} B={b:.4f} "
                f"diff={b-a:+.4f} p={p:.4f} {sig}"
            )

        lines.extend(["", f"Winner: {result.winner}",
                       f"Recommendation: {result.recommendation}"])
        return "\n".join(lines)
