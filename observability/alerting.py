"""告警规则引擎 — 简化版。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .metrics_store import MetricsStore


@dataclass
class AlertRule:
    name: str
    metric: str          # hallucination_rate | safety_block_rate | avg_latency_ms | error_rate
    condition: str       # "> 0.2" | "< 0.8" | "> 5000"
    duration_seconds: int  # 持续多长时间才触发
    severity: str = "WARNING"  # INFO | WARNING | CRITICAL
    message_template: str = ""

    def evaluate(self, value: float) -> bool:
        op, threshold = self.condition.split(" ", 1)
        threshold = float(threshold)
        if op == ">":
            return value > threshold
        elif op == "<":
            return value < threshold
        elif op == ">=":
            return value >= threshold
        elif op == "<=":
            return value <= threshold
        return False


@dataclass
class Alert:
    rule_name: str
    severity: str
    message: str
    value: float
    threshold: float
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "rule": self.rule_name,
            "level": self.severity,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
        }


class AlertEngine:
    """告警规则引擎。

    用法:
        engine = AlertEngine(metrics_store)
        engine.add_rule(AlertRule("high_hallucination", "hallucination_rate", "> 0.2", 300, "WARNING"))
        alerts = engine.evaluate()
        for a in alerts:
            log_alert(a)
    """

    def __init__(self, metrics: MetricsStore):
        self.metrics = metrics
        self.rules: list[AlertRule] = []
        self._breach_times: dict[str, float] = {}
        self._last_values: dict[str, float] = {}

    def add_rule(self, rule: AlertRule) -> None:
        self.rules.append(rule)

    def evaluate(self) -> list[Alert]:
        """评估所有规则，返回触发的告警列表。"""
        snapshot = self.metrics.snapshot(minutes=15)
        alerts = []

        for rule in self.rules:
            current = self._get_metric_value(snapshot, rule.metric)
            if current is None:
                current = 0.0
            self._last_values[rule.metric] = current

            if rule.evaluate(current):
                if rule.name not in self._breach_times:
                    self._breach_times[rule.name] = time.time()
                else:
                    duration = time.time() - self._breach_times[rule.name]
                    if duration >= rule.duration_seconds:
                        alerts.append(Alert(
                            rule_name=rule.name,
                            severity=rule.severity,
                            message=rule.message_template.format(
                                value=current,
                                threshold=rule.condition.split(" ", 1)[1],
                                duration=int(duration),
                            ) if rule.message_template else
                            f"{rule.metric}={current:.4f} {rule.condition} for {int(duration)}s",
                            value=current,
                            threshold=float(rule.condition.split(" ", 1)[1]),
                            timestamp=time.time(),
                        ))
            else:
                self._breach_times.pop(rule.name, None)

        return alerts

    def _get_metric_value(self, snapshot: dict, metric: str) -> float | None:
        if metric == "hallucination_rate":
            return snapshot.get("hallucination_rate")
        if metric == "safety_block_rate":
            return snapshot.get("safety_block_rate")
        if metric.startswith("avg_latency_"):
            module = metric.replace("avg_latency_", "")
            lat_map = snapshot.get("avg_latency_ms", {})
            return lat_map.get(module)
        if metric == "error_rate":
            return 1.0 - snapshot.get("success_rate", {}).get("retriever", 1.0)
        return None

    def reset(self) -> None:
        self._breach_times.clear()
        self._last_values.clear()
