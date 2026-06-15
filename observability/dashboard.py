"""健康检查看板 — 文本报表格式。"""

from __future__ import annotations

from dataclasses import dataclass

from .metrics_store import MetricsStore


@dataclass
class DashboardReport:
    title: str
    period_minutes: int
    snapshot: dict
    alert_count: int = 0
    alerts: list[dict] = None

    def to_text(self) -> str:
        s = self.snapshot
        lines = [
            f"========== {self.title} ==========",
            f"Period: last {self.period_minutes} min",
            f"Requests: {s.get('total_requests', 0)}",
            "",
            "--- Latency (avg ms) ---",
        ]
        for mod, lat in s.get("avg_latency_ms", {}).items():
            bar = self._bar(lat, 1000)
            lines.append(f"  {mod:>12}: {lat:>7.1f} ms {bar}")

        lines.extend([
            "",
            "--- Quality ---",
            f"  Hallucination Rate: {s.get('hallucination_rate', 0):.1%}",
            f"  Safety Block Rate:  {s.get('safety_block_rate', 0):.1%}",
            "",
            f"Alerts: {self.alert_count}",
        ])
        if self.alerts:
            lines.append("")
            for a in (self.alerts or [])[-3:]:
                lines.append(f"  [{a.get('level','INFO')}] {a.get('message','')}")

        lines.append("=" * 40)
        return "\n".join(lines)

    def _bar(self, value: float, max_val: float) -> str:
        ratio = min(value / max_val, 1.0)
        filled = int(ratio * 10)
        return "[" + "#" * filled + "." * (10 - filled) + "]"


class Dashboard:
    """健康检查看板。

    用法:
        dash = Dashboard(metrics_store)
        # 在定时任务中:
        report = dash.generate(minutes=15)
        print(report.to_text())
    """

    def __init__(self, metrics: MetricsStore,
                 default_minutes: int = 15):
        self.metrics = metrics
        self.default_minutes = default_minutes
        self._last_alerts: list[dict] = []

    def generate(self, minutes: int = None,
                 alert_results: list[dict] = None
                 ) -> DashboardReport:
        minutes = minutes or self.default_minutes
        snapshot = self.metrics.snapshot(minutes)
        alerts = alert_results or self._last_alerts

        return DashboardReport(
            title="RAG CS Health Dashboard",
            period_minutes=minutes,
            snapshot=snapshot,
            alert_count=len(alerts),
            alerts=alerts,
        )

    def update_alerts(self, alerts: list[dict]) -> None:
        self._last_alerts = alerts
