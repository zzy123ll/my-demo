"""Observability 配置。"""

from dataclasses import dataclass, field


@dataclass
class ObsConfig:
    service_name: str = "rag-cs-api"
    exporter_type: str = "console"  # console | json | jaeger
    json_export_path: str = "logs/traces.jsonl"
    metrics_window_minutes: int = 15
    alert_check_interval_seconds: int = 60
    dashboard_default_minutes: int = 15


def load_obs_config() -> ObsConfig:
    return ObsConfig()
