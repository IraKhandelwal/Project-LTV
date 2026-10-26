"""Deterministic, explicitly synthetic readings for the local replay dashboard."""

from datetime import datetime, timezone
from threading import RLock
from uuid import uuid4


SCENARIOS = {
    "steady": {"label": "Steady crowd", "start": (1.1, 0.7), "step": (0.0, 0.0)},
    "rising": {"label": "Gradually rising", "start": (1.4, 0.8), "step": (0.48, 0.28)},
    "surge": {"label": "Surge developing", "start": (2.5, 1.8), "step": (0.62, 0.5)},
    "critical": {"label": "Critical density", "start": (6.2, 4.4), "step": (0.0, 0.0)},
}

_lock = RLock()
_scenario = "steady"
_running = False
_tick = 0
_zones: list[dict] = []
_alerts: list[dict] = []
_last_alert: dict[tuple[str, str], datetime] = {}


def _level(density: float) -> str:
    if density > 6:
        return "red"
    if density >= 4:
        return "amber"
    if density >= 2:
        return "yellow"
    return "green"


def _recommendation(level: str) -> str:
    return {
        "yellow": "Officer review: assess crowd flow and consider approved guidance.",
        "amber": "Officer review: assess crush risk and consider an approved diversion.",
        "red": "Urgent officer review: movement may be severely restricted; consider approved action.",
    }[level]


def _make_zone(zone_id: str, name: str, density: float, index: int) -> dict:
    # The areas below only turn synthetic density into illustrative demo counts.
    area_m2 = (420.0, 560.0)[index]
    capacity_density = (6.5, 5.5)[index]
    return {
        "zone_id": zone_id,
        "name": name,
        "density_persons_m2": round(density, 2),
        "headcount": round(density * area_m2),
        "capacity_percent": round(min(100.0, density / capacity_density * 100), 1),
        "level": _level(density),
        "confidence": 0.0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _refresh_zones() -> None:
    global _zones
    config = SCENARIOS[_scenario]
    densities = [start + _tick * step for start, step in zip(config["start"], config["step"])]
    _zones = [
        _make_zone("north-gate", "North Gate", densities[0], 0),
        _make_zone("main-plaza", "Main Plaza", densities[1], 1),
    ]


def _emit_alerts() -> None:
    now = datetime.now(timezone.utc)
    for zone in _zones:
        level = zone["level"]
        if level == "green":
            continue
        key = (zone["zone_id"], level)
        previous = _last_alert.get(key)
        if previous and (now - previous).total_seconds() < 60:
            continue
        _last_alert[key] = now
        _alerts.insert(0, {
            "id": f"sim-{uuid4()}",
            "zone_id": zone["zone_id"],
            "level": level,
            "reading": f"{zone['density_persons_m2']:.1f} persons/m² · simulated",
            "recommendation": _recommendation(level),
            "acknowledged": False,
            "escalated": False,
            "created_at": now.isoformat(),
        })


def start_scenario(scenario: str) -> dict:
    global _scenario, _running, _tick, _alerts, _last_alert
    if scenario not in SCENARIOS:
        raise ValueError("Unknown demo scenario")
    with _lock:
        _scenario = scenario
        _running = True
        _tick = 0
        _alerts = []
        _last_alert = {}
        _refresh_zones()
        _emit_alerts()
        return get_state()


def stop_scenario() -> dict:
    global _running, _scenario, _tick, _alerts, _last_alert
    with _lock:
        _running = False
        _scenario = "steady"
        _tick = 0
        _alerts = []
        _last_alert = {}
        _refresh_zones()
        return get_state()


def advance() -> dict:
    global _tick
    with _lock:
        if _running:
            _tick += 1
            _refresh_zones()
            _emit_alerts()
            now = datetime.now(timezone.utc)
            for alert in _alerts:
                if (alert["level"] == "red" and not alert["acknowledged"]
                        and not alert["escalated"]
                        and (now - datetime.fromisoformat(alert["created_at"])).total_seconds() >= 30):
                    alert["escalated"] = True
                    alert["recommendation"] += " Escalated for supervisor review after 30 seconds without acknowledgement."
        return get_state()


def get_state() -> dict:
    # RLock allows callers such as start_scenario to take a consistent snapshot.
    with _lock:
        return {
            "scenario": _scenario,
            "scenario_label": SCENARIOS[_scenario]["label"],
            "running": _running,
            "tick": _tick,
            "zones": [dict(zone) for zone in _zones],
            "alerts": [dict(alert) for alert in _alerts],
            "synthetic": True,
        }


def acknowledge(alert_id: str) -> dict | None:
    with _lock:
        for alert in _alerts:
            if alert["id"] == alert_id:
                alert["acknowledged"] = True
                return dict(alert)
    return None


_refresh_zones()
