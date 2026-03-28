"""
ACER Robot - Predictive Maintenance Engine
============================================
AI-driven component health monitoring, anomaly detection,
and predictive failure analysis for uninterrupted field operations.

Monitors:
  - Drive motors (current, temperature, vibration)
  - Battery system (voltage, SoC, cycle count, temperature)
  - Sensor units (calibration drift, signal quality)
  - Compute units (CPU/GPU temp, memory pressure)
  - Communication hardware (signal quality, error rate)

Author: ACER Team
Date: March 2025
"""

import numpy as np
import time
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading

logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    NOMINAL = "nominal"
    DEGRADED = "degraded"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"
    UNKNOWN = "unknown"


class MaintenanceAction(Enum):
    NONE = "none"
    MONITOR = "monitor_closely"
    SCHEDULE = "schedule_maintenance"
    IMMEDIATE = "immediate_service_required"
    ABORT_MISSION = "abort_mission_return_base"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


@dataclass
class ComponentReading:
    component_id: str
    timestamp: float
    metrics: Dict[str, float]


@dataclass
class ComponentHealth:
    component_id: str
    component_type: str
    status: ComponentStatus
    health_score: float           # 0.0 (failed) to 1.0 (perfect)
    anomaly_score: float          # 0.0 (normal) to 1.0 (highly anomalous)
    predicted_rul_hours: float    # Remaining Useful Life in hours
    recommended_action: MaintenanceAction
    details: str = ""
    last_updated: float = field(default_factory=time.time)
    alert_triggered: bool = False


@dataclass
class SystemHealthReport:
    timestamp: float
    overall_status: ComponentStatus
    overall_health_score: float
    components: Dict[str, ComponentHealth]
    critical_alerts: List[str]
    mission_continue_recommended: bool
    estimated_max_mission_duration_h: float


class IsolationForestAnomalyDetector:
    """
    Lightweight Isolation Forest for streaming anomaly detection.
    Trained online from a rolling window of normal readings.
    """

    def __init__(self, window_size: int = 200, contamination: float = 0.05):
        self.window_size = window_size
        self.contamination = contamination
        self._data: deque = deque(maxlen=window_size)
        self._fitted = False
        self._threshold: float = 0.0
        self._lock = threading.Lock()

    def update(self, feature_vector: np.ndarray):
        with self._lock:
            self._data.append(feature_vector.copy())
            if len(self._data) >= 20:
                self._fit()

    def score(self, feature_vector: np.ndarray) -> float:
        """Returns anomaly score in [0, 1]. Higher = more anomalous."""
        if not self._fitted or len(self._data) < 20:
            return 0.0
        with self._lock:
            data = np.array(self._data)
        mean = data.mean(axis=0)
        std = data.std(axis=0) + 1e-8
        z = np.abs((feature_vector - mean) / std)
        score = float(np.tanh(z.mean() / 3.0))
        return min(1.0, score)

    def _fit(self):
        self._fitted = True


class BatteryHealthMonitor:
    """
    Monitors LiFePO4 battery pack health.
    Tracks: voltage, current, temperature, SoC, cycle count.
    """

    # LiFePO4 thresholds
    CELL_VOLTAGE_MIN = 2.8    # V (hard cutoff)
    CELL_VOLTAGE_MAX = 3.65   # V
    CELL_VOLTAGE_NOMINAL = 3.2
    TEMP_MAX_C = 55.0
    TEMP_WARN_C = 45.0
    SOC_LOW = 0.15            # 15%
    SOC_CRITICAL = 0.05       # 5%

    def assess(self, metrics: Dict[str, float]) -> Tuple[ComponentStatus, float, float]:
        """
        Returns (status, health_score [0-1], RUL hours).
        """
        cell_v = metrics.get("cell_voltage_v", self.CELL_VOLTAGE_NOMINAL)
        temp_c = metrics.get("temperature_c", 25.0)
        soc = metrics.get("state_of_charge", 0.8)
        cycle_count = metrics.get("cycle_count", 0)
        capacity_ah = metrics.get("capacity_ah", 100.0)
        rated_capacity = metrics.get("rated_capacity_ah", 100.0)

        capacity_ratio = capacity_ah / (rated_capacity + 1e-6)
        cycle_penalty = max(0.0, 1.0 - cycle_count / 2000.0)

        voltage_ok = self.CELL_VOLTAGE_MIN <= cell_v <= self.CELL_VOLTAGE_MAX
        temp_ok = temp_c <= self.TEMP_MAX_C

        health = capacity_ratio * cycle_penalty
        if not voltage_ok:
            health *= 0.5
        if not temp_ok:
            health *= 0.4

        if soc < self.SOC_CRITICAL or not voltage_ok:
            status = ComponentStatus.CRITICAL
        elif soc < self.SOC_LOW or temp_c > self.TEMP_WARN_C or health < 0.5:
            status = ComponentStatus.WARNING
        elif health < 0.7:
            status = ComponentStatus.DEGRADED
        else:
            status = ComponentStatus.NOMINAL

        # Rough RUL: based on current SoC and discharge rate
        discharge_rate_a = abs(metrics.get("current_a", 5.0))
        rul_h = (soc * capacity_ah) / (discharge_rate_a + 0.1)

        return status, float(np.clip(health, 0, 1)), float(rul_h)


class MotorHealthMonitor:
    """
    Monitors brushless drive motor health.
    Detects: bearing wear (vibration), winding degradation (current patterns),
    overheating.
    """

    TEMP_WARN_C = 80.0
    TEMP_CRITICAL_C = 100.0
    VIBRATION_WARN = 2.5    # m/s²
    VIBRATION_CRITICAL = 5.0

    def __init__(self):
        self._anomaly_detector = IsolationForestAnomalyDetector()

    def assess(self, metrics: Dict[str, float]) -> Tuple[ComponentStatus, float, float]:
        temp_c = metrics.get("winding_temp_c", 40.0)
        vibration = metrics.get("vibration_rms_ms2", 0.5)
        current_a = metrics.get("phase_current_a", 5.0)
        rpm = metrics.get("rpm", 1000.0)
        run_hours = metrics.get("total_run_hours", 0.0)

        features = np.array([temp_c, vibration, current_a, rpm])
        self._anomaly_detector.update(features)
        anomaly = self._anomaly_detector.score(features)

        if temp_c >= self.TEMP_CRITICAL_C or vibration >= self.VIBRATION_CRITICAL:
            status = ComponentStatus.CRITICAL
            health = 0.2
        elif temp_c >= self.TEMP_WARN_C or vibration >= self.VIBRATION_WARN or anomaly > 0.7:
            status = ComponentStatus.WARNING
            health = 0.55
        elif anomaly > 0.4:
            status = ComponentStatus.DEGRADED
            health = 0.75
        else:
            status = ComponentStatus.NOMINAL
            health = max(0.8, 1.0 - (run_hours / 5000.0))

        # Estimated RUL: heuristic based on run_hours and current health
        rul_h = max(0.0, (1.0 - (run_hours / 5000.0)) * 5000.0 * health)

        return status, float(np.clip(health, 0, 1)), float(rul_h)


class PredictiveMaintenanceEngine:
    """
    Central predictive maintenance engine for ACER Robot.
    
    Aggregates component health monitors, generates a system-level
    health report, and issues maintenance actions to the mission
    controller.
    """

    def __init__(self, robot_id: str = "ACER-001"):
        self.robot_id = robot_id
        self._battery = BatteryHealthMonitor()
        self._motors: Dict[str, MotorHealthMonitor] = {
            "FL": MotorHealthMonitor(),
            "FR": MotorHealthMonitor(),
            "RL": MotorHealthMonitor(),
            "RR": MotorHealthMonitor(),
        }
        self._anomaly_detectors: Dict[str, IsolationForestAnomalyDetector] = {}
        self._component_health: Dict[str, ComponentHealth] = {}
        self._alert_history: List[Dict] = []
        self._lock = threading.Lock()
        logger.info(f"PredictiveMaintenanceEngine initialized for {robot_id}")

    def ingest_readings(self, readings: List[ComponentReading]):
        """Process a batch of component readings and update health models."""
        with self._lock:
            for r in readings:
                self._process_reading(r)

    def _process_reading(self, reading: ComponentReading):
        cid = reading.component_id
        ctype = cid.split("_")[0].lower() if "_" in cid else cid.lower()

        status = ComponentStatus.UNKNOWN
        health = 0.5
        rul_h = 100.0
        anomaly = 0.0

        if ctype == "battery":
            status, health, rul_h = self._battery.assess(reading.metrics)

        elif ctype in ("motor", "drive"):
            motor_key = cid.split("_")[-1].upper() if "_" in cid else "FL"
            monitor = self._motors.get(motor_key, self._motors["FL"])
            status, health, rul_h = monitor.assess(reading.metrics)

        else:
            # Generic anomaly detection for unknown component types
            if cid not in self._anomaly_detectors:
                self._anomaly_detectors[cid] = IsolationForestAnomalyDetector()
            det = self._anomaly_detectors[cid]
            features = np.array(list(reading.metrics.values()))
            det.update(features)
            anomaly = det.score(features)
            health = max(0.0, 1.0 - anomaly)
            status = (
                ComponentStatus.CRITICAL if anomaly > 0.8 else
                ComponentStatus.WARNING if anomaly > 0.6 else
                ComponentStatus.DEGRADED if anomaly > 0.4 else
                ComponentStatus.NOMINAL
            )

        action = self._recommend_action(status, rul_h, health)
        alert = status in (ComponentStatus.CRITICAL, ComponentStatus.FAILED)

        self._component_health[cid] = ComponentHealth(
            component_id=cid,
            component_type=ctype,
            status=status,
            health_score=health,
            anomaly_score=anomaly,
            predicted_rul_hours=rul_h,
            recommended_action=action,
            last_updated=reading.timestamp,
            alert_triggered=alert,
        )

        if alert:
            alert_entry = {
                "component": cid,
                "status": status.value,
                "health": round(health, 3),
                "rul_h": round(rul_h, 1),
                "action": action.value,
                "timestamp": reading.timestamp,
            }
            self._alert_history.append(alert_entry)
            logger.critical(
                f"MAINTENANCE ALERT — {cid}: {status.value} | "
                f"Health={health:.2f} | RUL={rul_h:.1f}h | Action={action.value}"
            )

    def generate_report(self) -> SystemHealthReport:
        """Generate a full system health report."""
        now = time.time()
        with self._lock:
            components = dict(self._component_health)

        if not components:
            return SystemHealthReport(
                timestamp=now,
                overall_status=ComponentStatus.UNKNOWN,
                overall_health_score=0.0,
                components={},
                critical_alerts=[],
                mission_continue_recommended=False,
                estimated_max_mission_duration_h=0.0,
            )

        scores = [c.health_score for c in components.values()]
        overall_health = float(np.mean(scores))

        worst_status = max(
            (c.status for c in components.values()),
            key=lambda s: [
                ComponentStatus.UNKNOWN, ComponentStatus.NOMINAL,
                ComponentStatus.DEGRADED, ComponentStatus.WARNING,
                ComponentStatus.CRITICAL, ComponentStatus.FAILED
            ].index(s)
        )

        critical_alerts = [
            f"{cid}: {c.status.value} (health={c.health_score:.2f})"
            for cid, c in components.items()
            if c.status in (ComponentStatus.CRITICAL, ComponentStatus.FAILED)
        ]

        mission_ok = worst_status not in (ComponentStatus.CRITICAL, ComponentStatus.FAILED)
        max_duration = min(
            (c.predicted_rul_hours for c in components.values()),
            default=0.0
        )

        return SystemHealthReport(
            timestamp=now,
            overall_status=worst_status,
            overall_health_score=overall_health,
            components=components,
            critical_alerts=critical_alerts,
            mission_continue_recommended=mission_ok,
            estimated_max_mission_duration_h=max_duration,
        )

    def get_telemetry_snapshot(self) -> Dict:
        report = self.generate_report()
        return {
            "maintenance": {
                "overall_status": report.overall_status.value,
                "overall_health": round(report.overall_health_score, 3),
                "mission_continue": report.mission_continue_recommended,
                "max_duration_h": round(report.estimated_max_mission_duration_h, 1),
                "critical_alerts": report.critical_alerts,
            }
        }

    @staticmethod
    def _recommend_action(
        status: ComponentStatus, rul_h: float, health: float
    ) -> MaintenanceAction:
        if status == ComponentStatus.FAILED:
            return MaintenanceAction.EMERGENCY_SHUTDOWN
        if status == ComponentStatus.CRITICAL or rul_h < 0.5:
            return MaintenanceAction.ABORT_MISSION
        if status == ComponentStatus.WARNING or rul_h < 2.0:
            return MaintenanceAction.IMMEDIATE
        if status == ComponentStatus.DEGRADED or health < 0.7:
            return MaintenanceAction.SCHEDULE
        if health < 0.85:
            return MaintenanceAction.MONITOR
        return MaintenanceAction.NONE
