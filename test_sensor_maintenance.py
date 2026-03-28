"""
Tests for ACER Robot Sensor Fusion & Predictive Maintenance
"""

import pytest
import numpy as np
import time

from src.sensors.sensor_fusion import (
    SensorFusionEngine, SensorReading, SensorType,
    ExtendedKalmanFilter, ThermalAnalyzer, AcousticLocalizer
)
from src.maintenance.predictive_maintenance import (
    PredictiveMaintenanceEngine, ComponentReading,
    BatteryHealthMonitor, MotorHealthMonitor,
    ComponentStatus, MaintenanceAction
)


# ── Sensor Fusion Tests ────────────────────────────────────────

class TestExtendedKalmanFilter:

    def test_initial_state_is_zero(self):
        ekf = ExtendedKalmanFilter()
        assert np.allclose(ekf.position, np.zeros(3))

    def test_predict_moves_position_by_velocity(self):
        ekf = ExtendedKalmanFilter()
        ekf.x[3:6] = [1.0, 0.0, 0.0]  # Set velocity [1, 0, 0]
        ekf.predict(dt=1.0)
        assert ekf.position[0] == pytest.approx(1.0, abs=1e-6)

    def test_update_reduces_uncertainty(self):
        ekf = ExtendedKalmanFilter()
        initial_trace = np.trace(ekf.P)
        obs = np.zeros(6)
        ekf.update(obs)
        updated_trace = np.trace(ekf.P)
        assert updated_trace <= initial_trace

    def test_covariance_positive_definite(self):
        ekf = ExtendedKalmanFilter()
        ekf.predict(dt=0.1)
        eigenvalues = np.linalg.eigvals(ekf.P)
        assert np.all(eigenvalues > 0)


class TestThermalAnalyzer:

    def setup_method(self):
        self.analyzer = ThermalAnalyzer()

    def test_detects_human_temperature(self):
        frame = np.zeros((120, 160))
        frame[50:60, 70:80] = 36.8  # Human body temp

        hotspots = self.analyzer.detect_hotspots(frame)
        human = [h for h in hotspots if h["type"] == "human"]
        assert len(human) > 0

    def test_detects_vehicle_heat(self):
        frame = np.zeros((120, 160))
        frame[30:50, 50:90] = 250.0  # Vehicle engine temp

        hotspots = self.analyzer.detect_hotspots(frame)
        vehicles = [h for h in hotspots if h["type"] == "vehicle"]
        assert len(vehicles) > 0

    def test_no_hotspots_in_cold_frame(self):
        frame = np.ones((120, 160)) * 20.0  # Ambient temp
        hotspots = self.analyzer.detect_hotspots(frame)
        assert len(hotspots) == 0

    def test_hotspot_has_required_fields(self):
        frame = np.zeros((120, 160))
        frame[40:50, 60:70] = 37.0
        hotspots = self.analyzer.detect_hotspots(frame)
        if hotspots:
            h = hotspots[0]
            for key in ("type", "centroid", "area", "max_temperature", "confidence"):
                assert key in h


class TestSensorFusionEngine:

    def test_fuse_returns_state(self):
        engine = SensorFusionEngine()
        state = engine.fuse()
        assert state is not None
        assert state.timestamp > 0
        assert len(state.robot_position) == 3

    def test_ingest_and_fuse_imu(self):
        engine = SensorFusionEngine()
        reading = SensorReading(
            sensor_type=SensorType.IMU,
            timestamp=time.time(),
            data={"ax": 0.1, "ay": 0.0, "az": 9.8, "gx": 0, "gy": 0, "gz": 0},
            confidence=1.0,
        )
        engine.ingest(reading)
        state = engine.fuse()
        assert state.confidence_score > 0

    def test_confidence_increases_with_more_sensors(self):
        engine = SensorFusionEngine()

        state_no_data = engine.fuse()

        for sensor_type in [SensorType.IMU, SensorType.LIDAR, SensorType.THERMAL]:
            engine.ingest(SensorReading(
                sensor_type=sensor_type,
                timestamp=time.time(),
                data={"value": 1.0},
            ))
        state_with_data = engine.fuse()

        assert state_with_data.confidence_score >= state_no_data.confidence_score


# ── Predictive Maintenance Tests ───────────────────────────────

class TestBatteryHealthMonitor:

    def setup_method(self):
        self.monitor = BatteryHealthMonitor()

    def test_nominal_battery(self):
        metrics = {
            "cell_voltage_v": 3.2,
            "temperature_c": 25.0,
            "state_of_charge": 0.85,
            "cycle_count": 50,
            "capacity_ah": 100.0,
            "rated_capacity_ah": 100.0,
            "current_a": 5.0,
        }
        status, health, rul = self.monitor.assess(metrics)
        assert status == ComponentStatus.NOMINAL
        assert health > 0.8
        assert rul > 5.0

    def test_critical_low_soc(self):
        metrics = {
            "cell_voltage_v": 3.1,
            "temperature_c": 25.0,
            "state_of_charge": 0.03,
            "cycle_count": 100,
            "capacity_ah": 100.0,
            "rated_capacity_ah": 100.0,
            "current_a": 5.0,
        }
        status, health, rul = self.monitor.assess(metrics)
        assert status == ComponentStatus.CRITICAL

    def test_high_temp_warning(self):
        metrics = {
            "cell_voltage_v": 3.2,
            "temperature_c": 50.0,
            "state_of_charge": 0.7,
            "cycle_count": 100,
            "capacity_ah": 100.0,
            "rated_capacity_ah": 100.0,
            "current_a": 5.0,
        }
        status, _, _ = self.monitor.assess(metrics)
        assert status in (ComponentStatus.WARNING, ComponentStatus.CRITICAL)


class TestMotorHealthMonitor:

    def test_nominal_motor(self):
        monitor = MotorHealthMonitor()
        metrics = {
            "winding_temp_c": 45.0,
            "vibration_rms_ms2": 0.3,
            "phase_current_a": 8.0,
            "rpm": 2000.0,
            "total_run_hours": 100.0,
        }
        status, health, rul = monitor.assess(metrics)
        assert status == ComponentStatus.NOMINAL
        assert health > 0.7

    def test_overheated_motor_critical(self):
        monitor = MotorHealthMonitor()
        metrics = {
            "winding_temp_c": 105.0,
            "vibration_rms_ms2": 0.5,
            "phase_current_a": 10.0,
            "rpm": 2000.0,
            "total_run_hours": 500.0,
        }
        status, health, rul = monitor.assess(metrics)
        assert status == ComponentStatus.CRITICAL


class TestPredictiveMaintenanceEngine:

    def setup_method(self):
        self.engine = PredictiveMaintenanceEngine(robot_id="TEST-001")

    def test_empty_report(self):
        report = self.engine.generate_report()
        assert report.overall_status == ComponentStatus.UNKNOWN

    def test_nominal_battery_reading(self):
        reading = ComponentReading(
            component_id="battery_main",
            timestamp=time.time(),
            metrics={
                "cell_voltage_v": 3.2, "temperature_c": 25.0,
                "state_of_charge": 0.9, "cycle_count": 50,
                "capacity_ah": 100.0, "rated_capacity_ah": 100.0,
                "current_a": 5.0,
            }
        )
        self.engine.ingest_readings([reading])
        report = self.engine.generate_report()
        assert "battery_main" in report.components
        assert report.components["battery_main"].status == ComponentStatus.NOMINAL

    def test_critical_component_flags_mission(self):
        reading = ComponentReading(
            component_id="battery_main",
            timestamp=time.time(),
            metrics={
                "cell_voltage_v": 2.5,   # Below cutoff
                "temperature_c": 60.0,
                "state_of_charge": 0.02,
                "cycle_count": 1000,
                "capacity_ah": 60.0,
                "rated_capacity_ah": 100.0,
                "current_a": 10.0,
            }
        )
        self.engine.ingest_readings([reading])
        report = self.engine.generate_report()
        assert not report.mission_continue_recommended

    def test_telemetry_snapshot_structure(self):
        snap = self.engine.get_telemetry_snapshot()
        assert "maintenance" in snap
        assert "overall_status" in snap["maintenance"]
        assert "mission_continue" in snap["maintenance"]
