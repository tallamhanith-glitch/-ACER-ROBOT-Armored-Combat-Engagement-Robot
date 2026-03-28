"""
Tests for ACER Robot Threat Assessment Engine
"""

import pytest
import numpy as np
import time
from unittest.mock import patch, MagicMock

from src.threat_assessment.threat_engine import (
    ThreatAssessmentEngine,
    ThreatClassifier,
    ThreatPrioritizer,
    ThreatLevel,
    ThreatCategory,
    ThreatObject,
)


class TestThreatClassifier:

    def test_classify_returns_category_and_confidence(self):
        clf = ThreatClassifier.__new__(ThreatClassifier)
        clf._session = None
        clf.MODEL_INPUT_DIM = 512
        clf.CONFIDENCE_THRESHOLD = 0.85

        features = np.zeros(512)
        features[384:512] = 0.9  # High acoustic impulse → explosive

        category, confidence = clf._heuristic_classify(features)
        assert category == ThreatCategory.EXPLOSIVE
        assert 0.0 <= confidence <= 1.0

    def test_classify_human_signature(self):
        clf = ThreatClassifier.__new__(ThreatClassifier)
        clf._session = None

        features = np.zeros(512)
        features[:256] = 0.6   # High visual activity

        category, confidence = clf._heuristic_classify(features)
        assert category == ThreatCategory.PERSONNEL

    def test_classify_unknown_signature(self):
        clf = ThreatClassifier.__new__(ThreatClassifier)
        clf._session = None

        features = np.zeros(512)
        category, confidence = clf._heuristic_classify(features)
        assert category == ThreatCategory.UNKNOWN

    def test_pad_or_trim_short(self):
        arr = np.ones(100)
        result = ThreatClassifier._pad_or_trim(arr, 512)
        assert len(result) == 512
        assert result[99] == 1.0
        assert result[100] == 0.0

    def test_pad_or_trim_long(self):
        arr = np.ones(600)
        result = ThreatClassifier._pad_or_trim(arr, 512)
        assert len(result) == 512


class TestThreatPrioritizer:

    def setup_method(self):
        self.prio = ThreatPrioritizer()

    def test_personnel_moderate_at_distance(self):
        level = self.prio.prioritize(ThreatCategory.PERSONNEL, 0.9, 50.0)
        assert level == ThreatLevel.MODERATE

    def test_personnel_boosts_when_close(self):
        level = self.prio.prioritize(ThreatCategory.PERSONNEL, 0.9, 5.0)
        assert level.value >= ThreatLevel.CRITICAL.value

    def test_projectile_always_imminent(self):
        level = self.prio.prioritize(ThreatCategory.PROJECTILE, 0.9, 100.0)
        assert level == ThreatLevel.IMMINENT

    def test_low_confidence_reduces_level(self):
        level_high_conf = self.prio.prioritize(ThreatCategory.VEHICLE, 0.9, 50.0)
        level_low_conf = self.prio.prioritize(ThreatCategory.VEHICLE, 0.3, 50.0)
        assert level_low_conf.value <= level_high_conf.value

    def test_approaching_target_boosts_level(self):
        velocity = np.array([10.0, 0.0, 0.0])
        level_static = self.prio.prioritize(ThreatCategory.VEHICLE, 0.8, 50.0)
        level_moving = self.prio.prioritize(ThreatCategory.VEHICLE, 0.8, 50.0, velocity)
        assert level_moving.value >= level_static.value

    def test_protocol_for_each_level(self):
        for level in ThreatLevel:
            protocol = self.prio.get_protocol(level)
            assert isinstance(protocol, str) and len(protocol) > 0


class TestThreatAssessmentEngine:

    def setup_method(self):
        self.engine = ThreatAssessmentEngine(config={"model_path": "nonexistent.onnx"})

    def test_process_detection_creates_threat(self):
        features = np.zeros(512)
        features[384:512] = 0.95  # Explosive signature
        position = np.array([10.0, 5.0, 0.0])
        robot_pos = np.array([0.0, 0.0, 0.0])

        threat = self.engine.process_detection(features, position, robot_position=robot_pos)
        assert threat is not None
        assert threat.id.startswith("THREAT-")
        assert threat.distance_m == pytest.approx(11.18, abs=0.1)

    def test_tracking_updates_existing_threat(self):
        features = np.zeros(512)
        pos1 = np.array([10.0, 0.0, 0.0])
        pos2 = np.array([10.5, 0.0, 0.0])  # Close — same track

        t1 = self.engine.process_detection(features, pos1)
        t2 = self.engine.process_detection(features, pos2)
        assert t1.id == t2.id

    def test_assess_returns_highest_level(self):
        features_crit = np.zeros(512)
        features_crit[384:512] = 0.95
        pos = np.array([5.0, 0.0, 0.0])
        self.engine.process_detection(features_crit, pos, robot_position=np.zeros(3))

        result = self.engine.assess()
        assert result.highest_threat_level.value >= ThreatLevel.HIGH.value

    def test_clear_threat_removes_it(self):
        features = np.zeros(512)
        threat = self.engine.process_detection(features, np.array([20.0, 0.0, 0.0]))
        tid = threat.id
        self.engine.clear_threat(tid)
        active = self.engine.get_active_threats()
        assert all(t.id != tid for t in active)

    def test_threat_expires_after_timeout(self):
        self.engine.THREAT_TIMEOUT_S = 0.05  # Very short for test
        features = np.zeros(512)
        self.engine.process_detection(features, np.array([30.0, 0.0, 0.0]))
        time.sleep(0.1)
        result = self.engine.assess()
        assert len(result.active_threats) == 0

    def test_evacuation_vector_computed(self):
        features = np.zeros(512)
        self.engine.process_detection(
            features, np.array([5.0, 0.0, 0.0]),
            robot_position=np.array([0.0, 0.0, 0.0])
        )
        result = self.engine.assess(robot_position=np.array([0.0, 0.0, 0.0]))
        if result.active_threats:
            assert result.evacuation_vector is not None
            assert abs(np.linalg.norm(result.evacuation_vector) - 1.0) < 0.01

    def test_no_threats_gives_none_level(self):
        fresh_engine = ThreatAssessmentEngine(config={"model_path": "x.onnx"})
        result = fresh_engine.assess()
        assert result.highest_threat_level == ThreatLevel.NONE
        assert not result.engagement_authorized
