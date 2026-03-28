"""
ACER Robot - Threat Assessment & Classification Module
========================================================
Real-time ML-based threat detection, classification, and
prioritized response protocol generation.

Uses ONNX Runtime for low-latency inference of a pre-trained
multi-class threat classifier. Supports fusion of visual,
thermal, and acoustic signals for compound threat scoring.

Author: ACER Team
Date: March 2025
"""

import numpy as np
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
from collections import deque
import threading

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    NONE = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4
    IMMINENT = 5


class ThreatCategory(Enum):
    UNKNOWN = "unknown"
    PERSONNEL = "armed_personnel"
    VEHICLE = "hostile_vehicle"
    PROJECTILE = "incoming_projectile"
    IED = "ied_device"
    DRONE = "hostile_drone"
    CHEMICAL = "chemical_hazard"
    EXPLOSIVE = "explosion_event"


@dataclass
class ThreatObject:
    """Represents a single detected and classified threat."""
    id: str
    category: ThreatCategory
    threat_level: ThreatLevel
    confidence: float
    position: np.ndarray              # [x, y, z] in world frame
    velocity: Optional[np.ndarray]    # [vx, vy, vz] m/s
    distance_m: float
    bearing_deg: float
    first_detected: float             # Epoch timestamp
    last_updated: float
    response_protocol: str = ""
    neutralized: bool = False

    @property
    def time_to_impact(self) -> Optional[float]:
        """Estimate TtI for projectile threats."""
        if self.velocity is not None and self.distance_m > 0:
            speed = float(np.linalg.norm(self.velocity))
            if speed > 0.1:
                return self.distance_m / speed
        return None


@dataclass
class ThreatAssessmentResult:
    timestamp: float
    active_threats: List[ThreatObject]
    highest_threat_level: ThreatLevel
    recommended_action: str
    evacuation_vector: Optional[np.ndarray]
    engagement_authorized: bool


class ThreatClassifier:
    """
    ONNX-based multi-class threat classifier.
    
    Input: (1, 512) feature vector combining:
      - Visual detection features (256-dim)
      - Thermal signature features (128-dim)
      - Acoustic fingerprint features (128-dim)
    
    Output: Probability distribution over ThreatCategory classes.
    """

    MODEL_INPUT_DIM = 512
    CONFIDENCE_THRESHOLD = 0.85

    def __init__(self, model_path: str = "models/threat_classifier_v2.onnx"):
        self.model_path = model_path
        self._session = None
        self._load_model()

    def _load_model(self):
        try:
            import onnxruntime as ort  # type: ignore
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                self.model_path,
                sess_options=opts,
                providers=["TensorrtExecutionProvider", "CUDAExecutionProvider",
                           "CPUExecutionProvider"]
            )
            logger.info(f"Threat classifier loaded: {self.model_path}")
        except Exception as e:
            logger.warning(f"ONNX model load failed ({e}) — using heuristic classifier")
            self._session = None

    def classify(self, feature_vector: np.ndarray) -> Tuple[ThreatCategory, float]:
        """
        Classify a threat from a feature vector.
        Returns (category, confidence).
        """
        if len(feature_vector) != self.MODEL_INPUT_DIM:
            feature_vector = self._pad_or_trim(feature_vector, self.MODEL_INPUT_DIM)

        if self._session is not None:
            return self._onnx_infer(feature_vector)
        return self._heuristic_classify(feature_vector)

    def _onnx_infer(self, features: np.ndarray) -> Tuple[ThreatCategory, float]:
        inp = features.astype(np.float32).reshape(1, -1)
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: inp})
        probs = outputs[0][0]
        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        categories = list(ThreatCategory)
        category = categories[best_idx % len(categories)]
        return category, confidence

    def _heuristic_classify(self, features: np.ndarray) -> Tuple[ThreatCategory, float]:
        """Fallback rule-based classifier when ONNX model is unavailable."""
        # Thermal band (256:384) — high values = heat signature
        thermal_energy = float(np.mean(np.abs(features[256:384])))
        # Acoustic band (384:512) — impulse = gunshot/explosion
        acoustic_impulse = float(np.max(np.abs(features[384:512])))

        if acoustic_impulse > 0.8:
            return ThreatCategory.EXPLOSIVE, 0.72
        elif thermal_energy > 0.6:
            return ThreatCategory.VEHICLE, 0.65
        elif float(np.mean(np.abs(features[:256]))) > 0.5:
            return ThreatCategory.PERSONNEL, 0.60
        return ThreatCategory.UNKNOWN, 0.3

    @staticmethod
    def _pad_or_trim(arr: np.ndarray, target: int) -> np.ndarray:
        if len(arr) >= target:
            return arr[:target]
        return np.pad(arr, (0, target - len(arr)))


class ThreatPrioritizer:
    """
    Assigns threat levels and response priorities based on
    category, confidence, distance, velocity, and history.
    """

    CATEGORY_BASE_LEVEL: Dict[ThreatCategory, ThreatLevel] = {
        ThreatCategory.UNKNOWN: ThreatLevel.LOW,
        ThreatCategory.PERSONNEL: ThreatLevel.MODERATE,
        ThreatCategory.VEHICLE: ThreatLevel.HIGH,
        ThreatCategory.PROJECTILE: ThreatLevel.IMMINENT,
        ThreatCategory.IED: ThreatLevel.CRITICAL,
        ThreatCategory.DRONE: ThreatLevel.HIGH,
        ThreatCategory.CHEMICAL: ThreatLevel.CRITICAL,
        ThreatCategory.EXPLOSIVE: ThreatLevel.IMMINENT,
    }

    RESPONSE_PROTOCOLS: Dict[ThreatLevel, str] = {
        ThreatLevel.NONE: "CONTINUE_MISSION",
        ThreatLevel.LOW: "INCREASE_SURVEILLANCE",
        ThreatLevel.MODERATE: "ALERT_OPERATOR_MANEUVER",
        ThreatLevel.HIGH: "DEFENSIVE_POSTURE_REPORT",
        ThreatLevel.CRITICAL: "EMERGENCY_MANEUVER_REQUEST_SUPPORT",
        ThreatLevel.IMMINENT: "IMMEDIATE_EVASION_OR_NEUTRALIZE",
    }

    def prioritize(
        self,
        category: ThreatCategory,
        confidence: float,
        distance_m: float,
        velocity: Optional[np.ndarray] = None,
    ) -> ThreatLevel:
        base = self.CATEGORY_BASE_LEVEL.get(category, ThreatLevel.LOW)
        level_val = base.value

        # Boost for close proximity
        if distance_m < 10.0:
            level_val = min(ThreatLevel.IMMINENT.value, level_val + 2)
        elif distance_m < 30.0:
            level_val = min(ThreatLevel.IMMINENT.value, level_val + 1)

        # Boost for approaching targets
        if velocity is not None:
            speed = float(np.linalg.norm(velocity))
            if speed > 5.0:
                level_val = min(ThreatLevel.IMMINENT.value, level_val + 1)

        # Reduce for low confidence
        if confidence < 0.5:
            level_val = max(ThreatLevel.NONE.value, level_val - 1)

        return ThreatLevel(level_val)

    def get_protocol(self, level: ThreatLevel) -> str:
        return self.RESPONSE_PROTOCOLS.get(level, "HOLD")


class ThreatAssessmentEngine:
    """
    Main threat assessment engine — integrates classifier, prioritizer,
    and multi-target tracking with threat memory.
    """

    THREAT_TIMEOUT_S = 10.0   # Remove threats not updated for this long
    MAX_TRACKED_THREATS = 32

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.classifier = ThreatClassifier(
            model_path=self.config.get("model_path", "models/threat_classifier_v2.onnx")
        )
        self.prioritizer = ThreatPrioritizer()
        self._active_threats: Dict[str, ThreatObject] = {}
        self._threat_history: deque = deque(maxlen=500)
        self._lock = threading.Lock()
        self._threat_counter = 0
        logger.info("ThreatAssessmentEngine online")

    def process_detection(
        self,
        feature_vector: np.ndarray,
        position: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        robot_position: Optional[np.ndarray] = None,
    ) -> ThreatObject:
        """
        Process a new sensor detection and return a ThreatObject.
        Automatically tracks and updates existing threats by proximity.
        """
        category, confidence = self.classifier.classify(feature_vector)

        distance_m = float(np.linalg.norm(position - (robot_position or np.zeros(3))))
        bearing_deg = self._compute_bearing(position, robot_position)
        level = self.prioritizer.prioritize(category, confidence, distance_m, velocity)
        protocol = self.prioritizer.get_protocol(level)

        now = time.time()

        with self._lock:
            # Check if this matches an existing track
            existing_id = self._find_existing_track(position)

            if existing_id:
                threat = self._active_threats[existing_id]
                threat.position = position
                threat.velocity = velocity
                threat.confidence = max(threat.confidence, confidence)
                threat.threat_level = level
                threat.last_updated = now
                threat.response_protocol = protocol
            else:
                self._threat_counter += 1
                threat_id = f"THREAT-{self._threat_counter:04d}"
                threat = ThreatObject(
                    id=threat_id,
                    category=category,
                    threat_level=level,
                    confidence=confidence,
                    position=position,
                    velocity=velocity,
                    distance_m=distance_m,
                    bearing_deg=bearing_deg,
                    first_detected=now,
                    last_updated=now,
                    response_protocol=protocol,
                )
                self._active_threats[threat_id] = threat
                logger.warning(
                    f"New threat [{threat_id}]: {category.value} "
                    f"| Level={level.name} | Conf={confidence:.2f} "
                    f"| Dist={distance_m:.1f}m | Protocol={protocol}"
                )

            self._threat_history.append(dict(
                id=threat.id, category=category.value,
                level=level.name, confidence=confidence,
                timestamp=now
            ))

        return threat

    def assess(self, robot_position: Optional[np.ndarray] = None) -> ThreatAssessmentResult:
        """
        Generate a full threat assessment result for the current moment.
        Expires stale threats and computes recommended action.
        """
        now = time.time()
        with self._lock:
            # Expire stale threats
            expired = [
                tid for tid, t in self._active_threats.items()
                if (now - t.last_updated) > self.THREAT_TIMEOUT_S
            ]
            for tid in expired:
                logger.info(f"Threat {tid} expired")
                del self._active_threats[tid]

            active = list(self._active_threats.values())

        highest = ThreatLevel.NONE
        for t in active:
            if t.threat_level.value > highest.value:
                highest = t.threat_level

        action = self.prioritizer.get_protocol(highest)
        evac_vector = self._compute_evacuation_vector(active, robot_position)
        engagement = highest.value >= ThreatLevel.HIGH.value

        return ThreatAssessmentResult(
            timestamp=now,
            active_threats=active,
            highest_threat_level=highest,
            recommended_action=action,
            evacuation_vector=evac_vector,
            engagement_authorized=engagement,
        )

    def _find_existing_track(self, position: np.ndarray, radius: float = 3.0) -> Optional[str]:
        for tid, threat in self._active_threats.items():
            dist = float(np.linalg.norm(threat.position - position))
            if dist < radius:
                return tid
        return None

    def _compute_evacuation_vector(
        self,
        threats: List[ThreatObject],
        robot_pos: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if not threats or robot_pos is None:
            return None
        # Sum of repulsion vectors from all threats, weighted by level
        evac = np.zeros(3)
        for t in threats:
            diff = robot_pos - t.position
            dist = np.linalg.norm(diff) + 0.01
            weight = t.threat_level.value / ThreatLevel.IMMINENT.value
            evac += (diff / dist) * weight
        norm = np.linalg.norm(evac)
        return evac / norm if norm > 0.01 else None

    @staticmethod
    def _compute_bearing(target: np.ndarray, origin: Optional[np.ndarray]) -> float:
        if origin is None:
            origin = np.zeros(3)
        dx = target[0] - origin[0]
        dy = target[1] - origin[1]
        return float(np.degrees(np.arctan2(dy, dx)) % 360)

    def get_active_threats(self) -> List[ThreatObject]:
        with self._lock:
            return list(self._active_threats.values())

    def clear_threat(self, threat_id: str):
        with self._lock:
            if threat_id in self._active_threats:
                self._active_threats[threat_id].neutralized = True
                del self._active_threats[threat_id]
                logger.info(f"Threat {threat_id} cleared")
