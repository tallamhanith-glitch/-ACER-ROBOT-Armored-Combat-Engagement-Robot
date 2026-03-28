"""
ACER Robot - Sensor Fusion Engine
===================================
Fuses data from LiDAR, thermal cameras, optical cameras, IMU,
and acoustic arrays into a unified environmental awareness model.

Uses an Extended Kalman Filter (EKF) for state estimation and
a probabilistic occupancy grid for environmental mapping.

Author: ACER Team
Date: March 2025
"""

import numpy as np
import threading
import time
import logging
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SensorType(Enum):
    LIDAR = "lidar"
    THERMAL = "thermal"
    OPTICAL = "optical"
    IMU = "imu"
    ACOUSTIC = "acoustic"
    GPS = "gps"


@dataclass
class SensorReading:
    sensor_type: SensorType
    timestamp: float
    data: Any
    confidence: float = 1.0
    sensor_id: str = ""


@dataclass
class FusedEnvironmentState:
    """Unified state of the robot's environment after sensor fusion."""
    timestamp: float
    robot_position: np.ndarray        # [x, y, z]
    robot_velocity: np.ndarray        # [vx, vy, vz]
    robot_orientation: np.ndarray     # [roll, pitch, yaw]
    detected_objects: List[Dict]      # List of detected entities
    thermal_hotspots: List[Dict]      # Heat signature locations
    acoustic_sources: List[Dict]      # Sound source directions
    occupancy_grid: Optional[np.ndarray] = None
    confidence_score: float = 0.0


class ExtendedKalmanFilter:
    """
    EKF for fusing IMU + odometry + GPS for robot state estimation.
    State vector: [x, y, z, vx, vy, vz, roll, pitch, yaw]
    """

    def __init__(self, state_dim: int = 9, obs_dim: int = 6):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.x = np.zeros(state_dim)           # State estimate
        self.P = np.eye(state_dim) * 0.1       # Error covariance
        self.Q = np.eye(state_dim) * 0.01      # Process noise
        self.R = np.eye(obs_dim) * 0.05        # Observation noise

    def predict(self, dt: float, control_input: Optional[np.ndarray] = None):
        """Predict step — propagate state forward in time."""
        F = self._state_transition_matrix(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z: np.ndarray, H: Optional[np.ndarray] = None):
        """Update step — correct prediction with observation z."""
        if H is None:
            H = np.eye(self.obs_dim, self.state_dim)

        y = z - H @ self.x                          # Innovation
        S = H @ self.P @ H.T + self.R               # Innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)         # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P

    def _state_transition_matrix(self, dt: float) -> np.ndarray:
        F = np.eye(self.state_dim)
        # Position += velocity * dt
        F[0, 3] = dt  # x += vx*dt
        F[1, 4] = dt  # y += vy*dt
        F[2, 5] = dt  # z += vz*dt
        return F

    @property
    def position(self) -> np.ndarray:
        return self.x[:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:6]

    @property
    def orientation(self) -> np.ndarray:
        return self.x[6:9]


class ThermalAnalyzer:
    """
    Analyzes thermal camera frames to detect heat signatures.
    Thresholds calibrated for human body temp (36-37°C) and
    vehicle exhaust signatures (200-600°C).
    """

    HUMAN_TEMP_MIN = 34.0
    HUMAN_TEMP_MAX = 40.0
    VEHICLE_TEMP_MIN = 100.0
    FIRE_TEMP_MIN = 400.0

    def detect_hotspots(self, thermal_frame: np.ndarray) -> List[Dict]:
        """
        Detect and classify thermal hotspots in a frame.
        Returns list of dicts: {type, centroid, temperature, area, confidence}
        """
        hotspots = []

        # Threshold for human signatures
        human_mask = (thermal_frame >= self.HUMAN_TEMP_MIN) & \
                     (thermal_frame <= self.HUMAN_TEMP_MAX)
        human_spots = self._extract_blobs(thermal_frame, human_mask, "human")
        hotspots.extend(human_spots)

        # Threshold for vehicle/engine signatures
        vehicle_mask = (thermal_frame >= self.VEHICLE_TEMP_MIN) & \
                       (thermal_frame < self.FIRE_TEMP_MIN)
        vehicle_spots = self._extract_blobs(thermal_frame, vehicle_mask, "vehicle")
        hotspots.extend(vehicle_spots)

        # Fire/explosion detection
        fire_mask = thermal_frame >= self.FIRE_TEMP_MIN
        fire_spots = self._extract_blobs(thermal_frame, fire_mask, "fire")
        hotspots.extend(fire_spots)

        return hotspots

    def _extract_blobs(
        self, frame: np.ndarray, mask: np.ndarray, label: str
    ) -> List[Dict]:
        """Simple connected-component blob extraction."""
        from scipy import ndimage  # type: ignore
        labeled, num_features = ndimage.label(mask)
        blobs = []
        for i in range(1, num_features + 1):
            region = labeled == i
            area = int(np.sum(region))
            if area < 5:
                continue
            coords = np.argwhere(region)
            centroid = coords.mean(axis=0).tolist()
            max_temp = float(frame[region].max())
            mean_temp = float(frame[region].mean())
            blobs.append({
                "type": label,
                "centroid": centroid,
                "area": area,
                "max_temperature": max_temp,
                "mean_temperature": mean_temp,
                "confidence": min(1.0, area / 50.0),
            })
        return blobs


class AcousticLocalizer:
    """
    Localizes sound sources using time-difference-of-arrival (TDOA)
    across an 8-element microphone array.
    Detects: gunshots, engine noise, human voice, explosions.
    """

    SPEED_OF_SOUND = 343.0  # m/s at 20°C

    def __init__(self, mic_positions: np.ndarray):
        """
        mic_positions: (8, 3) array of microphone XYZ positions in robot frame.
        """
        self.mic_positions = mic_positions

    def localize(self, audio_frames: np.ndarray) -> List[Dict]:
        """
        Estimate direction of arrival from cross-correlation of mic pairs.
        audio_frames: (8, N) array of synchronized audio samples.
        Returns list of {direction_deg, elevation_deg, confidence, event_type}
        """
        sources = []
        ref = audio_frames[0]

        tdoa_estimates = []
        for i in range(1, len(audio_frames)):
            tdoa = self._gcc_phat(ref, audio_frames[i])
            tdoa_estimates.append(tdoa)

        # Simplified direction estimate from first pair
        if tdoa_estimates:
            avg_tdoa = np.mean(tdoa_estimates[:4])
            baseline = np.linalg.norm(
                self.mic_positions[1] - self.mic_positions[0]
            )
            cos_theta = np.clip(avg_tdoa * self.SPEED_OF_SOUND / baseline, -1, 1)
            angle_rad = np.arccos(cos_theta)
            sources.append({
                "direction_deg": float(np.degrees(angle_rad)),
                "elevation_deg": 0.0,
                "tdoa_s": float(avg_tdoa),
                "confidence": 0.75,
                "event_type": "unknown",
            })
        return sources

    def _gcc_phat(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Generalized Cross-Correlation with Phase Transform for TDOA estimation."""
        n = len(sig1) + len(sig2) - 1
        f1 = np.fft.rfft(sig1, n=n)
        f2 = np.fft.rfft(sig2, n=n)
        cross = f1 * np.conj(f2)
        denom = np.abs(cross) + 1e-10
        gcc = np.fft.irfft(cross / denom, n=n)
        lag = int(np.argmax(np.abs(gcc))) - (len(sig1) - 1)
        return float(lag)  # In samples — caller must divide by sample_rate


class SensorFusionEngine:
    """
    Central sensor fusion engine for ACER Robot.
    
    Combines all sensor streams into a unified FusedEnvironmentState
    using EKF for state estimation, and probabilistic methods for
    object and threat detection.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.ekf = ExtendedKalmanFilter()
        self.thermal_analyzer = ThermalAnalyzer()

        # Default 8-mic circular array (radius 0.15m)
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        mic_pos = np.column_stack([
            0.15 * np.cos(angles),
            0.15 * np.sin(angles),
            np.zeros(8)
        ])
        self.acoustic_localizer = AcousticLocalizer(mic_pos)

        self._lock = threading.Lock()
        self._sensor_buffers: Dict[SensorType, List[SensorReading]] = {
            st: [] for st in SensorType
        }
        self._last_fusion_time = time.time()
        self._fused_state: Optional[FusedEnvironmentState] = None

        logger.info("SensorFusionEngine initialized")

    def ingest(self, reading: SensorReading):
        """Thread-safe ingestion of a new sensor reading."""
        with self._lock:
            buf = self._sensor_buffers[reading.sensor_type]
            buf.append(reading)
            if len(buf) > 100:
                buf.pop(0)

    def fuse(self) -> FusedEnvironmentState:
        """
        Perform sensor fusion and return unified environment state.
        Called at the system's main loop frequency (typically 10-20 Hz).
        """
        now = time.time()
        dt = now - self._last_fusion_time
        self._last_fusion_time = now

        with self._lock:
            # 1. EKF predict step
            self.ekf.predict(dt)

            # 2. Update EKF with IMU / odometry
            imu_readings = self._get_latest(SensorType.IMU)
            if imu_readings:
                obs = self._parse_imu(imu_readings[-1])
                self.ekf.update(obs)

            # 3. Thermal analysis
            thermal_hotspots = []
            thermal_readings = self._get_latest(SensorType.THERMAL)
            if thermal_readings:
                frame = thermal_readings[-1].data
                if isinstance(frame, np.ndarray) and frame.ndim == 2:
                    thermal_hotspots = self.thermal_analyzer.detect_hotspots(frame)

            # 4. Acoustic localization
            acoustic_sources = []
            acoustic_readings = self._get_latest(SensorType.ACOUSTIC)
            if acoustic_readings:
                audio = acoustic_readings[-1].data
                if isinstance(audio, np.ndarray) and audio.ndim == 2:
                    acoustic_sources = self.acoustic_localizer.localize(audio)

            # 5. Object detections from optical
            detected_objects = []
            optical_readings = self._get_latest(SensorType.OPTICAL)
            if optical_readings and optical_readings[-1].data:
                detected_objects = optical_readings[-1].data  # Pre-detected objects

            # 6. Compute overall confidence
            n_active = sum(
                1 for st in SensorType
                if self._sensor_buffers[st]
            )
            confidence = n_active / len(SensorType)

            self._fused_state = FusedEnvironmentState(
                timestamp=now,
                robot_position=self.ekf.position.copy(),
                robot_velocity=self.ekf.velocity.copy(),
                robot_orientation=self.ekf.orientation.copy(),
                detected_objects=detected_objects,
                thermal_hotspots=thermal_hotspots,
                acoustic_sources=acoustic_sources,
                confidence_score=confidence,
            )

        return self._fused_state

    def get_latest_state(self) -> Optional[FusedEnvironmentState]:
        return self._fused_state

    def _get_latest(self, sensor_type: SensorType) -> List[SensorReading]:
        return self._sensor_buffers.get(sensor_type, [])

    def _parse_imu(self, reading: SensorReading) -> np.ndarray:
        """Extract 6-DOF observation vector from IMU reading."""
        data = reading.data
        if isinstance(data, dict):
            return np.array([
                data.get('ax', 0), data.get('ay', 0), data.get('az', 0),
                data.get('gx', 0), data.get('gy', 0), data.get('gz', 0),
            ])
        return np.zeros(6)
