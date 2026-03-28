# ACER Robot — System Architecture

## Overview

ACER (Armored Combat & Engagement Robot) is structured as a modular, layered system with strict separation between perception, reasoning, actuation, and communications. All modules communicate via thread-safe internal APIs and ROS2 topics/actions for hardware interfaces.

---

## Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          OPERATOR LAYER                              │
│  Control Dashboard  │  Mission Planner  │  Telemetry Viewer          │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ AES-256 Encrypted Comms
┌─────────────────────────────▼───────────────────────────────────────┐
│                       COMMUNICATION LAYER                            │
│  CommunicationFramework  │  ChannelManager  │  HeartbeatWatchdog     │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│                       COGNITION LAYER                                │
│  ThreatAssessmentEngine  │  MissionPlanner  │  BehaviorTree          │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│                       PERCEPTION LAYER                               │
│  SensorFusionEngine  │  ThermalAnalyzer  │  AcousticLocalizer        │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│                       ACTUATION LAYER                                │
│  ACERNavigator  │  MobilityController  │  ArmamentController         │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│                       HARDWARE LAYER                                 │
│  VLP-32C  │  FLIR  │  Cameras  │  VN-300  │  Drive Motors  │  Radio  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Sensor → Threat Response (Latency Budget: 150ms)

```
LiDAR/Thermal/Optical/Acoustic
        │ (< 30ms)
        ▼
SensorFusionEngine.fuse()
        │ (< 50ms)
        ▼
ThreatAssessmentEngine.process_detection()
        │ (< 20ms)
        ▼
ThreatAssessmentEngine.assess()
        │ (< 10ms)
        ▼
ACERNavigator  +  CommunicationFramework
```

### Telemetry Flow (2 Hz)

```
All Subsystems
    │
    ▼
ACERRobot._build_telemetry()
    │
    ▼
CommunicationFramework.broadcast_telemetry()
    │ AES-256 encrypted
    ▼
Best Available Channel → Command Station
```

---

## Module Responsibilities

### `src/navigation/navigator.py`
- Wraps ROS2 Nav2 action client for goal-based navigation
- Implements VFH+ for reactive obstacle avoidance
- Manages patrol routes with waypoint queuing
- Publishes navigation state and emergency stop signals

### `src/sensors/sensor_fusion.py`
- Maintains per-sensor circular buffers (100 readings each)
- Runs EKF predict+update at main loop frequency
- Produces `FusedEnvironmentState` with unified position, detected objects, thermal hotspots, acoustic sources
- Thread-safe ingestion from any sensor publisher thread

### `src/threat_assessment/threat_engine.py`
- Classifies sensor detections via ONNX model or heuristic fallback
- Tracks up to 32 simultaneous threats with proximity-based fusion
- Escalates threat levels based on: category × confidence × distance × approach velocity
- Automatically expires stale tracks and computes evacuation vectors

### `src/communication/comm_framework.py`
- Manages three physical channels with priority-based failover
- All payloads encrypted with AES-256-GCM + HKDF key derivation
- ZeroMQ PUB/SUB for low-latency non-blocking message passing
- Heartbeat watchdog triggers safe mode after configurable timeout

### `src/maintenance/predictive_maintenance.py`
- Per-component monitors with physics-based health models
- Isolation Forest anomaly detection for generic components
- Remaining Useful Life (RUL) estimated per component
- Mission abort recommendation propagated to main controller

---

## Threading Model

```
Main Thread: ACERRobot._main_loop() @ 20 Hz
    ├── SensorFusionEngine.fuse()
    ├── ThreatAssessmentEngine.process/assess()
    └── MaintenanceEngine.generate_report()

Comms Receive Thread (daemon): CommunicationFramework._receive_loop()
Comms Telemetry Thread (daemon): CommunicationFramework._telemetry_loop()
Maintenance Watchdog (daemon): HeartbeatWatchdog._watch()
ROS2 Spin Thread: rclpy executor for nav/sensor callbacks
```

All cross-thread data access is protected by `threading.Lock()`. The main loop is the only writer to the fused environment state.

---

## Security Design

- **Encryption**: AES-256-GCM with fresh 96-bit nonce per message
- **Key Derivation**: HKDF-SHA256 from pre-shared master secret
- **Authentication**: GCM authentication tag validates message integrity
- **Channel Isolation**: Each channel uses independent socket connections
- **Dead-man**: Missing heartbeat for 15s triggers autonomous safe mode
- **Secret Management**: Master secret loaded from environment variable `ACER_COMM_SECRET`, never stored in config files

---

## Fault Tolerance

| Failure | Detection | Response |
|---|---|---|
| Sensor offline | Missing buffer updates | Reduce fusion confidence, alert operator |
| ONNX model missing | Import error at init | Fall back to heuristic classifier |
| Primary radio loss | Channel health monitor | Auto-failover to Wi-Fi/Iridium |
| Heartbeat timeout | Watchdog thread | Emergency stop + safe mode |
| Critical battery | Maintenance engine | Mission abort command |
| Motor failure | Anomaly + RUL threshold | Abort + emergency return |
| Compute overload | Loop overrun detection | Warning logged, non-critical tasks skipped |
