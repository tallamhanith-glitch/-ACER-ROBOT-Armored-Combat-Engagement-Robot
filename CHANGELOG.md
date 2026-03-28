# Changelog

All notable changes to ACER Robot are documented here.

## [1.0.0] — March 2025

### Added
- **Navigation Module**: SLAM-based autonomous navigation with Nav2 integration
  - Vector Field Histogram+ (VFH+) obstacle avoidance
  - Dynamic replanning on map updates
  - Emergency stop protocol tied to threat alerts
  - Patrol route execution with configurable looping

- **Sensor Fusion Engine**: Multi-modal sensor integration
  - Extended Kalman Filter (EKF) for IMU + odometry + GPS state estimation
  - Thermal hotspot detection (human, vehicle, fire classification)
  - Acoustic source localization using GCC-PHAT TDOA on 8-mic array
  - Unified `FusedEnvironmentState` output at configurable frequency

- **Threat Assessment Engine**: ML-based real-time threat classification
  - ONNX Runtime inference with TensorRT acceleration
  - Multi-target tracking with proximity-based track fusion
  - Threat level escalation based on distance, velocity, and confidence
  - Evacuation vector computation for evasion planning
  - Heuristic fallback classifier when model unavailable

- **Secure Communication Framework**: Encrypted multi-channel comms
  - AES-256-GCM end-to-end encryption with HKDF-SHA256 key derivation
  - ZeroMQ pub/sub architecture for telemetry and command channels
  - Automatic channel failover: RFD900x → Wi-Fi mesh → Iridium satellite
  - Heartbeat watchdog with dead-man protocol (15s default timeout)
  - 2 Hz telemetry broadcast with operator command reception

- **Predictive Maintenance Engine**: AI-driven component health monitoring
  - Battery health monitor: voltage, SoC, temperature, cycle life
  - Motor health monitor with Isolation Forest anomaly detection
  - Remaining Useful Life (RUL) estimation per component
  - Mission abort recommendation on critical component failure
  - Telemetry snapshot integration for remote health monitoring

- **Main System Controller**: Top-level orchestrator at 20 Hz control loop
  - Integrates all subsystems with thread-safe operation
  - Mode switching: autonomous, teleoperated, hybrid, safe, emergency
  - YAML-driven configuration
  - Graceful shutdown with SIGINT/SIGTERM handling

- **Hardware Integration**:
  - Velodyne VLP-32C LiDAR driver interface
  - FLIR Lepton 3.5 thermal camera pipeline
  - VectorNav VN-300 IMU integration
  - RFD900x serial radio interface
  - Iridium 9603 satellite modem fallback

- **Tooling**:
  - Pre-mission sensor verification script (`scripts/sensor_check.py`)
  - Hardware initialization script (`scripts/hardware_init.sh`)
  - Calibration utility (`scripts/calibrate.py`)
  - GitHub Actions CI with pytest coverage reporting

### Configuration
- Full YAML config: `config/acer_config.yaml`
- Default patrol waypoints
- Per-channel communication settings
- Maintenance thresholds for all monitored components
