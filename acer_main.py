"""
ACER Robot - Main System Controller
=====================================
Top-level orchestrator that integrates all ACER subsystems:
navigation, sensor fusion, threat assessment, communication,
and predictive maintenance into a unified operational loop.

Author: ACER Team
Date: March 2025
"""

import time
import threading
import logging
import signal
import sys
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

from src.navigation.navigator import ACERNavigator, NavigationConfig, NavigationGoal
from src.sensors.sensor_fusion import SensorFusionEngine, SensorReading, SensorType
from src.threat_assessment.threat_engine import ThreatAssessmentEngine, ThreatLevel
from src.communication.comm_framework import CommunicationFramework, Message, MessageType
from src.maintenance.predictive_maintenance import (
    PredictiveMaintenanceEngine, ComponentReading
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("ACER.Main")


class OperationMode:
    AUTONOMOUS = "autonomous"
    TELEOPERATED = "teleoperated"
    HYBRID = "hybrid"
    SAFE = "safe"
    EMERGENCY = "emergency"


class ACERRobot:
    """
    ACER Robot Top-Level Controller
    
    Orchestrates all subsystems in a main control loop running at
    configurable frequency (default: 20 Hz).
    """

    LOOP_FREQUENCY_HZ = 20
    LOOP_PERIOD_S = 1.0 / LOOP_FREQUENCY_HZ

    def __init__(self, config_path: str = "config/acer_config.yaml"):
        self.config = self._load_config(config_path)
        robot_cfg = self.config.get("robot", {})

        self.robot_id: str = robot_cfg.get("id", "ACER-001")
        self.mode: str = robot_cfg.get("mode", OperationMode.AUTONOMOUS)
        self._running = False
        self._loop_thread: Optional[threading.Thread] = None

        logger.info(f"Initializing {self.robot_id} | Mode: {self.mode}")

        # ── Subsystem initialization ──────────────────────────────
        nav_cfg = self.config.get("navigation", {})
        self.navigation = ACERNavigator(NavigationConfig(
            max_linear_speed=nav_cfg.get("max_speed", 2.5),
            obstacle_clearance=nav_cfg.get("obstacle_clearance", 0.5),
        ))

        self.sensor_fusion = SensorFusionEngine(
            self.config.get("sensors", {})
        )

        self.threat_engine = ThreatAssessmentEngine(
            self.config.get("threat_assessment", {})
        )

        comm_cfg = self.config.get("communication", {})
        secret = comm_cfg.get("master_secret", "default-insecure-secret").encode()
        self.comms = CommunicationFramework(
            robot_id=self.robot_id,
            master_secret=secret,
            command_server_addr=comm_cfg.get("command_server", "tcp://127.0.0.1:5555"),
            heartbeat_timeout_s=comm_cfg.get("heartbeat_timeout_s", 15.0),
            on_command=self._handle_command,
            on_heartbeat_timeout=self._on_heartbeat_timeout,
        )

        self.maintenance = PredictiveMaintenanceEngine(self.robot_id)

        # Register telemetry provider
        self.comms.register_telemetry_provider(self._build_telemetry)

        # Graceful shutdown hooks
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"{self.robot_id} initialization complete.")

    # ─────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────

    def start(self):
        """Start all subsystems and main control loop."""
        logger.info("Starting ACER Robot systems...")
        self.comms.start()
        self._running = True
        self._loop_thread = threading.Thread(
            target=self._main_loop, daemon=False, name="ACER-MainLoop"
        )
        self._loop_thread.start()
        logger.info(f"{self.robot_id} operational at {self.LOOP_FREQUENCY_HZ} Hz")

        if self.mode == OperationMode.AUTONOMOUS:
            self._start_default_patrol()

    def stop(self):
        """Graceful shutdown of all subsystems."""
        logger.info("Shutting down ACER Robot...")
        self._running = False
        if self._loop_thread:
            self._loop_thread.join(timeout=5.0)
        self.comms.stop()
        logger.info("ACER Robot stopped.")

    def _main_loop(self):
        """
        Main control loop running at LOOP_FREQUENCY_HZ.
        Executes: sensor fusion → threat assessment → maintenance check → telemetry
        """
        while self._running:
            t_start = time.time()

            try:
                # 1. Fuse sensor data
                env_state = self.sensor_fusion.fuse()

                # 2. Process thermal/acoustic detections into threat engine
                for hotspot in env_state.thermal_hotspots:
                    if hotspot.get("confidence", 0) > 0.6:
                        import numpy as np
                        pos = np.array(hotspot.get("centroid", [0, 0, 0]) + [0])[:3]
                        features = np.zeros(512)
                        features[256] = hotspot.get("mean_temperature", 0) / 100.0
                        self.threat_engine.process_detection(
                            features, pos, robot_position=env_state.robot_position
                        )

                # 3. Get threat assessment
                assessment = self.threat_engine.assess(env_state.robot_position)

                # 4. React to critical threats
                if assessment.highest_threat_level == ThreatLevel.IMMINENT:
                    self._handle_imminent_threat(assessment)
                elif assessment.highest_threat_level == ThreatLevel.CRITICAL:
                    self.comms.send_threat_alert({
                        "level": assessment.highest_threat_level.name,
                        "action": assessment.recommended_action,
                        "threats": len(assessment.active_threats),
                    })

                # 5. Maintenance check (every 5 seconds)
                if int(time.time()) % 5 == 0:
                    maint_report = self.maintenance.generate_report()
                    if not maint_report.mission_continue_recommended:
                        logger.critical("Maintenance system advises mission abort")
                        self.comms.send_emergency("maintenance_abort")

            except Exception as e:
                logger.error(f"Main loop error: {e}", exc_info=True)

            # Maintain loop frequency
            elapsed = time.time() - t_start
            sleep_time = self.LOOP_PERIOD_S - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif elapsed > self.LOOP_PERIOD_S * 2:
                logger.warning(
                    f"Main loop overrun: {elapsed*1000:.1f}ms "
                    f"(target {self.LOOP_PERIOD_S*1000:.0f}ms)"
                )

    # ─────────────────────────────────────────────────────────────
    # Command Handling
    # ─────────────────────────────────────────────────────────────

    def _handle_command(self, msg: Message):
        """Route incoming commands to appropriate subsystems."""
        cmd = msg.payload.get("command", "")
        params = msg.payload.get("params", {})

        logger.info(f"Executing command: {cmd} | params={params}")

        if cmd == "NAVIGATE_TO":
            goal = NavigationGoal(
                x=params["x"], y=params["y"],
                yaw=params.get("yaw", 0.0)
            )
            self.navigation.navigate_to(goal)

        elif cmd == "SET_MODE":
            new_mode = params.get("mode", OperationMode.SAFE)
            self._set_mode(new_mode)

        elif cmd == "EMERGENCY_STOP":
            self.navigation._emergency_stop()
            self.mode = OperationMode.EMERGENCY

        elif cmd == "START_PATROL":
            waypoints = [
                NavigationGoal(x=wp["x"], y=wp["y"])
                for wp in params.get("waypoints", [])
            ]
            if waypoints:
                self.navigation.execute_patrol(waypoints, loop=params.get("loop", True))

        elif cmd == "CLEAR_THREAT":
            self.threat_engine.clear_threat(params.get("threat_id", ""))

        elif cmd == "GET_STATUS":
            self.comms.send(Message(
                msg_type=MessageType.STATUS,
                payload=self._build_telemetry(),
                robot_id=self.robot_id
            ))

        else:
            logger.warning(f"Unknown command: {cmd}")

    def _handle_imminent_threat(self, assessment):
        """Execute emergency evasion on imminent threat detection."""
        import numpy as np
        logger.critical(f"IMMINENT THREAT — executing evasion | {assessment.recommended_action}")
        self.comms.send_threat_alert({
            "level": "IMMINENT",
            "action": assessment.recommended_action,
            "engagement_authorized": assessment.engagement_authorized,
        })

        if assessment.evacuation_vector is not None:
            evac = assessment.evacuation_vector
            goal = NavigationGoal(
                x=float(evac[0] * 15),
                y=float(evac[1] * 15),
            )
            self.navigation.navigate_to(goal)

    def _on_heartbeat_timeout(self):
        """Dead-man protocol — halt all motion, enter safe mode."""
        logger.critical("Heartbeat timeout — entering SAFE mode")
        self.mode = OperationMode.SAFE
        self.navigation._emergency_stop()

    def _set_mode(self, new_mode: str):
        logger.info(f"Mode change: {self.mode} → {new_mode}")
        self.mode = new_mode

    # ─────────────────────────────────────────────────────────────
    # Telemetry
    # ─────────────────────────────────────────────────────────────

    def _build_telemetry(self) -> Dict[str, Any]:
        env = self.sensor_fusion.get_latest_state()
        maint = self.maintenance.get_telemetry_snapshot()

        telemetry = {
            "robot_id": self.robot_id,
            "mode": self.mode,
            "navigation_state": self.navigation.get_current_state().value,
            "active_threats": len(self.threat_engine.get_active_threats()),
        }

        if env:
            telemetry["position"] = env.robot_position.tolist()
            telemetry["velocity"] = env.robot_velocity.tolist()
            telemetry["orientation"] = env.robot_orientation.tolist()
            telemetry["sensor_confidence"] = round(env.confidence_score, 3)

        telemetry.update(maint)
        telemetry["channel_health"] = self.comms.get_channel_health()
        return telemetry

    # ─────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────

    def _start_default_patrol(self):
        """Load and execute default patrol route from config."""
        waypoints_cfg = self.config.get("default_patrol", {}).get("waypoints", [])
        if waypoints_cfg:
            waypoints = [
                NavigationGoal(x=wp["x"], y=wp["y"], yaw=wp.get("yaw", 0.0))
                for wp in waypoints_cfg
            ]
            self.navigation.execute_patrol(waypoints, loop=True)
            logger.info(f"Default patrol started with {len(waypoints)} waypoints")

    def _load_config(self, path: str) -> Dict:
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Config file not found: {path} — using defaults")
            return {}

    def _signal_handler(self, sig, frame):
        logger.info(f"Signal {sig} received — shutting down")
        self.stop()
        sys.exit(0)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ACER Robot Main Controller")
    parser.add_argument("--config", default="config/acer_config.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()

    robot = ACERRobot(config_path=args.config)
    robot.start()

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        robot.stop()


if __name__ == "__main__":
    main()
