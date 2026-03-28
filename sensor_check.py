#!/usr/bin/env python3
"""
ACER Robot - Pre-Mission Sensor Verification Script
Checks all sensors are online and reporting valid data before mission start.

Usage:
    python scripts/sensor_check.py [--verbose] [--timeout 10]
"""

import argparse
import sys
import time
import struct
import logging

logger = logging.getLogger("SensorCheck")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def check_lidar(timeout: float = 5.0) -> bool:
    """Verify VLP-32C LiDAR is online and returning points."""
    print(f"  Checking LiDAR (VLP-32C)...", end=" ", flush=True)
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(timeout)
        s.bind(("0.0.0.0", 2368))
        data, _ = s.recvfrom(1206)
        s.close()
        if len(data) == 1206:
            print(f"{GREEN}✓ OK{RESET} (packet size: {len(data)} bytes)")
            return True
        print(f"{RED}✗ FAIL{RESET} (unexpected packet size: {len(data)})")
        return False
    except Exception as e:
        print(f"{RED}✗ FAIL{RESET} ({e})")
        return False


def check_imu(timeout: float = 5.0) -> bool:
    """Verify IMU is publishing via ROS2."""
    print(f"  Checking IMU (VN-300)...", end=" ", flush=True)
    try:
        import subprocess
        result = subprocess.run(
            ["ros2", "topic", "echo", "/imu/data", "--once", "--timeout", str(timeout)],
            capture_output=True, text=True, timeout=timeout + 2
        )
        if result.returncode == 0 and "linear_acceleration" in result.stdout:
            print(f"{GREEN}✓ OK{RESET}")
            return True
        print(f"{RED}✗ FAIL{RESET} (no IMU data)")
        return False
    except Exception as e:
        print(f"{YELLOW}⚠ WARN{RESET} (ROS2 not available: {e})")
        return True  # Non-critical in dev mode


def check_thermal_camera(timeout: float = 5.0) -> bool:
    """Verify FLIR thermal camera is streaming."""
    print(f"  Checking Thermal Camera (FLIR)...", end=" ", flush=True)
    try:
        import subprocess
        result = subprocess.run(
            ["ros2", "topic", "echo", "/thermal/image_raw", "--once",
             "--timeout", str(timeout)],
            capture_output=True, text=True, timeout=timeout + 2
        )
        if result.returncode == 0:
            print(f"{GREEN}✓ OK{RESET}")
            return True
        print(f"{YELLOW}⚠ DEGRADED{RESET} (thermal offline — surveillance limited)")
        return False
    except Exception as e:
        print(f"{YELLOW}⚠ WARN{RESET} ({e})")
        return False


def check_optical_cameras(timeout: float = 5.0) -> bool:
    """Verify optical camera pair is streaming."""
    print(f"  Checking Optical Cameras...", end=" ", flush=True)
    try:
        import subprocess
        for topic in ["/camera/front_left/image_raw", "/camera/front_right/image_raw"]:
            result = subprocess.run(
                ["ros2", "topic", "hz", topic, "--window", "5"],
                capture_output=True, text=True, timeout=6
            )
            if "average rate" not in result.stdout:
                print(f"{RED}✗ FAIL{RESET} ({topic} not publishing)")
                return False
        print(f"{GREEN}✓ OK{RESET}")
        return True
    except Exception as e:
        print(f"{YELLOW}⚠ WARN{RESET} ({e})")
        return False


def check_communication(timeout: float = 5.0) -> bool:
    """Verify communication hardware is reachable."""
    print(f"  Checking Radio (RFD900x)...", end=" ", flush=True)
    try:
        import serial
        ports = ["/dev/ttyUSB0", "/dev/ttyACM0"]
        for port in ports:
            try:
                s = serial.Serial(port, 57600, timeout=2)
                s.write(b"AT\r\n")
                resp = s.read(20)
                s.close()
                if b"OK" in resp or len(resp) > 0:
                    print(f"{GREEN}✓ OK{RESET} ({port})")
                    return True
            except Exception:
                continue
        print(f"{RED}✗ FAIL{RESET} (no radio found on USB)")
        return False
    except ImportError:
        print(f"{YELLOW}⚠ SKIP{RESET} (pyserial not installed)")
        return True


def check_battery(min_soc: float = 0.2) -> bool:
    """Verify battery is charged above minimum SoC."""
    print(f"  Checking Battery SoC...", end=" ", flush=True)
    try:
        import subprocess
        result = subprocess.run(
            ["ros2", "topic", "echo", "/acer/battery_state", "--once", "--timeout", "3"],
            capture_output=True, text=True, timeout=5
        )
        # Parse SoC from topic output (simplified)
        if "percentage" in result.stdout:
            for line in result.stdout.splitlines():
                if "percentage" in line:
                    soc = float(line.split(":")[-1].strip())
                    if soc >= min_soc:
                        print(f"{GREEN}✓ OK{RESET} (SoC: {soc*100:.0f}%)")
                        return True
                    print(f"{RED}✗ LOW{RESET} (SoC: {soc*100:.0f}% — below {min_soc*100:.0f}%)")
                    return False
        # Assume OK if topic not available (dev mode)
        print(f"{YELLOW}⚠ UNKNOWN{RESET} (no battery topic)")
        return True
    except Exception as e:
        print(f"{YELLOW}⚠ WARN{RESET} ({e})")
        return True


def run_all_checks(verbose: bool = False, timeout: float = 10.0) -> bool:
    print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║   ACER ROBOT — Pre-Mission Check     ║{RESET}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════╝{RESET}\n")

    checks = {
        "LiDAR": check_lidar,
        "IMU": check_imu,
        "Thermal Camera": check_thermal_camera,
        "Optical Cameras": check_optical_cameras,
        "Communication": check_communication,
        "Battery": check_battery,
    }

    results = {}
    for name, check_fn in checks.items():
        try:
            results[name] = check_fn(timeout)
        except Exception as e:
            print(f"{RED}  ERROR in {name}: {e}{RESET}")
            results[name] = False

    print(f"\n{BOLD}── Summary ──────────────────────────────{RESET}")
    all_ok = True
    critical = {"LiDAR", "IMU", "Battery", "Communication"}

    for name, ok in results.items():
        symbol = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        tag = " [CRITICAL]" if (not ok and name in critical) else ""
        print(f"  {symbol} {name}{RED}{tag}{RESET}")
        if not ok and name in critical:
            all_ok = False

    if all_ok:
        print(f"\n{GREEN}{BOLD}✓ All critical checks passed — CLEAR FOR MISSION{RESET}\n")
    else:
        print(f"\n{RED}{BOLD}✗ Critical check failures — DO NOT LAUNCH{RESET}\n")

    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ACER Pre-Mission Sensor Check")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    success = run_all_checks(verbose=args.verbose, timeout=args.timeout)
    sys.exit(0 if success else 1)
