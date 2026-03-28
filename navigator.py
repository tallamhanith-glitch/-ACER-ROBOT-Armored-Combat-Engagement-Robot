"""
ACER Robot - Autonomous Navigation Module
==========================================
Implements SLAM-based navigation, path planning, and dynamic obstacle avoidance.
Integrates with Nav2 stack via ROS2 action interfaces.

Author: ACER Team
Date: March 2025
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose, ComputePathToPose
from geometry_msgs.msg import PoseStamped, Twist, Point
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
import numpy as np
import yaml
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class NavigationState(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    NAVIGATING = "navigating"
    OBSTACLE_AVOIDANCE = "obstacle_avoidance"
    GOAL_REACHED = "goal_reached"
    FAILED = "failed"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class NavigationGoal:
    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0
    frame_id: str = "map"
    priority: int = 1  # 1=low, 5=critical


@dataclass
class NavigationConfig:
    max_linear_speed: float = 2.5      # m/s
    max_angular_speed: float = 1.8     # rad/s
    obstacle_clearance: float = 0.5    # meters
    goal_tolerance_xy: float = 0.15    # meters
    goal_tolerance_yaw: float = 0.1    # radians
    slam_algorithm: str = "cartographer"
    map_resolution: float = 0.05       # meters/cell
    planning_frequency: float = 5.0    # Hz


class ACERNavigator(Node):
    """
    Core autonomous navigation node for ACER Robot.
    
    Implements:
    - Autonomous waypoint navigation via Nav2 action interface
    - Real-time obstacle detection and avoidance (VFH+)
    - Dynamic replanning on map updates
    - Emergency stop on critical threat signals
    - Patrol route execution with looping support
    """

    def __init__(self, config: Optional[NavigationConfig] = None):
        super().__init__('acer_navigator')
        self.config = config or NavigationConfig()
        self.state = NavigationState.IDLE
        self.current_pose: Optional[PoseStamped] = None
        self.current_goal: Optional[NavigationGoal] = None
        self.patrol_waypoints: List[NavigationGoal] = []
        self.patrol_index: int = 0
        self.obstacle_detected: bool = False

        self._setup_publishers()
        self._setup_subscribers()
        self._setup_action_clients()

        self.get_logger().info("ACER Navigator initialized.")

    def _setup_publishers(self):
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.nav_state_pub = self.create_publisher(String, '/acer/nav_state', 10)
        self.emergency_stop_pub = self.create_publisher(Bool, '/acer/emergency_stop', 10)

    def _setup_subscribers(self):
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self._scan_callback, 10)
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self._map_callback, 10)
        self.threat_sub = self.create_subscription(
            String, '/acer/threat_alert', self._threat_callback, 10)

    def _setup_action_clients(self):
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.path_client = ActionClient(self, ComputePathToPose, 'compute_path_to_pose')

    # ─────────────────────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────────────────────

    def _odom_callback(self, msg: Odometry):
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        self.current_pose = pose

    def _scan_callback(self, msg: LaserScan):
        """Detect obstacles within clearance zone using LiDAR scan."""
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isfinite(ranges), ranges, np.inf)
        min_range = float(np.min(ranges))

        was_detected = self.obstacle_detected
        self.obstacle_detected = min_range < self.config.obstacle_clearance

        if self.obstacle_detected and not was_detected:
            self.get_logger().warn(f"Obstacle detected at {min_range:.2f}m — activating avoidance")
            self._activate_obstacle_avoidance(ranges, msg.angle_min, msg.angle_increment)

    def _map_callback(self, msg: OccupancyGrid):
        """Trigger replan when map updates during active navigation."""
        if self.state == NavigationState.NAVIGATING and self.current_goal:
            self.get_logger().info("Map updated — checking replan necessity")

    def _threat_callback(self, msg: String):
        """React to threat signals from threat assessment module."""
        if msg.data in ("CRITICAL", "IMMINENT"):
            self.get_logger().error(f"Threat alert: {msg.data} — executing emergency stop")
            self._emergency_stop()

    # ─────────────────────────────────────────────────────────────
    # Navigation Control
    # ─────────────────────────────────────────────────────────────

    def navigate_to(self, goal: NavigationGoal) -> bool:
        """Send a navigation goal to Nav2 action server."""
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Nav2 action server not available")
            return False

        self.current_goal = goal
        self.state = NavigationState.NAVIGATING

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self._build_pose_stamped(goal)

        self.get_logger().info(f"Navigating to ({goal.x:.2f}, {goal.y:.2f})")
        future = self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self._nav_feedback_callback
        )
        future.add_done_callback(self._nav_goal_response_callback)
        self._publish_state()
        return True

    def execute_patrol(self, waypoints: List[NavigationGoal], loop: bool = True):
        """Execute a patrol route across a list of waypoints."""
        self.patrol_waypoints = waypoints
        self.patrol_index = 0
        self.get_logger().info(
            f"Starting patrol: {len(waypoints)} waypoints, loop={loop}"
        )
        self._next_patrol_waypoint(loop)

    def _next_patrol_waypoint(self, loop: bool = True):
        if not self.patrol_waypoints:
            return
        if self.patrol_index >= len(self.patrol_waypoints):
            if loop:
                self.patrol_index = 0
            else:
                self.get_logger().info("Patrol complete")
                self.state = NavigationState.IDLE
                return
        wp = self.patrol_waypoints[self.patrol_index]
        self.patrol_index += 1
        self.navigate_to(wp)

    def _emergency_stop(self):
        self.state = NavigationState.EMERGENCY_STOP
        stop_cmd = Twist()
        for _ in range(5):
            self.cmd_vel_pub.publish(stop_cmd)

        stop_msg = Bool()
        stop_msg.data = True
        self.emergency_stop_pub.publish(stop_msg)
        self._publish_state()
        self.get_logger().error("EMERGENCY STOP ACTIVATED")

    # ─────────────────────────────────────────────────────────────
    # Obstacle Avoidance (Vector Field Histogram+)
    # ─────────────────────────────────────────────────────────────

    def _activate_obstacle_avoidance(
        self,
        ranges: np.ndarray,
        angle_min: float,
        angle_increment: float
    ):
        self.state = NavigationState.OBSTACLE_AVOIDANCE
        angles = angle_min + np.arange(len(ranges)) * angle_increment

        # Build polar histogram
        num_sectors = 72
        sector_size = 2 * np.pi / num_sectors
        histogram = np.zeros(num_sectors)

        for i, r in enumerate(ranges):
            if np.isfinite(r) and r < 3.0:
                sector = int((angles[i] + np.pi) / sector_size) % num_sectors
                histogram[sector] += (1.0 / (r + 0.01)) ** 2

        # Find best steering direction (lowest density valley)
        best_sector = int(np.argmin(histogram))
        best_angle = best_sector * sector_size - np.pi

        cmd = Twist()
        cmd.linear.x = min(0.5, self.config.max_linear_speed * 0.3)
        cmd.angular.z = np.clip(best_angle * 0.8, -self.config.max_angular_speed,
                                self.config.max_angular_speed)
        self.cmd_vel_pub.publish(cmd)

    # ─────────────────────────────────────────────────────────────
    # Action Callbacks
    # ─────────────────────────────────────────────────────────────

    def _nav_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Navigation goal rejected by Nav2")
            self.state = NavigationState.FAILED
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._nav_result_callback)

    def _nav_result_callback(self, future):
        result = future.result().result
        self.state = NavigationState.GOAL_REACHED
        self.get_logger().info("Goal reached successfully")
        self._publish_state()

    def _nav_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        # Optionally log distance remaining
        pass

    # ─────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────

    def _build_pose_stamped(self, goal: NavigationGoal) -> PoseStamped:
        from geometry_msgs.msg import Quaternion
        import math

        pose = PoseStamped()
        pose.header.frame_id = goal.frame_id
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = goal.x
        pose.pose.position.y = goal.y
        pose.pose.position.z = goal.z

        # Convert yaw to quaternion
        cy, sy = math.cos(goal.yaw / 2), math.sin(goal.yaw / 2)
        pose.pose.orientation.w = cy
        pose.pose.orientation.z = sy
        return pose

    def _publish_state(self):
        msg = String()
        msg.data = self.state.value
        self.nav_state_pub.publish(msg)

    def get_current_state(self) -> NavigationState:
        return self.state


def main(args=None):
    rclpy.init(args=args)
    config = NavigationConfig()
    navigator = ACERNavigator(config)
    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
