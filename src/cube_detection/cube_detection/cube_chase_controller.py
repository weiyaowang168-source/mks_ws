#!/usr/bin/env python3
import math
from typing import Optional

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseArray
from std_msgs.msg import Float64


class CubeChaseWheelSpeedController(Node):
    """
    Visual-servo controller (PoseArray -> wheel speed commands)

    - Sub:  /cube_poses (PoseArray)   # positions are in camera frame (X,Y,Z) meters
    - Pub:  front_left/commands/motor/speed   (Float64)
            front_right/commands/motor/speed  (Float64)
            rear_left/commands/motor/speed    (Float64)
            rear_right/commands/motor/speed   (Float64)

    Logic:
    - Choose nearest cube (min Z)
    - Angular control:  ang_cmd = -k_w * X
    - Linear control:   lin_cmd = k_v * (Z - target_z)
    - Mixer: left = lin_cmd - ang_cmd, right = lin_cmd + ang_cmd
    """

    def __init__(self):
        super().__init__("cube_chase_wheel_speed_controller")

        # ===== Params (tune on-site) =====
        self.declare_parameter("pose_topic", "/cube_poses")
        self.declare_parameter("enable", True)

        # target distance to stop in front of cube
        self.declare_parameter("target_z", 0.60)          # meters
        self.declare_parameter("z_stop_deadband", 0.08)   # meters

        # "control gains" (still computed from X,Z)
        self.declare_parameter("k_v", 0.8)   # linear gain (based on z_err)
        self.declare_parameter("k_w", 2.0)   # angular gain (based on X)

        # IMPORTANT: now we output "motor speed units", so we need scaling
        # If your driver expects eRPM / ticks/s, tune these two scalers.
        self.declare_parameter("lin_to_speed", 8000.0)    # speed_units per (m/s-like lin cmd)
        self.declare_parameter("ang_to_speed", 12000.0)   # speed_units per (rad/s-like ang cmd)

        # limits in motor-speed units (NOT m/s)
        self.declare_parameter("speed_max", 12000.0)

        # safety
        self.declare_parameter("timeout_sec", 0.5)  # if no detection, stop
        self.declare_parameter("publish_rate_hz", 20.0)

        # wheel command topics
        self.declare_parameter("fl_topic", "front_left/commands/motor/speed")
        self.declare_parameter("fr_topic", "front_right/commands/motor/speed")
        self.declare_parameter("rl_topic", "rear_left/commands/motor/speed")
        self.declare_parameter("rr_topic", "rear_right/commands/motor/speed")

        # ===== Read params =====
        self.pose_topic = self.get_parameter("pose_topic").value
        self.enable = bool(self.get_parameter("enable").value)

        self.target_z = float(self.get_parameter("target_z").value)
        self.z_stop_deadband = float(self.get_parameter("z_stop_deadband").value)

        self.k_v = float(self.get_parameter("k_v").value)
        self.k_w = float(self.get_parameter("k_w").value)

        self.lin_to_speed = float(self.get_parameter("lin_to_speed").value)
        self.ang_to_speed = float(self.get_parameter("ang_to_speed").value)

        self.speed_max = float(self.get_parameter("speed_max").value)

        self.timeout_sec = float(self.get_parameter("timeout_sec").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        fl_topic = self.get_parameter("fl_topic").value
        fr_topic = self.get_parameter("fr_topic").value
        rl_topic = self.get_parameter("rl_topic").value
        rr_topic = self.get_parameter("rr_topic").value

        # ===== ROS I/O =====
        self.sub = self.create_subscription(PoseArray, self.pose_topic, self.on_poses, 10)

        self.fl_pub = self.create_publisher(Float64, fl_topic, 10)
        self.fr_pub = self.create_publisher(Float64, fr_topic, 10)
        self.rl_pub = self.create_publisher(Float64, rl_topic, 10)
        self.rr_pub = self.create_publisher(Float64, rr_topic, 10)

        self.last_msg_time: Optional[float] = None
        self.latest_poses: Optional[PoseArray] = None

        period = 1.0 / max(1e-3, self.publish_rate_hz)
        self.timer = self.create_timer(period, self.control_loop)

        self.get_logger().info(
            f"CubeChaseWheelSpeedController started. Sub={self.pose_topic} "
            f"Pub=[{fl_topic}, {fr_topic}, {rl_topic}, {rr_topic}]"
        )

    def on_poses(self, msg: PoseArray):
        self.latest_poses = msg
        self.last_msg_time = self.get_clock().now().nanoseconds * 1e-9

    def control_loop(self):
        now = self.get_clock().now().nanoseconds * 1e-9

        # if disabled -> stop
        if not self.enable:
            self.publish_stop()
            return

        # timeout -> stop
        if self.last_msg_time is None or (now - self.last_msg_time) > self.timeout_sec:
            self.publish_stop()
            return

        if self.latest_poses is None or len(self.latest_poses.poses) == 0:
            self.publish_stop()
            return

        # choose nearest cube (min Z)
        best = min(self.latest_poses.poses, key=lambda p: p.position.z)

        X = float(best.position.x)  # left/right in camera frame
        Z = float(best.position.z)  # forward distance

        # your original v,w logic (conceptually)
        # angular: turn to reduce X
        w = -self.k_w * X

        # linear: approach target_z
        z_err = Z - self.target_z
        if abs(z_err) < self.z_stop_deadband:
            v = 0.0
        else:
            v = self.k_v * z_err

        # ===== Convert v,w to motor-speed units =====
        lin_cmd = v * self.lin_to_speed
        ang_cmd = w * self.ang_to_speed

        # ===== Mixer (same as your auto_velocity_control) =====
        left_speed = lin_cmd - ang_cmd
        right_speed = lin_cmd + ang_cmd

        # clamp
        left_speed = max(-self.speed_max, min(self.speed_max, left_speed))
        right_speed = max(-self.speed_max, min(self.speed_max, right_speed))

        # publish 4 wheels
        self.publish_wheels(left_speed, right_speed)

    def publish_wheels(self, left_speed: float, right_speed: float):
        self.fl_pub.publish(Float64(data=float(left_speed)))
        self.rl_pub.publish(Float64(data=float(left_speed)))
        self.fr_pub.publish(Float64(data=float(right_speed)))
        self.rr_pub.publish(Float64(data=float(right_speed)))

    def publish_stop(self):
        self.publish_wheels(0.0, 0.0)


def main():
    rclpy.init()
    node = CubeChaseWheelSpeedController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
