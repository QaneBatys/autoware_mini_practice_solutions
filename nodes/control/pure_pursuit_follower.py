#!/usr/bin/env python3

import rospy
from shapely.geometry import LineString, Point
from shapely import prepare, distance
from autoware_mini.msg import Path, VehicleCmd
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion
from math import atan2, sin, cos, atan
from scipy.interpolate import interp1d
import numpy as np

class PurePursuitFollower:
    def __init__(self):
        # Parameters
        self.path_linestring = None
        self.distance_to_velocity_interpolator = None
        self.is_path_valid = False
        # Publishers
        self.vel_pub = rospy.Publisher('/control/vehicle_cmd', VehicleCmd, queue_size=10)
        self.lookahead_distance = rospy.get_param('~lookahead_distance', 5.0)
        self.wheel_base = rospy.get_param('/vehicle/wheel_base')
        # Subscribers
        rospy.Subscriber('path', Path, self.path_callback, queue_size=1) # Note: Changed to /global_path
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1)

    def path_callback(self, msg):  
        if len(msg.waypoints) == 0:
            self.path_linestring = None
            self.distance_to_velocity_interpolator = None
            self.is_path_valid = False
            return

        self.is_path_valid = True
        self.path_linestring = LineString([(w.position.x, w.position.y) for w in msg.waypoints])
        prepare(self.path_linestring)
        waypoints_xy = np.array([(w.position.x, w.position.y) for w in msg.waypoints])
        distances = np.cumsum(np.sqrt(np.sum(np.diff(waypoints_xy, axis=0)**2, axis=1)))
        distances = np.insert(distances, 0, 0)
        velocities = np.array([w.speed for w in msg.waypoints])

        # Create the interpolator
        self.distance_to_velocity_interpolator = interp1d( distances, velocities, kind='linear', bounds_error=False, fill_value=0.0)
        

    def current_pose_callback(self, msg):
        if not self.is_path_valid or self.path_linestring is None:
            vehicle_cmd = VehicleCmd()
            vehicle_cmd.ctrl_cmd.steering_angle = 0.0
            vehicle_cmd.ctrl_cmd.linear_velocity = 0.0 # Stop signal
            vehicle_cmd.header.stamp = msg.header.stamp
            vehicle_cmd.header.frame_id = 'base_link'
            self.vel_pub.publish(vehicle_cmd)
            return
        
        _, _, current_heading_angle = euler_from_quaternion((msg.pose.orientation.x, msg.pose.orientation.y , msg.pose.orientation.z , msg.pose.orientation.w))

        steering_angle = 0.0
        target_velocity = 0.0
        
        current_pose = Point([msg.pose.position.x, msg.pose.position.y])
        d_ego_from_path_start = self.path_linestring.project(current_pose)
        
        # Correctly calculate target distance along the arc (path length)
        path_length = self.path_linestring.length
        target_distance_arc = d_ego_from_path_start + self.lookahead_distance
        
        # Clip distance to path end to prevent 'NoneType' error
        if target_distance_arc > path_length:
            target_distance_arc = path_length

        lookahead_point = self.path_linestring.interpolate(target_distance_arc)

        if lookahead_point is None:
             vehicle_cmd = VehicleCmd()
             vehicle_cmd.ctrl_cmd.steering_angle = 0.0
             vehicle_cmd.ctrl_cmd.linear_velocity = 0.0
             vehicle_cmd.header.stamp = msg.header.stamp
             vehicle_cmd.header.frame_id = 'base_link'
             self.vel_pub.publish(vehicle_cmd)
             return

        lookahead_x = lookahead_point.x 
        lookahead_y = lookahead_point.y
        
        target_heading_angle = atan2(
            lookahead_y - msg.pose.position.y,
            lookahead_x - msg.pose.position.x,
        )

        Ld = distance(current_pose, lookahead_point)

        alpha = target_heading_angle - current_heading_angle
        if Ld > 0.0:
             steering_angle = atan((2.0 * self.wheel_base * sin(alpha)) / Ld)
        else:
            steering_angle = 0.0
        
        
        if self.distance_to_velocity_interpolator is not None:
            target_velocity = self.distance_to_velocity_interpolator(d_ego_from_path_start)
        
        # --- Publish Command ---
        vehicle_cmd = VehicleCmd()
        vehicle_cmd.ctrl_cmd.steering_angle = steering_angle
        vehicle_cmd.ctrl_cmd.linear_velocity = target_velocity
        vehicle_cmd.header.stamp = msg.header.stamp
        vehicle_cmd.header.frame_id = 'base_link'
        self.vel_pub.publish(vehicle_cmd)


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('pure_pursuit_follower')
    node = PurePursuitFollower()
    node.run()

# when lookup distance is 20, it does not follow path. It cuts off many corners.
