#!/usr/bin/env python3

import rospy
import math
import message_filters
import traceback
import shapely
import numpy as np
import threading
from numpy.lib.recfunctions import structured_to_unstructured
from ros_numpy import numpify
from autoware_mini.msg import Path, Log
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
from autoware_mini.geometry import project_vector_to_heading, get_distance_between_two_points_2d


class SpeedPlanner:

    def __init__(self):

        # parameters
        self.default_deceleration = rospy.get_param("default_deceleration")
        self.braking_reaction_time = rospy.get_param("braking_reaction_time")
        synchronization_queue_size = rospy.get_param("~synchronization_queue_size")
        synchronization_slop = rospy.get_param("~synchronization_slop")
        self.distance_to_car_front = rospy.get_param("distance_to_car_front")

        # variables
        self.collision_points = None
        self.current_position = None
        self.current_speed = None

        # Lock for thread safety
        self.lock = threading.Lock()

        # publishers
        self.local_path_pub = rospy.Publisher('local_path', Path, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/localization/current_velocity', TwistStamped, self.current_velocity_callback, queue_size=1, tcp_nodelay=True)

        collision_points_sub = message_filters.Subscriber('collision_points', PointCloud2, tcp_nodelay=True)
        local_path_sub = message_filters.Subscriber('extracted_local_path', Path, tcp_nodelay=True)

        ts = message_filters.ApproximateTimeSynchronizer([collision_points_sub, local_path_sub], queue_size=synchronization_queue_size, slop=synchronization_slop)

        ts.registerCallback(self.collision_points_and_path_callback)

    def current_velocity_callback(self, msg):
        self.current_speed = msg.twist.linear.x

    def current_pose_callback(self, msg):
        self.current_position = shapely.Point(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)

    def collision_points_and_path_callback(self, collision_points_msg, local_path_msg):
        try:
            with self.lock:
                collision_points = numpify(collision_points_msg) if len(collision_points_msg.data) > 0 else np.array([])
                current_position = self.current_position
                current_speed = self.current_speed

            if current_position is None or current_speed is None:
                rospy.logwarn_throttle(3, "%s - Current pose or speed not received! Returning.", rospy.get_name())
                return
            
            if not local_path_msg.waypoints:
                rospy.logwarn_throttle(3, "%s - Received an empty path! Returning.", rospy.get_name())
                return

            if len(collision_points) == 0:
                rospy.logdebug_throttle(3, "%s - No collision points found, publishing original path.", rospy.get_name())
                self.local_path_pub.publish(local_path_msg)
                return         

            local_path_linestring = shapely.LineString(
                [(wp.position.x, wp.position.y) for wp in local_path_msg.waypoints]
            )

            collision_points_shapely = shapely.points(structured_to_unstructured(collision_points[['x', 'y', 'z']]))
            collision_point_distances = np.array([local_path_linestring.project(collision_point_shapely) for collision_point_shapely in collision_points_shapely])
            
            projected_object_speeds = [] 
            

            for dist, point in zip(collision_point_distances, collision_points):
                heading_angle = self.get_heading_at_distance(local_path_linestring, dist)
                    
                vx, vy = point['vx'], point['vy']
                object_vector = Vector3(x=vx, y=vy, z=0.0)
                    
                projected_speed = self.project_vector_to_heading(heading_angle, object_vector)
                projected_object_speeds.append(projected_speed)
                    
                actual_speed = math.sqrt(vx**2 + vy**2)
                rospy.loginfo( "object velocity: %.10f transformed velocity: %.10f", actual_speed, projected_speed)
                
            projected_object_speeds = np.array(projected_object_speeds)
            distance_to_stop_values = collision_points['distance_to_stop']
            
            safety_buffer = self.braking_reaction_time * np.abs(projected_object_speeds)
            
            target_distances = collision_point_distances - safety_buffer
            
            deceleration_distances = target_distances - self.distance_to_car_front - distance_to_stop_values
            
            target_relative_velocity_squared = np.maximum(
                2.0 * self.default_deceleration * deceleration_distances, 
                0.0 
            )
            
            clamped_projected_speeds = np.maximum(0.0, projected_object_speeds)
            
            target_velocities = np.sqrt(target_relative_velocity_squared) - clamped_projected_speeds
            


            min_target_velocity = np.min(target_velocities)

            min_vel_idx = np.argmin(target_velocities)

            
            critical_obstacle_distance = collision_point_distances[min_vel_idx]
            critical_object_projected_speed = projected_object_speeds[min_vel_idx]
            critical_distance_to_stop = distance_to_stop_values[min_vel_idx]
            
            required_stopping_point_distance = critical_obstacle_distance - (self.distance_to_car_front + critical_distance_to_stop)
            
            collision_point_category = collision_points['category'][min_vel_idx] if 'category' in collision_points.dtype.names else 0 


            target_velocity = min_target_velocity
            for i, wp in enumerate(local_path_msg.waypoints):
                wp.speed = min(target_velocity, wp.speed) 
            
            
            path = Path()
            path.header = local_path_msg.header
            path.waypoints = local_path_msg.waypoints 
            path.closest_object_distance = critical_obstacle_distance
            path.closest_object_velocity = critical_object_projected_speed 
            path.is_blocked = True
            path.stopping_point_distance = required_stopping_point_distance
            path.collision_point_category = collision_point_category 
            
            self.local_path_pub.publish(path)

        except Exception as e:
            rospy.logerr_throttle(10, "%s - Exception in callback: %s", rospy.get_name(), traceback.format_exc())



    def get_heading_at_distance(self, linestring, distance):
        """
        Get heading of the path at a given distance
        :param distance: distance along the path
        :param linestring: shapely linestring
        :return: heading angle in radians
        """

        point_after_object = linestring.interpolate(distance + 0.1)
        point_before_object = linestring.interpolate(max(0, distance - 0.1))

        return math.atan2(point_after_object.y - point_before_object.y, point_after_object.x - point_before_object.x)


    def project_vector_to_heading(self, heading_angle, vector):
        """
        Project vector to heading
        :param heading_angle: heading angle in radians
        :param vector: vector
        :return: projected vector
        """

        return vector.x * math.cos(heading_angle) + vector.y * math.sin(heading_angle)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('speed_planner')
    node = SpeedPlanner()
    node.run()