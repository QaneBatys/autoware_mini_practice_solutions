#!/usr/bin/env python3

import rospy
import shapely
import math
import numpy as np
import threading
from ros_numpy import msgify
from autoware_mini.msg import Path, DetectedObjectArray
from sensor_msgs.msg import PointCloud2

DTYPE = np.dtype([
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('vx', np.float32),
    ('vy', np.float32),
    ('vz', np.float32),
    ('distance_to_stop', np.float32),
    ('deceleration_limit', np.float32),
    ('category', np.int32)
])

class CollisionPointsManager:

    def __init__(self):

        # parameters
        self.safety_box_width = rospy.get_param("safety_box_width")
        self.stopped_speed_limit = rospy.get_param("stopped_speed_limit")
        self.braking_safety_distance_obstacle = rospy.get_param("~braking_safety_distance_obstacle")
        self.braking_safety_distance_goal = rospy.get_param("~braking_safety_distance_goal")
        # variables
        self.detected_objects = None
        self.goal_waypoint_position = None

        # Lock for thread safety
        self.lock = threading.Lock()

        # publishers
        self.local_path_collision_pub = rospy.Publisher('collision_points', PointCloud2, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('extracted_local_path', Path, self.path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/detection/final_objects', DetectedObjectArray, self.detected_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)
        rospy.Subscriber('/planning/global_path', Path, self.global_path_callback, queue_size=1, tcp_nodelay=True)

    def detected_objects_callback(self, msg):
        self.detected_objects = msg.objects
    
    def global_path_callback(self, msg):
        if msg.waypoints:
            goal_waypoint = msg.waypoints[-1]
            with self.lock:
                self.goal_waypoint_position = goal_waypoint.position

    def path_callback(self, msg):
        with self.lock:
            detected_objects = self.detected_objects
            goal_waypoint_position = self.goal_waypoint_position
        collision_points = np.array([], dtype=DTYPE)

        # 1. Check for Empty Path and Handle Immediately
        if not msg.waypoints:
            rospy.logwarn_throttle(3, "%s - Received an empty path! Publishing empty collision points.", rospy.get_name())
            
            # Publish an empty PointCloud2 message with the correct header
            empty_collision_points_msg = msgify(PointCloud2, collision_points)
            empty_collision_points_msg.header = msg.header
            self.local_path_collision_pub.publish(empty_collision_points_msg)
            return

        local_path_linestring = shapely.LineString([(waypoint.position.x, waypoint.position.y) for waypoint in msg.waypoints])
        local_path_buffer = local_path_linestring.buffer(self.safety_box_width / 2, cap_style="flat")
        shapely.prepare(local_path_buffer)
        

        if detected_objects is not None and len(detected_objects) > 0:
            for obj in detected_objects:
                object_polygon = shapely.polygons(np.array(obj.convex_hull).reshape(-1, 3))

                if local_path_buffer.intersects(object_polygon):
                    intersection_result = object_polygon.intersection(local_path_buffer)
                    intersection_points = shapely.get_coordinates(intersection_result)

                    object_speed = math.sqrt(obj.velocity.x**2 + obj.velocity.y**2 + obj.velocity.z**2)
                    for x, y in intersection_points:
                        collision_points = np.append(collision_points, np.array([(x, y, obj.centroid.z, obj.velocity.x, obj.velocity.y, obj.velocity.z,
                                                                                  self.braking_safety_distance_obstacle, np.inf, 3 if object_speed < self.stopped_speed_limit else 4)], dtype=DTYPE))

        if goal_waypoint_position is not None:
            goal_point = shapely.Point(goal_waypoint_position.x, goal_waypoint_position.y)
            
            if local_path_buffer.intersects(goal_point):
                
                goal_vx, goal_vy, goal_vz = 0.0, 0.0, 0.0
                goal_x, goal_y, goal_z = goal_point.x, goal_point.y, goal_waypoint_position.z
                
                collision_points = np.append(collision_points, np.array([
                    (
                        goal_x, goal_y, goal_z, 
                        goal_vx, goal_vy, goal_vz,
                        self.braking_safety_distance_goal, 
                        np.inf,                          
                        1                                 
                    )
                ], dtype=DTYPE))

        if len(collision_points) == 0:
            rospy.logdebug_throttle(3, "%s - No goal or obstacles found. Publishing empty collision points.", rospy.get_name())

        
        collision_points_msg = msgify(PointCloud2, collision_points)
        collision_points_msg.header = msg.header
        self.local_path_collision_pub.publish(collision_points_msg)
        #print(collision_points)



    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('collision_points_manager')
    node = CollisionPointsManager()
    node.run()