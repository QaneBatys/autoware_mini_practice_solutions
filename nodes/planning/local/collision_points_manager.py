#!/usr/bin/env python3

import rospy
import shapely
import math
import numpy as np
import threading
from ros_numpy import msgify
from autoware_mini.msg import Path, DetectedObjectArray, TrafficLightResultArray
from sensor_msgs.msg import PointCloud2
from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from shapely.geometry import LineString

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
        self.braking_safety_distance_stopline = rospy.get_param("~braking_safety_distance_stopline")
        
        # Load Lanelet2 map for traffic light stop lines
        coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
        use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")
        lanelet2_map_path = rospy.get_param("~lanelet2_map_path")

        # Load the map using Lanelet2
        if coordinate_transformer == "utm":
            projector = UtmProjector(Origin(utm_origin_lat, utm_origin_lon), use_custom_origin, False)
        else:
            raise RuntimeError('Only "utm" is supported for lanelet2 map loading')
        lanelet2_map = load(lanelet2_map_path, projector)

        # Extract all stop lines from the lanelet2 map
        self.stoplines = self.get_stoplines(lanelet2_map)
        
        # variables
        self.detected_objects = None
        self.goal_waypoint_position = None
        self.traffic_light_status = {}
        self.local_path = None

        # Lock for thread safety
        self.lock = threading.Lock()

        # publishers
        self.local_path_collision_pub = rospy.Publisher('collision_points', PointCloud2, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('extracted_local_path', Path, self.path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/detection/final_objects', DetectedObjectArray, self.detected_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)
        rospy.Subscriber('/planning/global_path', Path, self.global_path_callback, queue_size=1, tcp_nodelay=True)
        # NEW: Subscribe to traffic light status
        rospy.Subscriber('/detection/traffic_light_status', TrafficLightResultArray, self.traffic_light_status_callback, queue_size=1, tcp_nodelay=True)

    def get_stoplines(self, lanelet2_map):
        """Extract all stop lines from lanelet2 map"""
        stoplines = {}
        for line in lanelet2_map.lineStringLayer:
            if line.attributes:
                # FIXED: Use proper Lanelet2 attribute access
                if "type" in line.attributes and line.attributes["type"] == "stop_line":
                    stoplines[line.id] = LineString([(p.x, p.y) for p in line])
        return stoplines

    def detected_objects_callback(self, msg):
        self.detected_objects = msg.objects
    
    def global_path_callback(self, msg):
        if msg.waypoints:
            goal_waypoint = msg.waypoints[-1]
            with self.lock:
                self.goal_waypoint_position = goal_waypoint.position

    def traffic_light_status_callback(self, msg):
        """Store the latest traffic light status"""
        with self.lock:
            self.traffic_light_status.clear()
            for result in msg.results:
                # recognition_result
                self.traffic_light_status[result.stopline_id] = result.recognition_result
            rospy.logdebug("Updated traffic light status: %s", self.traffic_light_status)

    def path_callback(self, msg):
        with self.lock:
            detected_objects = self.detected_objects
            goal_waypoint_position = self.goal_waypoint_position
            traffic_light_status = self.traffic_light_status.copy()
             # Store local path for traffic light processing
            self.local_path = msg  
        
        collision_points = np.array([], dtype=DTYPE)

        if not msg.waypoints:
            rospy.logwarn_throttle(3, "%s - Received an empty path! Publishing empty collision points.", rospy.get_name())
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

        traffic_light_points = self.create_traffic_light_collision_points(local_path_buffer, traffic_light_status)
        collision_points = np.append(collision_points, traffic_light_points)

        if len(collision_points) == 0:
            rospy.logdebug_throttle(3, "%s - No goal, obstacles, or traffic lights found. Publishing empty collision points.", rospy.get_name())
            
        collision_points_msg = msgify(PointCloud2, collision_points)
        collision_points_msg.header = msg.header
        self.local_path_collision_pub.publish(collision_points_msg)

    def create_traffic_light_collision_points(self, local_path_buffer, traffic_light_status):
        """Create collision points for red/yellow traffic lights"""
        collision_points = np.array([], dtype=DTYPE)
        
        for stopline_id, stopline_geometry in self.stoplines.items():
            # Check if this stopline has a red or yellow traffic light
            if stopline_id in traffic_light_status:
                traffic_light_state = traffic_light_status[stopline_id]
                
                # Only create collision points for RED  or YELLOW (0)
                if traffic_light_state == 0:  
                    # Check if buffered path intersects with stop line
                    if local_path_buffer.intersects(stopline_geometry):
                        # Find the intersection point
                        intersection = local_path_buffer.intersection(stopline_geometry)
                        
                        if intersection.is_empty:
                            continue
                            
                        # Get the point on the stop line closest to the path
                        if intersection.geom_type == 'Point':
                            stop_point = intersection
                        elif intersection.geom_type in ['LineString', 'MultiLineString']:
                            # Use the point on stop line closest to path start
                            if self.local_path and self.local_path.waypoints:
                                path_start = shapely.Point(self.local_path.waypoints[0].position.x, self.local_path.waypoints[0].position.y)
                                stop_point = stopline_geometry.interpolate(stopline_geometry.project(path_start))
                            else:
                                continue
                        else:
                            stop_point = intersection.centroid
                        
                        # Create collision point for traffic light
                        collision_point = np.array([(
                            stop_point.x, 
                            stop_point.y, 
                            0.0,  
                            0.0, 0.0, 0.0,  
                            self.braking_safety_distance_stopline,  
                            np.inf,  
                            2  # Category 2 for traffic lights
                        )], dtype=DTYPE)
                        
                        collision_points = np.append(collision_points, collision_point)
                        #rospy.loginfo("Created traffic light collision point for stopline %d at (%.2f, %.2f)", stopline_id, stop_point.x, stop_point.y)
        
        return collision_points

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('collision_points_manager')
    node = CollisionPointsManager()
    node.run()