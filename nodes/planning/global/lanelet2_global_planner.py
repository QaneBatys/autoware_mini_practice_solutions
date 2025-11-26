#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
import lanelet2
from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from lanelet2.core import BasicPoint2d
from lanelet2.geometry import findNearest
from lanelet2.routing import RoutingGraph, Route
from autoware_mini.msg import Path, Waypoint
import shapely.geometry as sg



class Lanelet2GlobalPlanner:
    def __init__(self):
        # Internal variables
        self.current_location = None
        self.goal_point = None
        self.lanelet2_map = None 
        self.graph = None 
        self.speed_limit_kph = rospy.get_param("~speed_limit", 40.0)
        self.speed_limit_mps = self.speed_limit_kph / 3.6
        self.output_frame = rospy.get_param("/output_frame", "map")
        self.distance_to_goal_limit = rospy.get_param("/lanelet2_global_planner/distance_to_goal_limit", 4.0) 
        self.lanelet2_map = self.load_lanelet2_map()
        if self.lanelet2_map:
            traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                          lanelet2.traffic_rules.Participants.VehicleTaxi)
            self.graph = lanelet2.routing.RoutingGraph(self.lanelet2_map, traffic_rules)
        # Subscribers
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_callback)
        rospy.Subscriber("/localization/current_pose", PoseStamped, self.current_position_callback) 
        # Publishers (add global path publisher later)
        self.waypoints_pub = rospy.Publisher("global_path", Path, queue_size=1, latch=True)
    
    def load_lanelet2_map(self):
        coordinate_transformer = rospy.get_param("/coordinate_transformer", "utm")
        use_custom_origin = rospy.get_param("/use_custom_origin", True)
        utm_origin_lat = rospy.get_param("/utm_origin_lat", 58.385345)
        utm_origin_lon = rospy.get_param("/utm_origin_lon", 26.726272)
        lanelet2_map_path = rospy.get_param("~lanelet2_map_path")

        if coordinate_transformer == "utm":
            projector = UtmProjector(Origin(utm_origin_lat, utm_origin_lon), use_custom_origin, False)
        else:
            rospy.logerr('%s: Unknown coordinate_transformer ("utm" should be used): %s', 
                         rospy.get_name(), coordinate_transformer)
            return None

        return load(lanelet2_map_path, projector)

    def convert_lanelet_sequence_to_waypoints(self, lanelet_sequence) :
        waypoints: list[Waypoint] = []
        last_point_added = None 
        
        for lanelet in lanelet_sequence:
            speed_mps = self.speed_limit_mps 
            
            if 'speed_ref' in lanelet.attributes:
                # speed_ref is in km/h, convert to m/s
                ref_speed_kph = float(lanelet.attributes['speed_ref'])
                ref_speed_mps = ref_speed_kph / 3.6
                speed_mps = min(ref_speed_mps, self.speed_limit_mps)

            for i, point in enumerate(lanelet.centerline):
                # Logic to skip overlapping points
                if len(waypoints) > 0 and i == 0 and point == last_point_added:
                    continue
                
                waypoint = Waypoint()                
                waypoint.position.x = point.x
                waypoint.position.y = point.y
                waypoint.position.z = point.z if hasattr(point, 'z') else 0.0
                waypoint.speed = speed_mps
                waypoints.append(waypoint)
                last_point_added = point
        return waypoints

    def publish_waypoints(self, waypoints):
        if waypoints is None:
            waypoints = []

        path = Path()        
        path.header.frame_id = self.output_frame 
        path.header.stamp = rospy.Time.now()
        path.waypoints = waypoints 
            
        self.waypoints_pub.publish(path)
            
        if not waypoints:
            #rospy.loginfo("%s: Published empty global path to signal stop/clear.", rospy.get_name())
        else:
            rospy.loginfo("%s: Published global path with %d waypoints to /global_path.", 
                          rospy.get_name(), len(waypoints))
        
        if waypoints:
            rospy.loginfo("%s: Published global path with %d waypoints. Final waypoint at (%.2f, %.2f)", 
                        rospy.get_name(), len(waypoints), 
                        waypoints[-1].position.x, waypoints[-1].position.y)
        else:
            rospy.loginfo("%s: Published EMPTY global path", rospy.get_name())


    def goal_callback(self, msg: PoseStamped):
        if self.current_location is None:
            rospy.logwarn("%s: Ignoring goal — current location unknown.", rospy.get_name())
            return

        # Convert user click to 2D
        user_goal_point_2d = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)
        self.goal_point = user_goal_point_2d

        try:
            start_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.current_location, 1)[0][1]
            goal_lanelet = findNearest(self.lanelet2_map.laneletLayer, user_goal_point_2d, 1)[0][1]
        except Exception as e:
            rospy.logerr("%s: Failed to find nearest lanelets: %s", rospy.get_name(), e)
            return

        route = self.graph.getRoute(start_lanelet, goal_lanelet, 0, True)
        if not route:
            rospy.logwarn("%s: No route found between start and goal.", rospy.get_name())
            return

        path = route.shortestPath()
        if path is None:
            rospy.logwarn("%s: shortestPath() returned None.", rospy.get_name())
            return

        path_no_lane_change = path.getRemainingLane(start_lanelet)
        if path_no_lane_change is None:
            rospy.logwarn("%s: No continuous lane sequence from start.", rospy.get_name())
            return

        waypoints = self.convert_lanelet_sequence_to_waypoints(path_no_lane_change)
        if not waypoints:
            rospy.logwarn("%s: No waypoints generated.", rospy.get_name())
            return

        
        # Create LineString from current waypoints
        path_line = sg.LineString([(wp.position.x, wp.position.y) for wp in waypoints])
        user_goal_point = sg.Point(user_goal_point_2d.x, user_goal_point_2d.y)
        # Find closest point on path to user goal
        closest_point_on_path = path_line.interpolate(path_line.project(user_goal_point))
        # Find the waypoint index to truncate
        truncate_index = len(waypoints) - 1  # Default to full path
        
        for i, wp in enumerate(waypoints):
            wp_point = sg.Point(wp.position.x, wp.position.y)
            if wp_point.distance(closest_point_on_path) < 0.1:  
                truncate_index = i
                break
        
        # If no waypoint was found, find the segment to interpolate
        if truncate_index == len(waypoints) - 1:
            # Find the line segment that contains the closest point
            min_dist = float('inf')
            for i in range(len(waypoints) - 1):
                segment = sg.LineString([(waypoints[i].position.x, waypoints[i].position.y),
                                    (waypoints[i+1].position.x, waypoints[i+1].position.y)])
                dist = segment.distance(closest_point_on_path)
                if dist < min_dist:
                    min_dist = dist
                    if dist < 0.1: 
                        # Create interpolated waypoint at exact goal position
                        goal_waypoint = Waypoint()
                        goal_waypoint.position.x = closest_point_on_path.x
                        goal_waypoint.position.y = closest_point_on_path.y
                        goal_waypoint.position.z = waypoints[i].position.z  # Use z from previous point
                        goal_waypoint.speed = waypoints[i].speed  # Use speed from previous point
                        
                        # Truncate path and add the goal waypoint
                        waypoints = waypoints[:i+1] + [goal_waypoint]
                        truncate_index = i + 1
                        break
        
        # Truncate the path at the goal point
        waypoints = waypoints[:truncate_index + 1]
        
        if waypoints:
            final_wp = waypoints[-1]
            final_wp.position.x = closest_point_on_path.x
            final_wp.position.y = closest_point_on_path.y
        
        self.goal_point = BasicPoint2d(closest_point_on_path.x, closest_point_on_path.y)
        self.publish_waypoints(waypoints)

    def current_position_callback(self, msg: PoseStamped):
        self.current_location = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)
        if self.goal_point is not None and self.current_location is not None:
            # Calculate 2D Euclidean distance 
            delta_x = self.current_location.x - self.goal_point.x
            delta_y = self.current_location.y - self.goal_point.y
            goal_distance = (delta_x**2 + delta_y**2)**0.5
            
            if goal_distance < self.distance_to_goal_limit:
                # Goal reached case
                self.publish_waypoints([])
                self.goal_point = None 
                #rospy.loginfo("%s: Goal reached (distance: %.2fm). Path has been cleared.", rospy.get_name(), goal_distance)

    def run(self):
        rospy.spin()

        

if __name__ == '__main__':
    rospy.init_node('lanelet2_global_planner')
    node = Lanelet2GlobalPlanner()
    node.run()
