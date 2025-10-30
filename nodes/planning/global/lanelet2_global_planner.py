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
            path = Path()        
            path.header.frame_id = self.output_frame 
            path.header.stamp = rospy.Time.now()
            path.waypoints = waypoints 
            
            self.waypoints_pub.publish(path)
            
            if not waypoints:
                # Empty path published to clear the path in the follower
                pass 
            else:
                rospy.loginfo("%s: Published global path with %d waypoints to /global_path.", 
                            rospy.get_name(), len(waypoints))


    def goal_callback(self, msg: PoseStamped):
        user_goal_point_2d = BasicPoint2d(msg.pose.position.x, msg.pose.position.y) 
        self.goal_point = user_goal_point_2d 
        # get start and end lanelets
        start_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.current_location, 1)[0][1]
        # We still use findNearest for routing, but we'll manually set the path end later.
        goal_lanelet = findNearest(self.lanelet2_map.laneletLayer, user_goal_point_2d, 1)[0][1]
        # find routing graph
        route = self.graph.getRoute(start_lanelet, goal_lanelet, 0, True)

        if route:
            path = route.shortestPath()
            path_no_lane_change = path.getRemainingLane(start_lanelet)
            waypoints = self.convert_lanelet_sequence_to_waypoints(path_no_lane_change)
            if waypoints:
                # 1. Get speed from the last waypoint (or use 0.0 for a definite stop)
                final_speed = waypoints[-1].speed
                # 2. Create the new final waypoint
                final_waypoint = Waypoint()
                final_waypoint.position.x = user_goal_point_2d.x
                final_waypoint.position.y = user_goal_point_2d.y
                final_waypoint.position.z = msg.pose.position.z if hasattr(msg.pose.position, 'z') else 0.0
                final_waypoint.speed = final_speed
                waypoints.append(final_waypoint)            
            self.publish_waypoints(waypoints)
        else:
            rospy.logwarn("%s: Path found, but no immediate continuous lane segment remaining.", rospy.get_name())


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
                rospy.loginfo("%s: Goal reached (distance: %.2fm). Path has been cleared.", 
                              rospy.get_name(), goal_distance)

    def run(self):
        rospy.spin()

        

if __name__ == '__main__':
    rospy.init_node('lanelet2_global_planner', anonymous=False)
    node = Lanelet2GlobalPlanner()
    node.run()
