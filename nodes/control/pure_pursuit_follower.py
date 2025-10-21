#!/usr/bin/env python3

import rospy

from shapely.geometry import LineString, Point
from shapely import prepare, distance
from autoware_mini.msg import Path, VehicleCmd
from geometry_msgs.msg import PoseStamped

class PurePursuitFollower:
    def __init__(self):

        # Parameters
        self.path_linestring = None
        # Publishers
        self.vel_pub = rospy.Publisher('/control/vehicle_cmd', VehicleCmd, queue_size=10)


        # Subscribers
        rospy.Subscriber('path', Path, self.path_callback, queue_size=1)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1)

    def path_callback(self, msg):
        # TODO
        self.path_linestring = LineString([(w.position.x, w.position.y) for w in msg.waypoints])
        prepare(self.path_linestring)

    def current_pose_callback(self, msg):
        # TODO
        # prepare path - creates spatial tree, making the spatial queries more efficient
        if self.path_linestring is not None:
            current_pose = Point([msg.pose.position.x, msg.pose.position.y])
            d_ego_from_path_start = self.path_linestring.project(current_pose)
            print(d_ego_from_path_start)
        vehicle_cmd = VehicleCmd()
        vehicle_cmd.ctrl_cmd.steering_angle = 0.2
        vehicle_cmd.ctrl_cmd.linear_velocity = 10.0
        vehicle_cmd.header.stamp = msg.header.stamp
        vehicle_cmd.header.frame_id = 'base_link'
        self.vel_pub.publish(vehicle_cmd)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('pure_pursuit_follower')
    node = PurePursuitFollower()
    node.run()