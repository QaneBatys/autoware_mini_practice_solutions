#!/usr/bin/env python3

import rospy
import numpy as np

from shapely import MultiPoint
from tf2_ros import TransformListener, Buffer, TransformException
from numpy.lib.recfunctions import structured_to_unstructured
from ros_numpy import numpify, msgify

from sensor_msgs.msg import PointCloud2
from autoware_mini.msg import DetectedObjectArray, DetectedObject
from std_msgs.msg import ColorRGBA, Header
from geometry_msgs.msg import Point32


BLUE80P = ColorRGBA(0.0, 0.0, 1.0, 0.8)

class ClusterDetector:
    def __init__(self):
        self.min_cluster_size = rospy.get_param('~min_cluster_size')
        self.output_frame = rospy.get_param('/detection/output_frame')
        self.transform_timeout = rospy.get_param('~transform_timeout')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        self.objects_pub = rospy.Publisher('detected_objects', DetectedObjectArray, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('points_clustered', PointCloud2, self.cluster_callback, queue_size=1, buff_size=2**24, tcp_nodelay=True)

        rospy.loginfo("%s - initialized", rospy.get_name())


    def cluster_callback(self, msg):
        data = numpify(msg)
        points = structured_to_unstructured(data[['x', 'y', 'z', 'label']], dtype=np.float32)
        
        if points.shape[0] == 0:
            rospy.logwarn("Received empty clustered point cloud. Skipping detection.")
            return 

        transform = None

        try:
            transform = self.tf_buffer.lookup_transform(self.output_frame, msg.header.frame_id, msg.header.stamp, rospy.Duration(self.transform_timeout))
        except (TransformException, rospy.ROSTimeMovedBackwardsException) as e:
            rospy.logwarn("%s - Failed to look up transform: %s", rospy.get_name(), e)
            return 

        tf_matrix = numpify(transform.transform).astype(np.float32)
        all_labels = points[:, 3].copy()
        # make copy of points
        points = points.copy()
        # turn into homogeneous coordinates
        points[:,3] = 1
        # transform points to target frame
        points = points.dot(tf_matrix.T)
        points[:, 3] = all_labels
        
        pub_msg = DetectedObjectArray()
        pub_msg.header.stamp = msg.header.stamp
        pub_msg.header.frame_id = self.output_frame

        all_labels_int = points[:, 3].astype(np.int32)
        cluster_labels = np.unique(all_labels_int) 


        for label in cluster_labels:
            mask = (all_labels_int == label)
            points3d = points[mask, :3]
            
            if points3d.shape[0] < self.min_cluster_size:
                continue

            centroid = np.mean(points3d, axis=0)
            centroid_x, centroid_y, centroid_z = centroid

            points_2d = MultiPoint(points3d[:,:2])
            hull = points_2d.convex_hull
            
            convex_hull_points = [a for hull_coords in [[x, y, centroid_z] for x, y in hull.exterior.coords] for a in hull_coords]

            
            obj = DetectedObject()
            
            obj.id = int(label)
            obj.label = "unknown"
            obj.color = BLUE80P
            obj.valid = True
            obj.position_reliable = True
            obj.velocity_reliable = False
            obj.acceleration_reliable = False
            
            obj.centroid.x = centroid_x
            obj.centroid.y = centroid_y
            obj.centroid.z = centroid_z
            
            obj.convex_hull = convex_hull_points
            pub_msg.objects.append(obj)
            
        self.objects_pub.publish(pub_msg)         
        print()




    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('cluster_detector', log_level=rospy.INFO)
    node = ClusterDetector()
    node.run()