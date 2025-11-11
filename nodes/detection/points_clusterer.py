#!/usr/bin/env python3

import rospy
import numpy as np 
from sensor_msgs.msg import PointCloud2
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
from ros_numpy import numpify, msgify
from sklearn.cluster import DBSCAN

class PointClusterer:
    def __init__(self):
        cluster_min_size = rospy.get_param('~cluster_min_size')
        cluster_epsilon = rospy.get_param('~cluster_epsilon')

        self.clusterer = DBSCAN(
            eps=cluster_epsilon, 
            min_samples=cluster_min_size
        )

        self.cluster_pub = rospy.Publisher(
            'points_clustered', PointCloud2, 
            queue_size=1, 
            tcp_nodelay=True
        )

        rospy.Subscriber('points_filtered', PointCloud2, self.points_callback, queue_size=1, buff_size=2**24, tcp_nodelay=True)

    def points_callback(self, msg):
        data = numpify(msg)
        points = structured_to_unstructured(data[['x', 'y', 'z']], dtype=np.float32)
        #print('points shape:', points.shape)
        labels = self.clusterer.fit_predict(points)
        #print('labels shape:', labels.shape)

        
        labels_reshaped = labels.reshape(-1, 1) 
        
        points_labeled_all = np.hstack((points, labels_reshaped))

        mask = points_labeled_all[:, 3] != -1 
        
        points_labeled_clustered = points_labeled_all[mask]

        dtype_spec = [
           ('x', np.float32),
           ('y', np.float32),
           ('z', np.float32),
           ('label', np.int32)
        ]
        
        data_clustered = unstructured_to_structured(points_labeled_clustered, dtype=np.dtype(dtype_spec))

        cluster_msg = msgify(PointCloud2, data_clustered)
        
        cluster_msg.header.stamp = msg.header.stamp
        cluster_msg.header.frame_id = msg.header.frame_id

        self.cluster_pub.publish(cluster_msg)



    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('point_clusterer')
    node = PointClusterer()
    node.run()