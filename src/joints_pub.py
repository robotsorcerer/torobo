#!/usr/bin/env python

import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

import os
import numpy as np
from os.path import expanduser, join


def talker(data):
    #pub = rospy.Publisher('/torobo/teach_joints', (Floats),queue_size=10)
    pub = rospy.Publisher('/torobo/teach_joints', numpy_msg(Floats),queue_size=10)
    r = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        pub.publish(data)
        r.sleep()

if __name__ == '__main__':
    filepath = join(expanduser('~'), 'Documents', 'LyapunovLearner', 'ToroboTakahashi', 'data')
    name = 'state_joint_pos_only.npy'
    filename = join(filepath, name)
    data =  np.ravel(np.load(filename))
    try:
        rospy.init_node('joints_pub_node')
        talker(data)
        rospy.spin()
    except KeyboardInterrupt:
        rospy.logerror("shutting down ros")
