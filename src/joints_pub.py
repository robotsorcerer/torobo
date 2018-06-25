#!/usr/bin/env python

import rospy
from rospy.numpy_msg import numpy_msg
from trac_ik_torobo.msg import Numpy64

import os
import numpy as np
from os.path import expanduser, join


def talker(data):
    pub = rospy.Publisher('/torobo/teach_joints', numpy_msg(Numpy64),queue_size=1)
    r = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        pub.publish(data)
        r.sleep()

if __name__ == '__main__':
    filepath = join(expanduser('~'), 'Documents', 'LyapunovLearner', 'ToroboTakahashi', 'data')
    name     = 'state_joint_pos_only.npy'
    filename = join(filepath, name)
    data_raw = np.load(filename)
    print(data_raw.shape)
    data     =  np.ravel(data_raw, order='A')
    np.set_printoptions(suppress=True)
    print(data[:70], data.shape)
    try:
        rospy.init_node('joints_pub_node')
        talker(data)
        rospy.spin()
    except KeyboardInterrupt:
        rospy.logerror("shutting down ros")
