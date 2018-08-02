#!/usr/bin/env python

import rospy
import rospkg
from rospy.numpy_msg import numpy_msg
from torobo_ik.msg import Numpy64

import os
import numpy as np
from os.path import expanduser, join

rospack = rospkg.RosPack()
lyap = rospack.get_path('lyapunovlearner')

def talker(data):
    pub = rospy.Publisher('/torobo_ik/teach_joints', numpy_msg(Numpy64),queue_size=1)
    r = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        pub.publish(data)
        r.sleep()

if __name__ == '__main__':
    filepath = join(lyap, 'ToroboTakahashi', 'data')
    name     = 'state_joint.npy'
    #name     = 'state_joint_pos_only.npy'
    filename = join(filepath, name)
    data_raw = np.load(filename)
    # print(data_raw.shape)
    data     =  np.ravel(data_raw, order='A')
    np.set_printoptions(suppress=True)
    # print(data[:70], data.shape)
    try:
        rospy.init_node('joints_pub_node')
        talker(data)
        rospy.spin()
    except KeyboardInterrupt:
        rospy.logerror("shutting down ros")
