#!/usr/bin/env python

import rospy
import numpy as np
from trac_ik_torobo.msg import Numpy64
from rospy.numpy_msg import numpy_msg


def callback(data):
    ret_data = data.data
    np.set_printoptions(suppress=True)


def listener():
    rospy.init_node('listener')
    rospy.Subscriber("/torobo_ik/teach_joints", numpy_msg(Numpy64), callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
