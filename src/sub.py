#!/usr/bin/env python

import rospy
import numpy as np
from trac_ik_torobo.msg import Numpy64
from rospy.numpy_msg import numpy_msg


def callback(data):
    #print rospy.get_name(), "I heard %s"%str(data.data.shape)
    #print(data.data.dtype)
    ret_data = data.data#.astype(np.float64)
    np.set_printoptions(suppress=True)
    print(ret_data.shape)
    # print(ret_data[:70].reshape(10, 7))


def listener():
    rospy.init_node('listener')
    rospy.Subscriber("/torobo/teach_joints", numpy_msg(Numpy64), callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
