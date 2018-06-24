#!/usr/bin/env python
PKG = 'numpy_tutorial'
#import roslib; roslib.load_manifest(PKG)

import rospy
import numpy as np
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg

def callback(data):
    np.set_printoptions(suppress=True)
    print rospy.get_name(), "I heard %s"%str(data.data.shape)
    ret_data = data.data.astype(np.float64)
    print(ret_data[:70])#.reshape(10, 7))

def listener():
    rospy.init_node('listener')
    rospy.Subscriber("/torobo/teach_joints", numpy_msg(Floats), callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
