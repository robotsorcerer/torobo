#! /usr/bin/env python
import os
import rospy
import numpy as np
from os.path import expanduser, join

from trac_ik_torobo.srv import Joints


def handle_joint_angles(req):
    print "returning joints: ", req.joints
	return JointsResponse(data)

def send_joint_angles(data):
    try:
        s = rospy.Service("/torobo/teach_joints", Joints, handle_joint_angles)
        rospy.spin()
    except KeyboardInterrupt:
        LOGGER.critical("shutting down ros")


if __name__ == '__main__':
    filepath = join(expanduser('~'), 'Documents', 'LyapunovLearner', 'ToroboTakahashi', 'data')
    name = 'state_joint.npy'
    filename = join(filepath, name)
    data =  np.load(filename)

    rospy.init_node('joints_service_node')
    send_joint_angles(data)
