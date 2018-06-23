#! /usr/bin/env python
import os
import rospy
import argparse
import numpy as np
from trac_ik_torobo.srv import Joints
from os.path import expanduser, join
from std_msgs.msg import Float32MultiArray

parser = argparse.ArgumentParser(description='teacher')
parser.add_argument('--teach', '-te', type=bool, default=0)
parser.add_argument('--publish', '-pb', type=bool, default=0)
args = parser.parse_args()

print(args)

def send_joints(data):
	rospy.wait_for_service('joints')

	try:
		joints_to_send = rospy.ServiceProxy('joints', Joints)
		resp = joints_to_send(data)
		return resp.joints
	except rospy.ServiceException, e:
		print "Service call failed: %s"%e


if __name__ == '__main__':
	rospy.init_node('joints_service_node')

	filepath = join(expanduser('~'), 'Documents', 'LyapunovLearner', 'ToroboTakahashi', 'data')
	name = 'state_joint.npy'
	filename = join(filepath, name)

	data = np.load(filename)

	send_joints(data)
