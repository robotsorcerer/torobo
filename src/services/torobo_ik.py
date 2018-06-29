#! /usr/bin/env python
import os
import rospy
import numpy as np
from trac_ik_torobo.srv import SolveDiffIK, SolveDiffIKRequest, SolveDiffIKResponse
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist



carts_to_send = rospy.ServiceProxy("/torobo/solve_diff_ik", SolveDiffIK)

def send_carts(msg):
	rospy.wait_for_service("/torobo/solve_diff_ik")

	try:
		resp = carts_to_send(msg)
		print(resp)
		# return resp.q_out
		return SolveDiffIKResponse(resp.q_out)
	except rospy.ServiceException, e:
		print "Service call failed: %s"%e


if __name__ == '__main__':
	rospy.init_node('joints_service_node')

	req = SolveDiffIKRequest
	req.desired_vel.linear.x = -0.191556
	req.desired_vel.linear.y = -0.216731
	req.desired_vel.linear.z = 0.963396
	req.desired_vel.angular.x = -2.05147
	req.desired_vel.angular.y = -2.32108
	req.desired_vel.angular.z = -0.766848
	req.q_in = [0,0,0,0,0,0,0]

	# msg {
	# 'desired_vel': {
	# 'linear': {
	# 	'x': -0.191556,
	# 	'y': -0.216731,
	# 	'z': 0.963396,
	# 	},
	# 'angular': {
	# 	'x': -2.05147,
	# 	'y': -2.32108,
	# 	'z':-0.766848,
	# 	},
	#  }
	# }
	# 	msg = Twist()
	# 	msg.linear.x = -0.191556
	# 	msg.linear.y = -0.216731
	# 	msg.linear.z = 0.963396
	# 	msg.angular.x = -2.05147
	# 	msg.angular.y = -2.32108
	# 	msg.angular.z = -0.766848
	# }

	# qinit = Float64()
	# qinit = [0.0] * 7
	rate = rospy.Rate(30)
	while not rospy.is_shutdown():
		resp = send_carts(req)
		rate.sleep()
