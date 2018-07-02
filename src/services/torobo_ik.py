#! /usr/bin/env python
import os
import rospy
import numpy as np
from trac_ik_torobo.srv import SolveDiffIK, SolveDiffIKRequest, SolveDiffIKResponse
from geometry_msgs.msg import Twist

carts_to_send = rospy.ServiceProxy("/torobo/solve_diff_ik", SolveDiffIK)

def send_carts(msg):
	rospy.wait_for_service("/torobo/solve_diff_ik")

	try:
		resp = carts_to_send(msg)
		return SolveDiffIKResponse(resp.q_out)
	except rospy.ServiceException, e:
		print "Service call failed: %s"%e


if __name__ == '__main__':
	rospy.init_node('joints_service_node')

	msg = Twist()
	msg.linear.x  = -0.191556
	msg.linear.y  = -0.216731
	msg.linear.z  =  0.963396
	msg.angular.x = -2.05147
	msg.angular.y = -2.32108
	msg.angular.z = -0.766848

	q_in = [0,0,0,0,0,0,0]

	tosend = SolveDiffIKRequest()
	tosend.desired_vel = msg
	tosend.q_in = q_in

	rate = rospy.Rate(30)
	while not rospy.is_shutdown():
		resp = send_carts(tosend)
		print(resp)
		rate.sleep()
