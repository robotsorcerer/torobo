#! /usr/bin/env /usr/bin/python
from __future__ import print_function

import rospy
import random
import logging

from moveit_msgs.srv import GetPositionIK, GetPositionIKResponse
from moveit_msgs.msg import PositionIKRequest


LOGGER = logging.getLogger(__name__)

def get_ik_client(ik_request):
	rospy.wait_for_service('/compute_ik')
	try:
		retrieve_ik = rospy.ServiceProxy('compute_ik', GetPositionIK)

		resp = retrieve_ik(ik_request)

		print(resp)
		return GetPositionIKResponse(resp.solution.joint_state)
	except rospy.ServiceException, e:
		LOGGER.debug("Service call failed: %s"%e)


if __name__ == '__main__':
	ik_request = PositionIKRequest()
	x = random.random()
	print('x: ', x)
	ik_request.robot_state.joint_state.position = [0.0]*3
	ik_request.robot_state.joint_state.velocity = [x, 2*x, 4*x]

	resp = get_ik_client(ik_request)
	print(resp)
