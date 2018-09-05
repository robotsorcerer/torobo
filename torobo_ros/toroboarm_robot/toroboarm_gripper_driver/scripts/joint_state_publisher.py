#! /usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import zmq
import json
import math

def talker():
    # Create ZMQ client to communicate with ToroboArmManager
    rospy.loginfo("Connect to ToroboArmManager.")
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect("tcp://127.0.0.1:5554")
    subscriber.setsockopt(zmq.SUBSCRIBE, b"ToroboArmManager")

    # Create ROS Message Publisher
    rospy.loginfo("Create Publisher.")
    pub = rospy.Publisher('joint_states', JointState, queue_size=10)
    rospy.init_node('toroboarm_gripper_driver')
    # rate = rospy.Rate(10) # 10hz

    # Initialize Parameters
    joint_state = JointState()
    joint_state.header = Header()
    joint_state.header.stamp = rospy.Time.now()
    joint_state.name = ['first_joint', 'second_joint', 'third_joint', 'fourth_joint', 'fifth_joint', 'sixth_joint', 'right_finger_joint']
    joint_state.position = [0, 0, 0, 0, 0, 0, 0]
    joint_state.velocity = [0, 0, 0, 0, 0, 0, 0]
    joint_state.effort = [0, 0, 0, 0, 0, 0, 0]

    while not rospy.is_shutdown():
        rospy.loginfo("Wait for new message.")
        # Receive Message From ToroboArmController
        [address, contents] = subscriber.recv_multipart()
        rospy.loginfo(address + ": " + contents)

        # Parse contents
        cur_state = json.loads(contents)
        for i in range(7):
            joint_state.position[i] = math.radians(cur_state["jointState"][i]["position"])
            joint_state.velocity[i] = math.radians(cur_state["jointState"][i]["velocity"])
            joint_state.effort[i] = math.radians(cur_state["jointState"][i]["torque"])

        # Publish Status as a ROS message
        joint_state.header.stamp = rospy.Time.now()
        rospy.loginfo(joint_state)
        pub.publish(joint_state)
        # # rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
