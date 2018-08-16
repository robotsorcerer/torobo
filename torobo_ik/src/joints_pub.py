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

def prepro(filename):
    with open(filename, 'r+') as foo:
        data = foo.readlines()
    moplan_data = [x.split() for x in data]   

    proper_data = []
    for i in range(len(moplan_data)):
        if not moplan_data[i]:
            continue
        temp = moplan_data[i]
        to_append = []
        temp = [x.split(',')[0] for x in temp]
        for x in temp:
            if '[' in x:
                x = x.split("[")[1]
            elif ']' in x:
                x = x.split(']')[0]
            to_append.append(float(x))
        proper_data.append(to_append)

    proper_data = np.array(proper_data)

    return proper_data

def talker(data):
    pub = rospy.Publisher('/torobo_ik/teach_joints', numpy_msg(Numpy64),queue_size=1)
    r = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        pub.publish(data)
        r.sleep()

if __name__ == '__main__':
    filepath = join(lyap, 'scripts', 'data')
    name     = 'state_joint_1.csv'
    filename = join(filepath, name)

    data_raw = prepro(filename)

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
