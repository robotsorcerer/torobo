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

def prepro(file_path):
    # ripped of /lyapunovlearner/notes/multi_saver.ipynb
    names = []
    for _ in range(1, 10):
        names.append('state_joint_pos_vel{}.csv'.format(_))

    skips = [names[1], names[7], names[8]]
    names_other = sorted(list(set(names).difference(skips)))
    # print('names: ', ['{}'.format(x) for x in names],'\nnames_other: ', names_other)

    raw_data = dict()
    for name in names_other:
        with open(join(file_path, name), 'r+') as f:
            data = f.readlines()

        moplan_data = [x.split("\n") for x in data]

        proper_data = []
        for i in range(0, len(moplan_data), 3):
            if not moplan_data[i]:
                continue
            temp = moplan_data[i:i+3]
            # if name == 'state_joint_pos_vel8.csv':
            #     print('i: ', i, len(temp))
            #     if i > 944:
            #         break
            temp = temp[0] + temp[1] + temp[2]
            [temp.remove('') for x in range(temp.count(''))]
            to_append = []
            for x in temp:
                if '[' in x:
                    x = x.split("[")[1]
                elif ']' in x:
                    x = x.split(']')[0]
                to_append += [float(xx) for xx in x.split()]
            proper_data.append(to_append)

        proper_data = np.array(proper_data)

        raw_data[name] = proper_data

    #find max row in all of gathered data and extend accordingly
    max_row = max([x.shape[0] for x in raw_data.values()])

    # now form augmented data
    for keys, data in raw_data.items():
        if data.ndim > 1:
            if data.shape[0] < max_row:
                temp = np.zeros((max_row, data.shape[1]))
                temp[:data.shape[0], :data.shape[1]] = data
                temp[data.shape[0]:, :data.shape[1]] = data[-1]
                raw_data[keys] = temp

    #print('verifying consistency of data')
    i = 0
    for keys, data in raw_data.items():
        if data.ndim > 1:
            if i == 0:
                all_data = data
            else:
                all_data = np.r_[all_data, data]
            i += 1


    return all_data   # will be [(data_len x N)  x 14] >> N = # of experiments

def talker(data):
    pub = rospy.Publisher('/torobo_ik/teach_joints', numpy_msg(Numpy64),queue_size=1)
    r = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        pub.publish(data)
        r.sleep()

if __name__ == '__main__':
    filepath = join(lyap, 'scripts', 'data', 'aug_16')

    data_raw = prepro(filepath)

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
