#! /usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from trajectory_msgs.msg import JointTrajectory

import math
import csv
import zmq
import os
from datetime import datetime
import json
import time

joint_name_map = {"first_joint":1, "second_joint":2, "third_joint":3, "fourth_joint": 4, "fifth_joint":5, "sixth_joint":6, "right_finger_joint": 7}

#  Socket to talk to server
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

def callback(data):
    rospy.loginfo("ToroboArm Driver:: Received new command.")
    rospy.loginfo("ToroboArm Driver:: " + rospy.get_caller_id())
    sendCommand(data)

def sendCommand(data):
    rospy.loginfo("ToroboArm Driver:: Send Command to Toroboarm")

    # os.mkdir('~/pvt_points')
    filepath = "/tmp/" + datetime.now().strftime("%Y%m%d%H%M%S") + "_pvt.csv"
    f = open(filepath, "ab")
    csvWriter = csv.writer(f)

    # Remap data to each dict
    # Transform radius to degree
    rows = []
    for point in data.points:
        row = []
        row.append(point.time_from_start.secs + point.time_from_start.nsecs * 10 ** -9)
        positions = {} # [deg]
        velocities = {} # [deg/sec]
        accelerations = {} # [deg/sec2]

        # Remapping
        index = 0
        for joint_name in data.joint_names:
            positions[str(joint_name_map[joint_name])] = math.degrees(point.positions[index])
            velocities[str(joint_name_map[joint_name])] = math.degrees(point.velocities[index])
            accelerations[str(joint_name_map[joint_name])] = math.degrees(point.accelerations[index])
            index += 1

        # Create CSV Row Data
        for num in range(len(data.joint_names)):
            row.append(positions[str(num+1)])
            row.append(velocities[str(num+1)])

        # Append to List
        rows.append(row)

    # Write List to CSV
    csvWriter.writerows(rows)
    f.close()
    rospy.loginfo("command_sender: " + filepath + " is generated.")
    time.sleep(1)

    # Send Command
    command_json = {"command": "--tpvt", "csv_file": filepath}
    socket.send(json.dumps(command_json))
    res = socket.recv()

    time.sleep(2)

    command_json = {"command": "--ts"}
    socket.send(json.dumps(command_json))
    res = socket.recv()
    rospy.loginfo("Response: " + res)

def listener():
    rospy.init_node('command_sender', anonymous=True)
    rospy.Subscriber('/joint_path_command', JointTrajectory, callback)
    rospy.spin()

def main():
    listener()

if __name__ == '__main__':
    main()
