toroboarm_robot
========
Specific Components that are used for ToroboArm.

## Environment

* ROS Indigo
* Ubuntu 14.04

## Installation

Clone & Make it!

```shell
$ cd ~/catkin_ws/src
$ git clone git@bitbucket.org:tokyorobotics/toroboarm_robot.git
$ cd ~/catkin_ws
$ rosdep install --from-paths src -iy
$ catkin_make
```

## Visualize ToroboArm's 3D Model in rviz

Please execute below.

```shell
$ roslaunch toroboarm_description display_toroboarm.launch
```

## Motion Planning Demo (MoveIt!)

Please execute below.

```shell
$ roslaunch toroboarm_moveit_config demo.launch
```

## Motion Planning Simulation (MoveIt! + Gazebo)

Please execute below.
```
$ roslaunch toroboarm_bringup bringup_sim.launch
```

## Motion Plannning with ToroboArm (MoveIt! + ToroboArm)

Please execute below.
```
$ roslaunch toroboarm_bringup bringup_real.launch
```
