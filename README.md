This package provides examples programs to use the standalone TRAC-IK solver and related code.

Currently, there only exists an ik\_tests program that compares KDL's Pseudoinverse Jacobian IK solver with TRAC-IK.  The pr2_arm.launch files runs this test on the default PR2 robot's 7-DOF right arm chain.

###As of v1.4.3, this package is part of the ROS Indigo/Jade binaries: `sudo apt-get install ros-jade-trac-ik`


### Launch the Server for Torobo
+  First launch the joint to cartesian launcher that converts recored joint angles during the teach motion to cartesian coordinates
+ Then launch lyapunov learner main


### Calling IK solver
+ This may consist of calling the server like so in terminal"
    - Note that you can call this  service like so:
    ```
        rosservice call /torobo/solve_diff_ik "'desired_vel': {'linear': {'x': 0.0, 'y': 0.1, 'z': 0.2}}"
    ```
