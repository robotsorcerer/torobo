### Introduction

This codebase represents a wip for our proposed learning stable control laws for nonlinear complex robot trajectories.

### Packages Organization

+ [Torobo_ik](/torobo_ik): Contains the ik solver and service calls' source code for converting the joint positions of the robot to cartesian workspace. This leverages on the `TRAC_IK_SOLVER` and the `KDL_LIBRARY`.

+ [Torobo Control](/torobo_control): Details the meta-package for the whole packages that we use in this project.

+ [LyapunovLearner](/lyapunovlearner): Basic implementation of Khansari-Zadeh's `Stable Estimator of Dynamical Systems` paper and `Learning Control Lyapunov Functions for Nonlinear Robot Trajectories.`

+ [LWPR](/dp_lwpr/StatisticalLearning/lwpr): This is an implementation of Stefan Schaal and Sethu Vijaykumar's `Locally Weighted Projected Regression` from their 2003 ICML [paper](https://scholar.google.com/scholar_url?url=http://wcms.inf.ed.ac.uk/ipab/rlsc/lecture-notes/vijayakumar-ICML2000.pdf&hl=en&sa=T&oi=gsb-ggp&ct=res&cd=0&d=7613324365422160615&ei=KeVaW6SIL8aNywSrtqqwBw&scisig=AAGBfm2s8fnR6yjG_UE14p03vxjksCKbMw). This is the test [Matlab](/dp_lwpr/StatisticalLearning/lwpr) source code for

  -  [1D data](/dp_lwpr/StatisticalLearning/lwpr/test_lwpr_1D.m),

  -  [2D data](/dp_lwpr/StatisticalLearning/lwpr/test_lwpr_2D.m), and

  -  [nD data](/dp_lwpr/StatisticalLearning/lwpr/test_lwpr_nD.m)

  The original matlab lwpr code is located [here](/dp_lwpr/StatisticalLearning/lwpr/lwpr.m).

+ [LWPR Translations](/dp_lwpr/src/scripts/lwpr.py): These are my translations for the example that the original authors provided in the matlab code.

  - This can be tested with `python test_lwpr_1D.py`. The current errors come from the traceback:

  ```bash
      Traceback (most recent call last):
      File "test_lwprs_1D.py", line 172, in <module>
        test_lwpr_1D([])
      File "test_lwprs_1D.py", line 79, in test_lwpr_1D
        Yp[i] = yp
      ValueError: setting an array element with a sequence.
  ```

      - Converting between matlab and python has its own gotchas, such as the ability to interpret a scalar as a matrix (a la, matlab). We need to test that this is working thoroughly before integrating it with the robot.

+ [LWPR 3rd party python source](/lwpr): This contains the source code for the python/c/mex source code by a 3rd party developer. I earlier tried to install this on a linux system as the readme file directs but I found that after running `configure`, make was giving weird errors. Would appreciate if you could look into this.
