#ifndef __IK_SOLVER_H__
#define __IK_SOLVER_H__

#include <dr_kdl/dr_kdl.hpp>
#include <sensor_msgs/JointState.h>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <trac_ik_torobo/SolveDiffIK.h>

namespace pfn{

  class IKVelocitySolver{
  public:
    	/// KDL tree.
    	dr::KdlTree tree_;
    	/// Inverse differential kinematics solver.
    	std::unique_ptr<KDL::ChainIkSolverVel_pinv> solver_;
    	/// Ros node.
    	ros::NodeHandle nh_;
    	/// Service server.
    	ros::ServiceServer diff_ik_server_;
    	ros::Publisher answer_publisher_;
      friend class Converter; // fwd declaration
  public:
    IKVelocitySolver(ros::NodeHandle nh);
    bool onSolveRequest(trac_ik_torobo::SolveDiffIK::Request & req, trac_ik_torobo::SolveDiffIK::Response & res);

  };

}

#endif
