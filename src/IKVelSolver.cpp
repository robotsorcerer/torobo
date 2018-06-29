#include <ros/ros.h>
#include <trac_ik_torobo/ik_solver.h>

#include <memory>

using namespace pfn;

/*
Note that you can call this  service like so:
rosservice call /torobo/solve_diff_ik "'desired_vel': {'linear': {'x': 0.0, 'y': 0.1, 'z': 0.2}}"


*/
IKVelocitySolver::IKVelocitySolver(ros::NodeHandle nh)
  : nh_(nh)
  {
    		solver_ = std::make_unique<KDL::ChainIkSolverVel_pinv>(tree_.getChain("link0", "link7"));
    		//diff_ik_server_ = nh_.advertiseService("/torobo/solve_diff_ik", &IKVelocitySolver::onSolveRequest, this);
    		answer_publisher_ = nh_.advertise<sensor_msgs::JointState>("result_states", 100, false);
  }

bool IKVelocitySolver::onSolveRequest(trac_ik_torobo::SolveDiffIK::Request & req, trac_ik_torobo::SolveDiffIK::Response & res)
      {
    		KDL::JntArray q_in(7);
    		KDL::JntArray q_out(7);
    		KDL::Vector rot, trans;

    		trans.x(req.desired_vel.linear.x);
    		trans.y(req.desired_vel.linear.y);
    		trans.z(req.desired_vel.linear.z);
    		rot.x(req.desired_vel.angular.x);
    		rot.y(req.desired_vel.angular.y);
    		rot.z(req.desired_vel.angular.z);

    		KDL::Twist desired_vel{trans, rot};
    		for (size_t i = 0; i < req.q_in.size(); i++) {
    			q_in(i) = req.q_in.at(i);
    		}

    		solver_->CartToJnt(q_in, desired_vel, q_out);
    		res.q_out.resize(q_out.rows());
    		for (size_t i = 0; i < res.q_out.size(); i++) {
    			res.q_out.at(i) = q_out(i);
    		}

    		sensor_msgs::JointState result_state;
    		result_state.header.stamp = ros::Time::now();
    		result_state.velocity = res.q_out;
    		answer_publisher_.publish(result_state);

    		return !res.q_out.empty();
    }
//copy constructor
// IKVelocitySolver::IKVelocitySolver()
// int main(int argc, char** argv) {
// 	ros::init(argc, argv, "torobo_ik_vel");
// 	pfn::IKVelocitySolver solver;
//
// 	ros::Rate r{100};
//
// 	while (ros::ok()) {
// 		ros::spinOnce();
// 		r.sleep();
// 	}
//
// 	ros::waitForShutdown();
// 	return 0;
// }
