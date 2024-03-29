#include <ros/ros.h>
#include <ros/time.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <actionlib/server/simple_action_server.h>
#include <control_msgs/FollowJointTrajectoryAction.h>

class ToroboArm
{
protected:
  ros::NodeHandle nh_;

  actionlib::SimpleActionServer<control_msgs::FollowJointTrajectoryAction> as_;
  std::string action_name_;

  trajectory_msgs::JointTrajectory goal_;

  ros::Subscriber sub_pose_arm_;
  ros::Publisher pub_move_arm_;

public:

  ToroboArm(std::string name) :
    as_(nh_, name, false),
    action_name_(name)
  {
    as_.registerGoalCallback(boost::bind(&ToroboArm::goalCB, this));
    as_.registerPreemptCallback(boost::bind(&ToroboArm::preemptCB, this));

    pub_move_arm_ = nh_.advertise<trajectory_msgs::JointTrajectory>("/joint_path_command", 1, this);

    as_.start();
  }

  void goalCB()
  {
	goal_ = as_.acceptNewGoal()->trajectory;
	ROS_INFO("Goal Recieived");
	pub_move_arm_.publish(goal_);
  }

  void preemptCB()
  {
    ROS_INFO("%s: Preempted", action_name_.c_str());
    as_.setPreempted();
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "toroboarm/whole_arm_controller/follow_joint_trajectory");

  ToroboArm toroboarm(ros::this_node::getName());
  ros::spin();

  return 0;
}
