#include <unistd.h>
#include <iostream>
#include <array>
#include <chrono>
#include <thread>
#include <boost/date_time.hpp>
#include <trac_ik/trac_ik.hpp>
#include <ros/ros.h>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <std_msgs/Float32.h>

#include <Eigen/Core>

#include <rospy_tutorials/Floats.h>

rospy_tutorials::Floats np_msg_data;

std::vector<std::vector<double>> jointsarray;
Eigen::MatrixXd RawJoints(10001, 7);

void joints_cb(const rospy_tutorials::Floats::ConstPtr& np_msg)
{
    ROS_INFO_STREAM("numpy message : " << np_msg->data.size());
    RawJoints.resize(10001, 7);
    for (int i=0; i < 10001; ++i)
    {
      for (auto j =0; j < 7; ++j)
      {
        RawJoints(i, j) = np_msg->data[i*7+j];
      }
    //  std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

std::vector<KDL::Frame>  convert(Eigen::MatrixXd&& saved_joints)
{
  double eps = 1e-5;
  double num_waypts = saved_joints.rows();


  if (num_waypts < 1)
  {  num_waypts = 1; }

  KDL::Chain chain;
  KDL::JntArray ll, ul; //lower joint limits, upper joint limits

  // Set up KDL IK
  KDL::ChainFkSolverPos_recursive fk_solver(chain); // Forward kin. solver
  KDL::ChainIkSolverVel_pinv vik_solver(chain); // PseudoInverse vel solver
  KDL::ChainIkSolverPos_NR_JL kdl_solver(chain,ll,ul,fk_solver, vik_solver, 1, eps); // Joint Limit Solver

  // populate
  int num_jts = 7;
  KDL::JntArray q(num_jts);
  bool kinematics_status;
  std::vector<KDL::Frame> CartPosList;

  KDL::Frame cartpos;
  for (auto i=0; i < num_waypts; i++) {
    for (uint j=0; j<7; j++) {  // positions go from indices 1 through 8
      q(j)=saved_joints(i, j);
    }

    kinematics_status = fk_solver.JntToCart(q, cartpos);
    if(kinematics_status>=0){
        // std::cout << cartpos.p.x() << cartpos.p.y() << cartpos.p.z()<<std::endl;
        // ROS_INFO("%s \n","KDL FK Succeeded!");
        CartPosList.push_back(cartpos);
        //std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    else{
        ROS_INFO("%s \n","Error: could not calculate forward kinematics :(");
      }
  }
    ROS_INFO_STREAM("Cart List size  " << CartPosList.size());
    return CartPosList;
  }

int main(int argc, char** argv)
{
  ros::init(argc, argv, "ik_torobo");
  ros::NodeHandle nh("~");

  ros::Subscriber sub = nh.subscribe("/torobo/teach_joints", 1000, joints_cb);

  double timeout;
  std::vector<KDL::Frame> CartPosList = convert(std::move(RawJoints)); //, std::move(CartPosList));

  ros::spinOnce();


  std::cout << std::fixed;
  std::cout << std::setprecision(4);
  std::cout << "Raw Joints first 10:\n " << RawJoints.block(0, 0, 10, 7) << "\n";

  if (!ros::ok()){
    ros::shutdown();
  }
  // for(auto it = CartPosList.cbegin(); it != CartPosList.cend(); ++it)
  // {
  //   ROS_INFO("[x, y, z]: [%.4f, %.4f, %.4f]", it->p.x(), it->p.y(), it->p.z());
  // }
  return 0;
}
