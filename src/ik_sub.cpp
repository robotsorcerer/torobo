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
using namespace Eigen;

#include <rospy_tutorials/Floats.h>

rospy_tutorials::Floats np_msg_data;

std::vector<std::vector<double>> jointsarray;

void joints_cb(const rospy_tutorials::Floats::ConstPtr& np_msg)
{
  //np_msg_data = np_msg;
  ROS_INFO("size: [%d]", np_msg->data.size());
    for (int i=0; i < np_msg->data.size(); ++i)
    {
      // for (auto j =0; j < 8; ++j)
      // {
      //   jointsarray[i].push_back(np_msg->data[j]);
      // }
      std::cout << np_msg->data[i] << "\n";
      std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }
   //ROS_INFO("numpy message : [%s]" << np_msg->data.c_str());
}

void convert(const std_msgs::Float32& saved_joints,  double timeout)
{
  double eps = 1e-5;
  double num_waypts = 10001; //sizeof(saved_joints)/sizeof(saved_joints[0]); //[0];
  ROS_INFO_STREAM("num_waypts: " << num_waypts);
  // std::cout << "saved_joints" << saved_joints << "\n";


  if (num_waypts < 1)
  {  num_waypts = 1; }

  KDL::Chain chain;
  KDL::JntArray ll, ul; //lower joint limits, upper joint limits

  // Set up KDL IK
  KDL::ChainFkSolverPos_recursive fk_solver(chain); // Forward kin. solver
  KDL::ChainIkSolverVel_pinv vik_solver(chain); // PseudoInverse vel solver
  KDL::ChainIkSolverPos_NR_JL kdl_solver(chain,ll,ul,fk_solver, vik_solver, 1, eps); // Joint Limit Solver

  // populate
  std::vector<KDL::JntArray> JointList;
  std::vector<KDL::Frame> CartPosList;
  KDL::JntArray q(chain.getNrOfJoints());
/*
  for(int rows=0; rows< saved_joints.shape[0]; ++rows){
    KDL::Frame cartpos;
    for (int cols=1; cols< 8; ++cols){
      // q(cols)=saved_joints(rows)(cols);
    }
    // ROS_INFO_STREAM("q: " << q );
  }
  */
  // for (uint i=0; i < num_waypts; i++) {
  //   KDL::Frame cartpos;
  //   for (uint j=1; j<8; j++) {  // positions go from indices 1 through 8
  //     q(j)=saved_joints[i][j];
  //   }
  //   bool kinematics_status;
  //   kinematics_status = fk_solver.JntToCart(JointList[i], cartpos);
  //   if(kinematics_status>=0){
  //       std::cout << cartpos.p.x() << cartpos.p.y() << cartpos.p.z()<<std::endl;
  //       ROS_INFO("%s \n","KDL FK Succeeded!");
  //       CartPosList.push_back(cartpos);
  //   }
  //   else{
  //       printf("%s \n","Error: could not calculate forward kinematics :(");
  //     }
  //   JointList.push_back(q);
  // }

  ROS_INFO_STREAM("Joint List size  " << JointList.size());
    ROS_INFO_STREAM("Cart List size  " << CartPosList.size());

  }



int main(int argc, char** argv)
{
  srand(1);
  ros::init(argc, argv, "ik_torobo");
  ros::NodeHandle nh("~");

  ros::Subscriber sub = nh.subscribe("/torobo/teach_joints", 1000, joints_cb);

  double timeout;
  //ROS_INFO_STREAM("numpy message : " << np_msg_data.data);
  //convert(saved_joints, timeout);
  //for(auto it=jointsarray.cbegin(); it!=jointsarray.cend(); ++it)
  //{
  //  std::cout<<" " << *it << "\n";
  //}

  ros::spin();

  return 0;
}
