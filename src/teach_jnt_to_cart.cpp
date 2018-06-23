#include <unistd.h>
#include "cnpy.h"
#include <fstream>
#include <string>
#include <complex>
#include <boost/date_time.hpp>
#include <trac_ik/trac_ik.hpp>
#include <ros/ros.h>
#include <kdl/chainiksolverpos_nr_jl.hpp>

#include <trac_ik_torobo/Joints.h>
#include <std_msgs/Float64.h>


std::string saved_npy_path = "/home/olalekan/Documents/LyapunovLearner/ToroboTakahashi/data/state_joint.npy";
std::string saved_txt_path = "/home/olalekan/Documents/LyapunovLearner/ToroboTakahashi/data/state_joint.csv";


std::vector<std::string> read_data(const std::string& saved_txt_path){
   std::ifstream file_stream(saved_txt_path);
   if(!file_stream.is_open())
      ROS_FATAL("Could not open file");
   std::vector<std::string> arr;

   while(!file_stream.eof())
   {
        for (std::string joints; file_stream.getline(&joints[0], 1000, '\n'); ) {
            arr.push_back(joints);
        }
   }
   for(auto joints : arr)
      ROS_INFO_STREAM( joints);
   return arr;
 }

void convert(const std::vector<std::string>& saved_joints,  double timeout)
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

  ros::init(argc, argv, "ik_torobo");
  ros::NodeHandle nh("~");

  auto saved_joints = read_data(saved_txt_path);

  int num_waypts;
  double timeout;
  //convert(saved_joints, timeout);

  ros::spin();

  return 0;
}
