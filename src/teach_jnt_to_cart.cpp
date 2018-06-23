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
//std_msgs::Float64* np_data;
//std::vector<double> np_data;
/*
cnpy::NpyArray get_joints(const std::string& saved_npy_path){
  cnpy::NpyArray joints_data = cnpy::npy_load(saved_npy_path);
  ROS_INFO_STREAM("saved array shape: " << joints_data.shape[0] << ", " << joints_data.shape[1]);
  ROS_INFO_STREAM("saved array size: " << joints_data.shape.size());
  ROS_INFO_STREAM("saved array word size: " << joints_data.word_size);

  auto joint_values =joints_data.data<std::vector<double>>();
  ROS_INFO_STREAM("joint_values " << joint_values);
  for (auto it = joint_values->cbegin(); it != joint_values->cend(); ++it)
    ROS_INFO_STREAM("joint_values " << *it);

  return joints_data;
}
*/
std::string read_data(const std::string& saved_txt_path){
   std::ifstream file_stream(saved_txt_path);
   if(!file_stream.is_open())
      ROS_FATAL("Could not open file");
   std::string str;
   std::vector<std::string> arr;
   std::string file_contents;

   while(!filestream.eof())
   {
        for (std::string joints; file_stream.getline(&joints[0], 1000, '\n'); ) {
            arr.push_back(joints);
        }
   }
   for(auto joints : arr):
      ROS_INFO_STREAM( joints);
   return arr;
 }

void convert(const std_msgs::Float64& saved_joints,  double timeout)
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

bool joints(trac_ik_torobo::Joints::Request &req,
            trac_ik_torobo::Joints::Response &res)
{
  //np_data->data = res.joints;
  np_data.push_back(res.joints);
  // ROS_INFO_STREAM("res: " << res.joints);

  return true;
}


int main(int argc, char** argv)
{
  srand(1);
  ros::init(argc, argv, "ik_torobo");
  ros::NodeHandle nh("~");

  // ros::ServiceServer server = nh.advertiseService("/torobo/teach_joints", joints);
  //nh.serviceClient client ;
  //client = nh.serviceClient<trac_ik_torobo::Joints>("/torobo/teach_joints");
  //trac_ik_torobo::Joints srv;
  //ros::service::call("/torobo/teach_joints", srv);

  //client.call(srv);


  int num_waypts;
  double timeout;

  //auto saved_joints = srv.response;
  //ROS_INFO_STREAM("saved_joints: " << saved_joints );
 ß auto saved_joints = read_data(saved_txt_path);


  convert(saved_joints, timeout);

  ros::spin();

  return 0;
}
