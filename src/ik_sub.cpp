#include <chrono>
#include <thread>
#include <unistd.h>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <ros/ros.h>
#include <ros/spinner.h>
#include <boost/date_time.hpp>
#include <trac_ik/trac_ik.hpp>
#include "dr_kdl/dr_kdl.hpp"
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>

#include <Eigen/Core>
#include <mutex>
#include <trac_ik_torobo/Numpy64.h>


class Converter{

  public:
      Converter(ros::NodeHandle nh)
      :nh_(nh), hardware_concurrency(std::thread::hardware_concurrency()), spinner(hardware_concurrency/6),
      save_path("/home/olalekan/Documents/LyapunovLearner/scripts/data/cart_pos.csv"),
      cartPosFile(save_path), running(false), updateJoints(false), rows(10001), cols(7), counter(0), disp(true)
      {
              sub = nh_.subscribe("/torobo/teach_joints", 1, &Converter::joints_cb, this);
      }

      ~Converter() { }

      Converter(Converter const&)=delete;
      Converter& operator=(Converter const&) = delete;

      void spawn()
      {
        begin();
        end();
      }

private:
    ros::NodeHandle nh_;
    ros::Subscriber sub;
    std::vector<std::thread> threads;
    std::vector<KDL::Frame> CartPosList;
    std::vector<std::vector<double>> jointsarray;
    Eigen::MatrixXd RawJoints;
    unsigned long const hardware_concurrency;
    ros::AsyncSpinner spinner;
    std::string save_path;
    std::ofstream cartPosFile;
    bool running, updateJoints;
    int rows, cols, counter;
    std::mutex mutex;
    const std::string urdf;
    KDL::Tree kdl_tree;
    bool disp;


private:
      void begin()
      {
        if(spinner.canStart())
        {
          spinner.start();
          ROS_INFO("spinning with %lu threads", hardware_concurrency/6);
        }

        while(!updateJoints)
        {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        threads.push_back(std::thread(&Converter::convert, this));
      	std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
      }

      void end()
      {
        spinner.stop();
      }

    void joints_cb(const trac_ik_torobo::Numpy64::ConstPtr& np_msg)
    {
        Eigen::MatrixXd RawJoints;
        // rows = np_msg->data.size();
        this->RawJoints.resize(rows, cols);

        RawJoints.resize(rows, cols);
        for (int i=0; i < rows; ++i)
        {
          for (auto j =0; j < cols; ++j)  // get only joint positions
          {
            RawJoints(i, j) = np_msg->data[i*cols+j];
          }
        }
        if( disp)
          ROS_INFO_STREAM("RawJoints \n" << RawJoints.block(0, 0, 10, 7));
        std::lock_guard<std::mutex> lock(mutex);
        this->RawJoints = RawJoints;
        updateJoints = true;
    }

    KDL::Tree get_kdl_tree()
    {
      KDL::Tree kdl_tree = dr::KdlTree::fromParameter("/robot_description");
      this->kdl_tree = kdl_tree;
      return kdl_tree;
    }

    void convert()
    {
      Eigen::MatrixXd saved_joints;
      // saved_joints.resize(rows, cols);

      if (updateJoints)
      {
          std::lock_guard<std::mutex> lock(mutex);
          saved_joints = this->RawJoints;
          int cols = saved_joints.cols();
          updateJoints = false;

          read_cart(saved_joints);
        }
      }

    void read_cart(Eigen::MatrixXd const& saved_joints)
    {
          KDL::Chain chain;
          KDL::Tree kdl_tree = get_kdl_tree();
          kdl_tree.getChain("link0", "link7", chain);

          // Set up KDL IK
          KDL::ChainFkSolverPos_recursive fk_solver = KDL::ChainFkSolverPos_recursive(chain); // Forward kin. solver
          // KDL::ChainIkSolverVel_pinv vik_solver(chain); // PseudoInverse vel solver
          // KDL::ChainIkSolverPos_NR_JL kdl_solver(chain,ll,ul,fk_solver, vik_solver, 1, eps); // Joint Limit Solver

          double elapsed = 0;
          KDL::Frame cartpos;
          bool kinematics_status = false;
          double timeout = 0.005;
          std::vector<KDL::Frame> CartPosList;
          unsigned int num_jts = chain.getNrOfJoints();

          boost::posix_time::ptime start_time;
          boost::posix_time::time_duration diff;

          int num_waypts = saved_joints.rows();
          if (disp)
            ROS_INFO("num_jts: %d, rows [%d]", num_jts, num_waypts );

          for (auto i=0; i < num_waypts; i++)
          {
              KDL::JntArray q(num_jts);
              for (int j=0; j<cols; j++)
              {
                q(j)=saved_joints(i, j);
              }

              start_time = boost::posix_time::microsec_clock::local_time();

              // do{
                  kinematics_status = fk_solver.JntToCart(q, cartpos);
                  diff = boost::posix_time::microsec_clock::local_time() - start_time;
                  elapsed = diff.total_nanoseconds() / 1e9;
                  ROS_INFO("kinematics_status: %d" , kinematics_status);
                // } while((kinematics_status == true) && (elapsed < timeout));
              CartPosList.push_back(cartpos);
          }

          print_out(CartPosList);
          ROS_INFO("Shutting down ros");
          ros::shutdown();
      }

    void print_out(const std::vector<KDL::Frame>& CartPosList)
    {
      for(auto it = CartPosList.cbegin(); it != CartPosList.cend()+20; ++it)
      {
        if(disp)
        {
          ROS_INFO("[x, y, z]: [%.4f, %.4f, %.4f]", it->p.x(), it->p.y(), it->p.z());
        }
        cartPosFile << it->p.x() << ", " << it->p.y() << ", " <<  it->p.z() << "\n";
      }
      std::cout << "system data collection done" << std::endl;
    }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "ik_torobo");
  ros::NodeHandle nh("~");

  Converter converter(nh);

  converter.spawn();

  if (!ros::ok()){
    ros::shutdown();
  }

  return EXIT_SUCCESS;
}
