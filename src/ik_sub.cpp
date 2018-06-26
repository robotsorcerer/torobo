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
#include <boost/filesystem.hpp>


using namespace boost::filesystem;

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
    std::vector<KDL::Frame> CartPosList;
    std::vector<Eigen::Vector3d> CartVelList;
    std::vector<double> TimeIdx;


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
        // raw joints contain the indices of time as well as the seven joint angles
        // per time during the teaching motion
        this->RawJoints.resize(rows, cols);

        RawJoints.resize(rows, cols);
        for (int i=0; i < rows; ++i)
        {
          for (auto j =0; j < cols; ++j)  // get only joint positions
          {
            RawJoints(i, j) = np_msg->data[i*cols+j+1];
          }
          TimeIdx.push_back(np_msg->data[i*cols]);
        }
        if(disp)
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

          calculate_pos(saved_joints);
          calculate_vel();
          save_cart_data();
          ROS_INFO("Shutting down ros");
          ros::shutdown();
        }
      }

    void calculate_pos(Eigen::MatrixXd const& saved_joints)
    {
          KDL::Chain chain;
          KDL::Tree kdl_tree = get_kdl_tree();
          kdl_tree.getChain("link0", "link7", chain);

          // Set up KDL IK
          KDL::ChainFkSolverPos_recursive fk_solver = KDL::ChainFkSolverPos_recursive(chain); // Forward kin. solver
          //KDL::ChainIkSolverVel_pinv vik_solver     = KDL::ChainIkSolverVel_pinv(chain); // PseudoInverse vel solver
          // KDL::ChainIkSolverPos_NR_JL kdl_solver(chain,ll,ul,fk_solver, vik_solver, 1, eps); // Joint Limit Solver

          double elapsed = 0;
          KDL::Frame cartpos;
          bool kinematics_status = false;
          double timeout = 0.005;
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

              kinematics_status = fk_solver.JntToCart(q, cartpos);
              diff = boost::posix_time::microsec_clock::local_time() - start_time;
              elapsed = diff.total_nanoseconds() / 1e9;

              this->CartPosList.push_back(cartpos);
          }
      }

    void calculate_vel()
    {
        std::vector<KDL::Frame> CartPosList = this->CartPosList;
        std::vector<Eigen::Vector3d> CartVelList;
        double xdot, ydot, zdot;

        int idx = 0;
        Eigen::Vector3d CartVel;
        for(auto it = CartPosList.cbegin(); it != CartPosList.cend()-1; ++it)
        {
            if(TimeIdx[idx]==0)
            {
              xdot = ydot = zdot = 0.0;
            }
            else
            {
              xdot = ((it+1)->p.x() - it->p.x() ) / TimeIdx[idx];
              ydot = ((it+1)->p.y() - it->p.y() ) / TimeIdx[idx];
              zdot = ((it+1)->p.z() - it->p.z() ) / TimeIdx[idx];
            }

            CartVel << xdot, ydot, zdot;
            CartVelList.push_back(CartVel);
            ++idx;
        }
        auto it = CartPosList.cend();
        // append the last element
        CartVelList.back() <<  it->p.x(), it->p.y(), it->p.z();  // should be zero anyway
        this->CartVelList  = CartVelList;
    }

    void save_cart_data()
    {
      // check_file_exists();
      std::vector<KDL::Frame> CartPosList = this->CartPosList;
      std::vector<Eigen::Vector3d> CartVelList = this->CartVelList;

      auto itPos = CartPosList.begin();
      auto itVel = CartVelList.begin();

      while(itPos != CartPosList.end() && itVel != CartVelList.end())
      {
          Eigen::Vector3d vel = *itVel;
          if(disp)
          {
            ROS_INFO("[x, y, z, xd, yd, zd]: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f]", \
                      itPos->p.x(), itPos->p.y(), itPos->p.z(),
                      vel[0], vel[1], vel[2]  );
          }
          cartPosFile << itPos->p.x() << ", " << itPos->p.y() << ", " <<  itPos->p.z() <<
                      ", " << vel[0] << ", " << vel[1] << ", " << vel[2] << "\n";

           ++itPos;
           ++itVel;
      }
    }

    bool check_file_exists()
    {
        path p(save_path);
        if(exists(p))
        {
          if(is_regular_file(p))
          {
            boost::filesystem::remove(save_path);
          }
          return true;
        }
        else
        {
          boost::filesystem::ofstream(save_path);  // create the empty file
          return false;
        }
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
