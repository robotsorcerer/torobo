#include <chrono>
#include <thread>
#include <unistd.h>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <ros/ros.h>
#include <ros/package.h>  /*roslib*/
#include <ros/spinner.h>
#include <boost/date_time.hpp>
#include <trac_ik/trac_ik.hpp>
#include "dr_kdl/dr_kdl.hpp"
#include <kdl_parser/kdl_parser.hpp>

#include <mutex>
#include <Eigen/Core>
#include <boost/cerrno.hpp>
#include <boost/filesystem.hpp>

// ik headers
#include <std_msgs/Float64.h>
#include <kdl/chainiksolver.hpp>
#include <sensor_msgs/JointState.h>
#include <kdl/chainiksolverpos_nr.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/chainfksolvervel_recursive.hpp>

#include <torobo_ik/Numpy64.h>
#include <torobo_ik/ik_solver.h>
#include <torobo_ik/SolveDiffIK.h>

using namespace boost::filesystem;
using namespace pfn;

bool getROSPackagePath(const std::string pkgName, boost::filesystem::path & pkgPath)
{
    pkgPath = ros::package::getPath(pkgName);
    if (pkgPath.empty())
    {
        printf("Could not find package '%s' ", pkgName.c_str());
        return false;
    }
    else
    {
        printf("%s package found here: %s", pkgName.c_str(), pkgPath.c_str());
        return true;
    }
}

static bool copyDirectory(const boost::filesystem::path srcPath, const boost::filesystem::path dstPath)
{
    boost::filesystem::create_directory(dstPath);

    for (boost::filesystem::directory_iterator end, dir(srcPath.c_str()); dir != end; ++dir)
    {
        boost::filesystem::path fn = (*dir).path().filename();
        boost::filesystem::path srcFile = (*dir).path();
        boost::filesystem::path dstFile = dstPath / fn;
        boost::filesystem::copy_file(srcFile, dstFile);
    }
    return true;
}

class Converter{
  private:
    ros::NodeHandle nh_;
    ros::Subscriber sub;
    std::vector<std::thread> threads;
    std::vector<std::vector<double>> jointsarray;
    Eigen::MatrixXd RawJoints;
    unsigned long const hardware_concurrency;
    ros::AsyncSpinner spinner;
    std::string save_path, data_file;
    std::ofstream cartPosFile;
    bool running, updateJoints;
    int rows, cols, counter;
    std::mutex mutex;
    const std::string urdf;
    KDL::Tree kdl_tree;
    bool disp, saved, save_to_file;
    std::vector<KDL::Frame> CartPosList;
    std::vector<KDL::FrameVel> CartVelList;
    std::vector<double> TimeIdx;

    // IK Solver Members
    std::unique_ptr<KDL::Chain> chain;
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver;
    std::unique_ptr<KDL::ChainIkSolverVel_pinv> vik_solver;
    std::unique_ptr<KDL::ChainIkSolverPos_NR> pik_solver;
    /// Service server.
    ros::ServiceServer diff_ik_server_;
    ros::Publisher ik_pub_;
    ros::ServiceClient ik_client;
    std::string base_link, tip_link;
    boost::filesystem::path data_dir;
    std::stringstream ss;

  public:
      Converter()
      :hardware_concurrency(std::thread::hardware_concurrency()), spinner(hardware_concurrency/6),
      cartPosFile(save_path, std::ios::binary), running(false), updateJoints(false), cols(14), counter(0)
      {
          nh_.getParam("/torobo_ik/disp", disp);
          nh_.getParam("/torobo_ik/saved", saved);
          nh_.getParam("/torobo_ik/chain_end", tip_link);
          nh_.getParam("/torobo_ik/data_file", data_file);
          nh_.getParam("/torobo_ik/chain_start", base_link);
          nh_.getParam("/torobo_ik/save_to_file", save_to_file);

          getROSPackagePath("lyapunovlearner", data_dir);
          data_dir  = data_dir / "scripts" / "data" / data_file;
          save_path = data_dir.c_str();
          ROS_INFO_STREAM("\n\nsave_path " << save_path);

          get_kdl_tree();

          fk_solver         = std::make_unique<KDL::ChainFkSolverPos_recursive>(*this->chain.get()); // Forward kin. solver
          vik_solver        = std::make_unique<KDL::ChainIkSolverVel_pinv>(*this->chain.get());      // PseudoInverse vel solver
          pik_solver        = std::make_unique<KDL::ChainIkSolverPos_NR>(*this->chain.get(), *fk_solver, *vik_solver, 200, 1e-6);       // PseudoInverse vel solver
          sub               = nh_.subscribe("/torobo_ik/teach_joints", 10, &Converter::joints_cb, this);
          diff_ik_server_   = nh_.advertiseService("/torobo_ik/solve_diff_ik", &Converter::onSolveRequest, this);
          ik_pub_           = nh_.advertise<sensor_msgs::JointState>("/torobo_ik/ik_results", 100, false);
          ik_client         = nh_.serviceClient<torobo_ik::SolveDiffIK>("/torobo_ik/solve_diff_ik");
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
      void begin()
      {
        if(spinner.canStart())
        {
          running = true;
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
        running = false;
      }

      KDL::Tree get_kdl_tree()
      {
        KDL::Chain chain;
        KDL::Tree kdl_tree = dr::KdlTree::fromParameter("/robot_description");
        kdl_tree.getChain(base_link, tip_link, chain);
        this->chain = std::make_unique<KDL::Chain>(chain);
        this->kdl_tree = kdl_tree;
        return kdl_tree;
      }

    bool onSolveRequest(torobo_ik::SolveDiffIK::Request & req, torobo_ik::SolveDiffIK::Response & res)
    {
    		KDL::JntArray q_in(static_cast<int>(cols/2));
    		KDL::JntArray q_out(static_cast<int>(cols/2));
    		KDL::Vector rot, trans;

    		trans.x(req.desired_pos.linear.x);
    		trans.y(req.desired_pos.linear.y);
    		trans.z(req.desired_pos.linear.z);
    		rot.x(req.desired_pos.angular.x);
    		rot.y(req.desired_pos.angular.y);
    		rot.z(req.desired_pos.angular.z);

    		KDL::Twist desired_pos{trans, rot};
    		for (size_t i = 0; i < req.q_in.size(); i++) {
    			q_in(i) = req.q_in.at(i);
    		}

        KDL::Vector vec{req.desired_pos.linear.x, req.desired_pos.linear.y, req.desired_pos.linear.z} ;
        KDL::Frame dest_frame(vec);
    		vik_solver->CartToJnt(q_in, desired_pos, q_out);
        pik_solver->CartToJnt(q_in, dest_frame, q_out);
    		res.q_out.resize(q_out.rows());
    		for (size_t i = 0; i < res.q_out.size(); i++) {
    			res.q_out.at(i) = q_out(i);
    		}

    		sensor_msgs::JointState result_state;
    		result_state.header.stamp = ros::Time::now();
    		result_state.velocity = res.q_out;
    		ik_pub_.publish(result_state);

    		return !res.q_out.empty();
     }

    void joints_cb(const torobo_ik::Numpy64::ConstPtr& np_msg)
    {
        Eigen::MatrixXd RawJoints;
        // raw joints contain the indices of time as well as the seven joint angles
        // per time during the teaching motion
        rows = np_msg->data.size()/this->cols;
        this->RawJoints.resize(rows, this->cols);
        RawJoints.resize(rows, this->cols);

        for (int i=0; i < rows; ++i)
        {
          for (auto j =0; j < this->cols; ++j)  // get only joint positions
          {
            RawJoints(i, j) = np_msg->data[i*this->cols+j];
          }
          TimeIdx.push_back(np_msg->data[i*this->cols]);
        }

        std::lock_guard<std::mutex> lock(mutex);
        this->RawJoints = RawJoints;
        updateJoints = true;
        ++counter;

        if (disp){
          ROS_INFO_STREAM("RawJoints size: " << RawJoints.rows() << "," << RawJoints.cols());
          ROS_INFO_STREAM("Raw Joints [slice(10, 15), slice(7, 14)]: \n" << RawJoints.block(10, 6, 15, 13));
        }
    }

    void convert()
    {
      Eigen::MatrixXd saved_joints;

      for(; running && ros::ok() ;)
      {
          if (updateJoints)
          {
            std::lock_guard<std::mutex> lock(mutex);
            saved_joints = this->RawJoints;
            int cols = saved_joints.cols();
            int rows = saved_joints.rows();

            auto temp = saved_joints.block(0, 0, rows, 7);
            // ROS_INFO_STREAM("temp: " << temp.rows() << ", " << temp.cols());

            calculate_pos(temp); // cause position is first 7 cols
            calculate_vel(saved_joints);
            if(save_to_file)
            {
              if(!saved)
              {
                ROS_INFO("saving calculated cart data to file");
                save_cart_data();
                ROS_INFO("Finished saving calculated cart data to [%s]", save_path);
              }
            }
              updateJoints = false;
          }
      }
    }

    void calculate_pos(Eigen::MatrixXd const& saved_pos)
    {
      KDL::Chain *chain = this->chain.get(); // returns a pointer to the stored object
      // Set up KDL IK
      KDL::ChainFkSolverPos_recursive fk_solver  = KDL::ChainFkSolverPos_recursive(*chain); // Forward kin. solver

      double elapsed = 0;
      KDL::Frame cartpos;
      bool kinematics_status = false;
      double timeout = 0.005;
      unsigned int num_jts = chain->getNrOfJoints();

      boost::posix_time::ptime start_time;
      boost::posix_time::time_duration diff;

      for (auto i=0; i < saved_pos.rows(); i++)
      {
          KDL::JntArray q(num_jts);
          for (int j=0; j<saved_pos.cols(); j++)
          {
            q(j)=saved_pos(i, j);
          }

          start_time = boost::posix_time::microsec_clock::local_time();

          kinematics_status = fk_solver.JntToCart(q, cartpos);
          diff = boost::posix_time::microsec_clock::local_time() - start_time;
          elapsed = diff.total_nanoseconds() / 1e9;

          this->CartPosList.push_back(cartpos);
      }
    }

    void calculate_vel(Eigen::MatrixXd const& saved_vels)
    {

        KDL::Chain *chain = this->chain.get(); // returns a pointer to the stored object
        // KDL::ChainIkSolverVel_pinv vik_solver      = KDL::ChainIkSolverVel_pinv(*chain);      // PseudoInverse vel solver
        KDL::ChainFkSolverVel_recursive vfk_solver = KDL::ChainFkSolverVel_recursive (*chain);      // PseudoInverse vel solver

        double elapsed = 0;
        KDL::FrameVel cartvel;
        bool kinematics_status = false;
        double timeout = 0.005;
        unsigned int num_jts = chain->getNrOfJoints();
        // ROS_INFO("num_jts: %d", num_jts);

        boost::posix_time::ptime start_time;
        boost::posix_time::time_duration diff;

        // ROS_INFO("vel rows: %d vel cols: %d ", saved_vels.rows(), saved_vels.cols());

        for (auto i=0; i < saved_vels.rows(); i++)
        {

            KDL::JntArray q(num_jts), qdot(num_jts);
            for (int j=0; j< num_jts; j++)
            {
              q(j)=saved_vels(i, j);
            }

            for (int j=num_jts; j< saved_vels.cols(); j++)
            {
              qdot(j-num_jts)=saved_vels(i, j);
            }

            KDL::JntArrayVel qvel = KDL::JntArrayVel(q, qdot); //(num_jts);

            start_time = boost::posix_time::microsec_clock::local_time();

            kinematics_status = vfk_solver.JntToCart(qvel, cartvel);
            diff = boost::posix_time::microsec_clock::local_time() - start_time;
            elapsed = diff.total_nanoseconds() / 1e9;

            this->CartVelList.push_back(cartvel);
        }
    }

    void save_cart_data()
    {
      // check_file_exists();
      std::vector<KDL::Frame> CartPosList = this->CartPosList;
      std::vector<KDL::FrameVel> CartVelList = this->CartVelList;

      auto itPos = CartPosList.begin();
      auto itVel = CartVelList.begin();

      std::ofstream cartPosFile(save_path, std::ios::binary);

      while(itPos != CartPosList.end()+1 && itVel != CartVelList.end()+1)
      {
          KDL::Twist twist = itVel->GetTwist();
          if(disp)
          {
            ROS_INFO("[x:, y:, z, xd:, yd:, zd:]: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f",
                      itPos->p.x(), itPos->p.y(), itPos->p.z(), twist[0], twist[1], twist[2]  );
          }
          cartPosFile << itPos->p.x() << "," << itPos->p.y() << "," << itPos->p.z() << ","
                      << twist[0] << "," << twist[1] << "," << twist[2] << "\n";

           ++itPos;
           ++itVel;
      }
      saved = true;
      ROS_INFO_STREAM("save_path: " << save_path);
    }

    // ripped off: https://www.boost.org/doc/libs/1_39_0/libs/filesystem/test/operations_test.cpp
    void create_file( const boost::filesystem::path & ph, const std::string & contents )
    {
      std::ofstream f( ph.c_str() );
      if ( !contents.empty() ) f << contents;
    }

    bool check_file_exists()
    {
        path p(save_path);
        if(boost::filesystem::is_regular_file(p))
        {
          boost::filesystem::remove(save_path);
          ROS_WARN("File already in directory : %s. Removing it", save_path.c_str());
          return true;
        }
      else
      {
        ROS_INFO("Creating new file in directory: %s", save_path.c_str());
        // boost::filesystem::ofstream(save_path);  // create the empty file
        create_file(save_path, "");
        return false;
      }
    }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "ik_sub");

  Converter converter;
  converter.spawn();

  if (!ros::ok()){
    return EXIT_SUCCESS;
  }

  ros::waitForShutdown();;
}
