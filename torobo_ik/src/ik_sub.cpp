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
#include <trac_ik_torobo/ik_solver.h>
#include <trac_ik_torobo/Numpy64.h>
#include <boost/filesystem.hpp>

// ik headers
#include <std_msgs/Float64.h>
#include <sensor_msgs/JointState.h>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <trac_ik_torobo/SolveDiffIK.h>

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
        //cout << "     Source file: " << srcFile.c_str() << endl;
        boost::filesystem::path dstFile = dstPath / fn;
        //cout << "Destination file: " << dstFile.c_str() << endl;
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
    std::string save_path;
    std::ofstream cartPosFile;
    bool running, updateJoints;
    int rows, cols, counter;
    std::mutex mutex;
    const std::string urdf;
    KDL::Tree kdl_tree;
    bool disp, saved, save_to_file;
    std::vector<KDL::Frame> CartPosList;
    std::vector<Eigen::Vector3d> CartVelList;
    std::vector<double> TimeIdx;

    // IK Solver Members
    // friend class IKVelocitySolver; //friend class forward declaration
    std::unique_ptr<KDL::Chain> chain;
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver;
    std::unique_ptr<KDL::ChainIkSolverVel_pinv> vik_solver;
    /// Inverse differential kinematics solver.
    std::unique_ptr<KDL::ChainIkSolverVel_pinv> solver_;
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
      // save_path("/home/olalekan/Documents/LyapunovLearner/scripts/data/cart_pos.csv"),
      cartPosFile(save_path), running(false), updateJoints(false), rows(10001), cols(7), counter(0)
      {
          nh_.getParam("/trac_ik_torobo/chain_start", base_link);
          nh_.getParam("/trac_ik_torobo/chain_end", tip_link);
          nh_.getParam("/trac_ik_torobo/save_to_file", save_to_file);
          nh_.getParam("/trac_ik_torobo/disp", disp);
          nh_.getParam("/trac_ik_torobo/saved", saved);

          getROSPackagePath("lyapunovlearner", data_dir);
          data_dir  = data_dir / "scripts" / "data" / "cart_pos.csv";
          save_path = data_dir.c_str(); 
          ROS_INFO_STREAM("save_path " << save_path);

          get_kdl_tree();

          fk_solver      = std::make_unique<KDL::ChainFkSolverPos_recursive>(*this->chain.get()); // Forward kin. solver
          vik_solver     = std::make_unique<KDL::ChainIkSolverVel_pinv>(*this->chain.get());      // PseudoInverse vel solver
          sub               = nh_.subscribe("/torobo/teach_joints", 10, &Converter::joints_cb, this);
          diff_ik_server_   = nh_.advertiseService("/torobo/solve_diff_ik", &Converter::onSolveRequest, this);
          ik_pub_           = nh_.advertise<sensor_msgs::JointState>("/torobo/ik_results", 100, false);
          ik_client         = nh_.serviceClient<trac_ik_torobo::SolveDiffIK>("/torobo/solve_diff_ik");
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
        // threads.push_back(std::thread(&Converter::getIK, this));
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

    bool onSolveRequest(trac_ik_torobo::SolveDiffIK::Request & req, trac_ik_torobo::SolveDiffIK::Response & res)
    {
    		KDL::JntArray q_in(cols);
    		KDL::JntArray q_out(cols);
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

    		vik_solver->CartToJnt(q_in, desired_vel, q_out);
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
        {
          ROS_INFO_STREAM("RawJoints \n" << RawJoints.block(0, 0, cols, cols));
        }

        std::lock_guard<std::mutex> lock(mutex);
        this->RawJoints = RawJoints;
        updateJoints = true;
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

            calculate_pos(saved_joints);
            calculate_vel();
            if(save_to_file)
            {
              if(!saved)
              {
                save_cart_data();
              }
            }
              updateJoints = false;
          }
      }
    }

    void calculate_pos(Eigen::MatrixXd const& saved_joints)
    {
          KDL::Chain *chain = this->chain.get(); // returns a pointer to the stored object

          // Set up KDL IK
          KDL::ChainFkSolverPos_recursive fk_solver  = KDL::ChainFkSolverPos_recursive(*chain); // Forward kin. solver
          KDL::ChainIkSolverVel_pinv vik_solver      = KDL::ChainIkSolverVel_pinv(*chain);      // PseudoInverse vel solver

          double elapsed = 0;
          KDL::Frame cartpos;
          bool kinematics_status = false;
          double timeout = 0.005;
          unsigned int num_jts = chain->getNrOfJoints();

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
      saved = true;
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

    // deprecated
    void getIK()
    {
      trac_ik_torobo::SolveDiffIK srv;

      for(int i=0; i < 7; ++i)
        srv.request.q_in.push_back(i * std::pow(0.1, 2));

      if (ik_client.call(srv))
      {
        for(int i=0; i < srv.response.q_out.size(); ++i)
          ROS_INFO_STREAM(srv.response.q_out[i]);
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
