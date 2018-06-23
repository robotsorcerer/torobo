cmake_minimum_required(VERSION 2.8.3)
project(trac_ik_torobo)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED
  COMPONENTS
    trac_ik_lib
    std_msgs
    message_generation
    rospy_tutorials
)

find_package(PkgConfig)
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
include_directories( ${EIGEN3_INCLUDE_DIRS} )

add_definitions(${EIGEN_DEFINITIONS})

find_package(Boost REQUIRED COMPONENTS date_time)
find_package(orocos_kdl REQUIRED)


add_service_files(FILES Joints.srv)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    trac_ik_lib std_msgs message_runtime rospy_tutorials
  DEPENDS
    Boost
    orocos_kdl
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${orocos_kdl_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

macro(add_exec name)
  add_executable(${name} src/${name}.cpp)
  target_link_libraries(${name} ${catkin_LIBRARIES} ${Boost_LIBRARIES}
	 											${orocos_kdl_LIBRARIES} cnpy -lm -ldl)
	install(TARGETS ${name}
	  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
	  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
	  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
	)
endmacro()

add_exec(ik_torobo)
add_exec(teach_jnt_to_cart)
add_exec(ik_sub)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
