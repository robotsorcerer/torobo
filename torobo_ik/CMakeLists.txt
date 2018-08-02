cmake_minimum_required(VERSION 2.8.3)
project(torobo_ik)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_CXX_EXTENSIONS off)
#add_compile_options(-Wall)
# -Wextra -Wpedantic)

find_package(catkin REQUIRED
  COMPONENTS
    trac_ik_lib
    std_msgs
    geometry_msgs
    message_generation
    rospy_tutorials
    roslib
    kdl_parser
    dr_kdl
)

find_package(PkgConfig)
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
include_directories( ${EIGEN3_INCLUDE_DIRS} )

add_definitions(${EIGEN_DEFINITIONS})

find_package(Boost REQUIRED COMPONENTS date_time filesystem system)
find_package(orocos_kdl REQUIRED)

add_message_files(FILES Numpy64.msg)
add_service_files(DIRECTORY srv
                  FILES Joints.srv  SolveDiffIK.srv)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS
    message_runtime
    trac_ik_lib
    std_msgs
    geometry_msgs
    roslib
  DEPENDS
    Boost
    orocos_kdl
)

message(STATUS "\n\nExported Target:  ${${PROJECT_NAME}_EXPORTED_TARGETS}")

#add_dependencies(ik_sub ${${PROJECT_NAME}_EXPORTED_TARGETS})
#add_custom_target(OUTPUT SolveDiffIK.h
#                  DEPENDS ${${PROJECT_NAME}_EXPORTED_TARGETS})
#add_custom_target(OUTPUT SolveDiffIK.h  DEPENDS ik_sub)

include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${orocos_kdl_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)


macro(add_exec name)
  add_executable(${name} src/${name}.cpp)

  target_link_libraries(${name} ${catkin_LIBRARIES} ${Boost_LIBRARIES}
                        ${EIGEN3_LIBRARIES} ${orocos_kdl_LIBRARIES}
                        ${Boost_FILESYSTEM_LIBRARY}
                        -lm -ldl)
	install(TARGETS ${name}
	  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
	  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
	  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
	)
endmacro()

add_executable(ik_sub src/ik_sub.cpp )
target_link_libraries(ik_sub ${catkin_LIBRARIES} ${Boost_LIBRARIES}
											${EIGEN3_LIBRARIES} ${orocos_kdl_LIBRARIES}
											${Boost_FILESYSTEM_LIBRARY}
											-lm -ldl)
# add_exec(torobo_ik)

add_dependencies(ik_sub  ${catkin_EXPORTED_TARGETS})

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
