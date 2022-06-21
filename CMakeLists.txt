set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cmake_minimum_required(VERSION 3.5)
project(lio_connector)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")

find_package(catkin REQUIRED COMPONENTS
    tf
    roscpp
    rospy
    cv_bridge
    # pcl library
    pcl_conversions
    # msgs
    std_msgs
    sensor_msgs
    geometry_msgs
    nav_msgs
    message_generation
    visualization_msgs
)

find_package(GTSAM REQUIRED QUIET)
find_package(PCL REQUIRED QUIET)
find_package(Eigen3 REQUIRED)
find_package(libnabo REQUIRED)


catkin_package(
   INCLUDE_DIRS include
   DEPENDS PCL GTSAM
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  src/third_parties
)

# link directories
link_directories(
  include
  ${GTSAM_LIBRARY_DIRS}
  ${PCL_LIBRARY_DIRS}
)

#for fushion
# Maximum Clique Solver
add_library(fast_max-clique_finder
        src/third_parties/fast_max-clique_finder/src/findClique.h
        src/third_parties/fast_max-clique_finder/src/graphIO.h
        src/third_parties/fast_max-clique_finder/src/findClique.cpp
        src/third_parties/fast_max-clique_finder/src/findCliqueHeu.cpp
        src/third_parties/fast_max-clique_finder/src/utils.cpp
        src/third_parties/fast_max-clique_finder/src/graphIO.cpp
        )
target_compile_options(fast_max-clique_finder PRIVATE -w)

add_library(scan_context
        src/third_parties/scanContext/scanContext.h
        src/third_parties/scanContext/scanContext.cpp
        )


## Declare a C++ library
add_executable(${PROJECT_NAME}_main src/main.cpp)
add_dependencies(${PROJECT_NAME}_main ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_main ${catkin_LIBRARIES} ${PCL_LIBRARIES}
        gtsam libnabo::nabo fast_max-clique_finder scan_context)
