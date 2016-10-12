cmake_minimum_required(VERSION 2.8)
project(test_rs)
add_compile_options(-std=c++11 -fPIC -lusb-1.0 -lpthread -lboost_iostreams -lboost_filesystem -lboost_system)
include_directories(include)
include_directories(/home/aerolabio/librealsense/examples/myexample/third_party)

#########################################################
# FIND Threads libusb-1.0
#########################################################
find_package(Threads REQUIRED)

find_package(PkgConfig)
pkg_check_modules(PC_LIBUSB REQUIRED libusb-1.0)
include_directories( ${LIBUSB_1_INCLUDE_DIRS})

#########################################################


## System dependencies are found with CMake's conventions

#########################################################
# FIND BOOST
#########################################################
find_package(Boost REQUIRED system thread)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
add_definitions(${Boost_DEFINITIONS})
if(NOT Boost_FOUND)
    message(ERROR " Boost not found!")
endif(NOT Boost_FOUND)
#########################################################
# FIND OPENGL
#########################################################
find_package(OpenGL REQUIRED)
include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})
if(NOT OPENGL_FOUND)
    message(ERROR " OPENGL not found!")
endif(NOT OPENGL_FOUND)
#########################################################
# FIND GLUT
#########################################################
find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIRS} )
#########################################################
# FIND GLFW
#########################################################
find_package(PkgConfig REQUIRED)
 pkg_search_module(GLFW REQUIRED glfw3)
include_directories(${GLFW_INCLUDE_DIRS})
#########################################################
# FIND OpenCV
#########################################################
find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})
#########################################################
# FIND realsense
#########################################################
include_directories("/home/aerolabio/librealsense/include")
link_directories("/usr/local/lib")
set(LIBS ${LIBS} realsense)

link_directories( ${realsense_LIBRARY})
include_directories( ${realsense_INCLUDE_DIR})

#########################################################


###########
## Build ##
###########

set(SRC_LIST "src/main.cpp")
aux_source_directory(. SRC_LIST)

## Declare a C++ executable
add_executable(${PROJECT_NAME} src/main.cpp
                    src/utils.cpp
              )

## Specify libraries to link a library or executable target against
target_link_libraries( ${PROJECT_NAME}
${LIBS} ${GLFW_LIBRARIES} ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${GLFW_LIBRARIES} ${Boost_LIBRARIES})
