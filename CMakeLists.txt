################################################################################
# user build settings

SET(TRACE FALSE)
SET(HAVE_G2O TRUE) 
SET(USE_ROS FALSE) # Set FALSE if you want to build this package without Catkin
SET(DEBUG_OUTPUT FALSE) # Only relevant if build without ROS

################################################################################

SET(PROJECT_NAME hso)

PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED (VERSION 2.8.3)

SET(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo

SET(CMAKE_VERBOSE_MAKEFILE OFF)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")

# Set definitions
IF(TRACE)
  ADD_DEFINITIONS(-DVIHSO_TRACE)
ENDIF()
IF(HAVE_G2O)
  ADD_DEFINITIONS(-DUSE_BUNDLE_ADJUSTMENT) 
ENDIF()
IF(USE_ROS)
  ADD_DEFINITIONS(-DSVO_USE_ROS)
ENDIF()
IF(DEBUG_OUTPUT)
  ADD_DEFINITIONS(-DVIHSO_DEBUG_OUTPUT)
ENDIF()

# Set build flags, set ARM_ARCHITECTURE environment variable on Odroid
# Set build flags. Set IS_ARM on odroid board as environment variable
SET(CMAKE_CXX_FLAGS "-Wall -D_LINUX -D_REENTRANT -march=native -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unknown-pragmas")
IF(DEFINED ENV{ARM_ARCHITECTURE})
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon -march=armv7-a")
ELSE()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmmx -msse -msse -msse2 -msse3 -mssse3")
ENDIF()
IF(CMAKE_COMPILER_IS_GNUCC)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
ELSE()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
ENDIF()
#set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

# Add plain cmake packages 
#set( OpenCV_DIR "/usr/share/opencv4" )
find_package(OpenCV QUIET)
#FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Eigen REQUIRED)
FIND_PACKAGE(Boost REQUIRED COMPONENTS thread system)
FIND_PACKAGE(Pangolin REQUIRED)


# Include dirs
INCLUDE_DIRECTORIES(
  ${PROJECT_SOURCE_DIR}/thirdparty/g2o            # g2o
  ${PROJECT_SOURCE_DIR}/thirdparty/fast/include   # fast
  ${PROJECT_SOURCE_DIR}/thirdparty/Sophus         # sophus
  ${PROJECT_SOURCE_DIR}/include			# hso
  ${Eigen_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
  /usr/include/suitesparse # for cholmod
)

# Set link libraries
LIST(APPEND LINK_LIBS 
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
  ${Pangolin_LIBRARIES}
)



# Set sourcefiles
LIST(APPEND SOURCEFILES
  ${PROJECT_SOURCE_DIR}/src/frame_handler_mono.cpp
  ${PROJECT_SOURCE_DIR}/src/frame_handler_base.cpp
  ${PROJECT_SOURCE_DIR}/src/frame.cpp
  ${PROJECT_SOURCE_DIR}/src/point.cpp
  ${PROJECT_SOURCE_DIR}/src/map.cpp
  ${PROJECT_SOURCE_DIR}/src/pose_optimizer.cpp
  ${PROJECT_SOURCE_DIR}/src/initialization.cpp
  ${PROJECT_SOURCE_DIR}/src/matcher.cpp
  ${PROJECT_SOURCE_DIR}/src/reprojector.cpp
  ${PROJECT_SOURCE_DIR}/src/feature_alignment.cpp
  ${PROJECT_SOURCE_DIR}/src/feature_detection.cpp
  ${PROJECT_SOURCE_DIR}/src/depth_filter.cpp
  ${PROJECT_SOURCE_DIR}/src/config.cpp
  ${PROJECT_SOURCE_DIR}/src/viewer.cpp
  ${PROJECT_SOURCE_DIR}/src/PhotomatricCalibration.cpp
  ${PROJECT_SOURCE_DIR}/src/CoarseTracker.cpp
  ${PROJECT_SOURCE_DIR}/src/camera.cpp
  ${PROJECT_SOURCE_DIR}/src/ImageReader.cpp
  ${PROJECT_SOURCE_DIR}/src/bundle_adjustment.cpp
  ${PROJECT_SOURCE_DIR}/src/SystemNode.cpp
  ${PROJECT_SOURCE_DIR}/src/SettingParameters.cpp
  ${PROJECT_SOURCE_DIR}/src/ImuTypes.cpp
  ${PROJECT_SOURCE_DIR}/src/G2oTypes.cpp
  ${PROJECT_SOURCE_DIR}/src/LocalMapping.cpp

  ${PROJECT_SOURCE_DIR}/src/vikit/robust_cost.cpp
  ${PROJECT_SOURCE_DIR}/src/vikit/math_utils.cpp
  ${PROJECT_SOURCE_DIR}/src/vikit/vision.cpp
  ${PROJECT_SOURCE_DIR}/src/vikit/homography.cpp
  ${PROJECT_SOURCE_DIR}/src/vikit/performance_monitor.cpp)

#set(HSO_LIBS ${PROJECT_SOURCE_DIR}/build/lib/libhso.so)
set(G2O_LIBS ${PROJECT_SOURCE_DIR}/thirdparty/g2o/lib/libg2o.so)
set(FAST_LIBS ${PROJECT_SOURCE_DIR}/thirdparty/fast/build/libfast.so)
set(SOPHUS_LIBS ${PROJECT_SOURCE_DIR}/thirdparty/Sophus/build/libSophus.so)


# Create hso library
ADD_LIBRARY(hso SHARED ${SOURCEFILES})
TARGET_LINK_LIBRARIES(hso 
${LINK_LIBS}
${G2O_LIBS} cholmod cxsparse
${FAST_LIBS}
${SOPHUS_LIBS}
)

################################################################################

#EUROC
ADD_EXECUTABLE(vi_euroc test/vi_euroc.cc)
TARGET_LINK_LIBRARIES(vi_euroc hso)
#TUM
ADD_EXECUTABLE(vi_tum test/vi_tum.cc)
TARGET_LINK_LIBRARIES(vi_tum hso)