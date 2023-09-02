#ifndef VIHSO_GLOBAL_H_
#define VIHSO_GLOBAL_H_

#include <list>
#include <vector>
#include <string>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#include <Eigen/Core>
#include <opencv4/opencv2/opencv.hpp>
#include <sophus/se3.h>
#include <boost/shared_ptr.hpp>
#include <Eigen/StdVector>

#include "vihso/vikit/performance_monitor.h"

#define RESET "\033[0m"
#define BLACK "\033[30m" /* Black */
#define RED "\033[31m" /* Red */
#define GREEN "\033[32m" /* Green */
#define YELLOW "\033[33m" /* Yellow */
#define BLUE "\033[34m" /* Blue */
#define MAGENTA "\033[35m" /* Magenta */
#define CYAN "\033[36m" /* Cyan */
#define WHITE "\033[37m" /* White */
#define BOLDBLACK "\033[1m\033[30m" /* Bold Black */
#define BOLDRED "\033[1m\033[31m" /* Bold Red */
#define BOLDGREEN "\033[1m\033[32m" /* Bold Green */
#define BOLDYELLOW "\033[1m\033[33m" /* Bold Yellow */
#define BOLDBLUE "\033[1m\033[34m" /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m" /* Bold Magenta */
#define BOLDCYAN "\033[1m\033[36m" /* Bold Cyan */
#define BOLDWHITE "\033[1m\033[37m" /* Bold White */

#ifndef RPG_SVO_VIKIT_IS_VECTOR_SPECIALIZED
#define RPG_SVO_VIKIT_IS_VECTOR_SPECIALIZED
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector3d)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector2d)
#endif

  #define VIHSO_INFO_STREAM(x) std::cerr<<"\033[0;0m[INFO] "<<x<<"\033[0;0m"<<std::endl;
  #ifdef VIHSO_DEBUG_OUTPUT
    #define VIHSO_DEBUG_STREAM(x) VIHSO_INFO_STREAM(x)
  #else
    #define VIHSO_DEBUG_STREAM(x)
  #endif
  #define VIHSO_WARN_STREAM(x) std::cerr<<"\033[0;33m[WARN] "<<x<<"\033[0;0m"<<std::endl;
  #define VIHSO_ERROR_STREAM(x) std::cerr<<"\033[1;31m[ERROR] "<<x<<"\033[0;0m"<<std::endl;
  #include <chrono>
  #define VIHSO_WARN_STREAM_THROTTLE(rate, x) \
    do { \
      static double __log_stream_throttle__last_hit__ = 0.0; \
      std::chrono::time_point<std::chrono::system_clock> __log_stream_throttle__now__ = \
      std::chrono::system_clock::now(); \
      if (__log_stream_throttle__last_hit__ + rate <= \
          std::chrono::duration_cast<std::chrono::seconds>( \
          __log_stream_throttle__now__.time_since_epoch()).count()) { \
        __log_stream_throttle__last_hit__ = \
        std::chrono::duration_cast<std::chrono::seconds>( \
        __log_stream_throttle__now__.time_since_epoch()).count(); \
        VIHSO_WARN_STREAM(x); \
      } \
    } while(0)

namespace vihso
{
  using namespace Eigen;
  using namespace Sophus;

  const double EPS = 0.0000000001;
  const double PI = 3.14159265;

#ifdef VIHSO_TRACE
  extern hso::PerformanceMonitor* g_permon;
  #define VIHSO_LOG(value) g_permon->log(std::string((#value)),(value))
  #define VIHSO_LOG2(value1, value2) VIHSO_LOG(value1); VIHSO_LOG(value2)
  #define VIHSO_LOG3(value1, value2, value3) VIHSO_LOG2(value1, value2); VIHSO_LOG(value3)
  #define VIHSO_LOG4(value1, value2, value3, value4) VIHSO_LOG2(value1, value2); VIHSO_LOG2(value3, value4)
  #define VIHSO_START_TIMER(name) g_permon->startTimer((name))
  #define VIHSO_STOP_TIMER(name) g_permon->stopTimer((name))
#else
  #define VIHSO_LOG(v)
  #define VIHSO_LOG2(v1, v2)
  #define VIHSO_LOG3(v1, v2, v3)
  #define VIHSO_LOG4(v1, v2, v3, v4)
  #define VIHSO_START_TIMER(name)
  #define VIHSO_STOP_TIMER(name)
#endif

  
  class Frame;
  typedef boost::shared_ptr<Frame> FramePtr;


}

#endif // VIHSO_GLOBAL_H_
