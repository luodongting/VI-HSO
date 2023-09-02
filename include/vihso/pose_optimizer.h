#ifndef SVO_POSE_OPTIMIZER_H_
#define SVO_POSE_OPTIMIZER_H_
#include <vihso/global.h>
#include <vihso/feature.h>
namespace vihso {
using namespace Eigen;
using namespace Sophus;
using namespace std;
typedef Matrix<double,6,6> Matrix6d;
typedef Matrix<double,2,6> Matrix26d;
typedef Matrix<double,6,1> Vector6d;
namespace pose_optimizer {
void optimizeGaussNewton(
    const double reproj_thresh,
    const size_t n_iter,
    const bool verbose,
    FramePtr& frame,
    double& estimated_scale,
    double& error_init,
    double& error_final,
    size_t& num_obs);
void optimizeLevenbergMarquardt2nd(
    const double reproj_thresh, const size_t n_iter, const bool verbose,
    FramePtr& frame, double& estimated_scale, double& error_init, double& error_final,
    size_t& num_obs);
void optimizeLevenbergMarquardt3rd(
    const double reproj_thresh, const size_t n_iter, const bool verbose,
    FramePtr& frame, double& estimated_scale, double& error_init, double& error_final,
    size_t& num_obs);
void optimizeLevenbergMarquardtMagnitude(
    const double reproj_thresh, const size_t n_iter, const bool verbose,
    FramePtr& frame, double& estimated_scale, double& error_init, double& error_final,
    size_t& num_obs);
    static int residual_buffer[10000]={0};
} 
} 
#endif 
