#ifndef VIHSO_FEATURE_ALIGNMENT_H_
#define VIHSO_FEATURE_ALIGNMENT_H_

#include <vihso/global.h>

namespace vihso {

class Point;
struct Feature;

namespace feature_alignment {

bool align1D(
    const cv::Mat& cur_img,
    const Vector2f& dir,                 
    float* ref_patch_with_border,
    float* ref_patch,
    const int n_iter,
    Vector2d& cur_px_estimate,
    double& h_inv,
    float* cur_patch = NULL);

bool align2D(
    const cv::Mat& cur_img,
    float* ref_patch_with_border,
    float* ref_patch,
    const int n_iter,
    Vector2d& cur_px_estimate,
    bool no_simd = true,
    float* cur_patch = NULL);

} // namespace feature_alignment
} // namespace vihso

#endif // VIHSO_FEATURE_ALIGNMENT_H_
