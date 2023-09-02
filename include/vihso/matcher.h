#ifndef VIHSO_MATCHER_H_
#define VIHSO_MATCHER_H_
#include <vihso/global.h>
namespace hso {
  class AbstractCamera;
  namespace patch_score {
  template<int HALF_PATCH_SIZE> class ZMSSD;
  }
}
namespace vihso {
class Point;
class Frame;
class Feature;
class Reprojector;
struct Seed;
namespace warp {
void getWarpMatrixAffine(
    const hso::AbstractCamera& cam_ref,
    const hso::AbstractCamera& cam_cur,
    const Vector2d& px_ref,
    const Vector3d& f_ref,
    const double depth_ref,
    const SE3& T_cur_ref,
    const int level_ref,
    Matrix2d& A_cur_ref);
int getBestSearchLevel(
    const Matrix2d& A_cur_ref,
    const int max_level);
void warpAffine(
    const Matrix2d& A_cur_ref,
    const cv::Mat& img_ref,
    const Vector2d& px_ref,
    const int level_ref,
    const int level_cur,
    const int halfpatch_size,
    uint8_t* patch);
void warpAffine(
    const Matrix2d& A_cur_ref,
    const cv::Mat& img_ref,
    const Vector2d& px_ref,
    const int level_ref,
    const int level_cur,
    const int halfpatch_size,
    float* patch);
void createPatch(
    float* patch,
    const Vector2d& px_scaled,
    const Frame* cur_frame,
    const int halfpatch_size,
    const int level);
void convertPatchFloat2Int(
    float* patch_f,
    uint8_t* patch_i,
    const int patch_size);
void createPatchFromPatchWithBorder(
    uint8_t* patch,
    uint8_t* patch_with_border,
    const int patch_size);
void createPatchFromPatchWithBorder(
    float* patch,
    float* patch_with_border,
    const int patch_size);
} 
class Matcher
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static const int halfpatch_size_ = 4;
  static const int patch_size_ = 8;
  typedef hso::patch_score::ZMSSD<halfpatch_size_> PatchScore;
  struct Options
  {
    bool align_1d;              
    int align_max_iter;         
    double max_epi_length_optim;
    size_t max_epi_search_steps;
    bool subpix_refinement;     
    bool epi_search_edgelet_filtering;
    double epi_search_edgelet_max_angle;
    Options() :
      align_1d(false),
      align_max_iter(10),
      max_epi_length_optim(2.0),
      max_epi_search_steps(100),
      subpix_refinement(true),
      epi_search_edgelet_filtering(true),
      epi_search_edgelet_max_angle(0.4)
    {}
  } options_;
  uint8_t patch_[patch_size_*patch_size_] __attribute__ ((aligned (16)));
  uint8_t patch_with_border_[(patch_size_+2)*(patch_size_+2)] __attribute__ ((aligned (16)));
  float patch_f_[patch_size_*patch_size_] __attribute__ ((aligned (16))); 
  float patch_with_border_f_[(patch_size_+2)*(patch_size_+2)] __attribute__ ((aligned (16))); 
  Matrix2d A_cur_ref_;          
  Vector2d epi_dir_;            
  double epi_length_;           
  double h_inv_;                
  int search_level_;            
  bool reject_;
  Feature* ref_ftr_;            
  Vector2d px_cur_;             
  Matcher() = default;
  ~Matcher() = default;
  bool findMatchDirect(Point& pt, Frame& frame, Vector2d& px_cur);
  bool findMatchSeed(const Seed& seed, Frame& frame, Vector2d& px_cur, float ncc_thresh = 0.6);
  bool findEpipolarMatchDirect(
      const Frame& ref_frame,
      const Frame& cur_frame,
      const Feature& ref_ftr,
      const double d_estimate,
      const double d_min,
      const double d_max,
      double& depth,
      Vector2i& epl_start,
      Vector2i& epl_end,
      bool homoIsValid = false,
      Matrix3d homography = Eigen::Matrix3d::Identity());
  void createPatchFromPatchWithBorder();
  int doLineStereo(
    Frame& ref_frame, Frame& cur_frame, const Feature& ref_ftr,
    const double min_idepth, const double prior_idepth, const double max_idepth,
    double& result_depth, Vector2i& EPL_start, Vector2i& EPL_end);
  bool findEpipolarMatchPrevious(
      Frame& ref_frame,
      Frame& cur_frame,
      const Feature& ref_ftr,
      const double d_estimate,
      const double d_min,
      const double d_max,
      double& depth);
  bool checkNCC(float* patch1, float* patch2, float thresh);
  bool checkNormal(const Frame& frame, int level, Vector2d pxLevel, Vector2d normal, float thresh=0.866);
  bool KLTLimited2D(const cv::Mat& targetImg,
  float* hostPatchWithBorder,
  float* hostPatch,
  const int n_iter,
  Vector2d& targetPxEstimate,
  float* targetPatch = NULL,
  bool debugPrint = false);
  bool KLTLimited1D(const cv::Mat& targetImg,
  float* hostPatchWithBorder,
  float* hostPatch,
  const int n_iter,
  Vector2d& targetPxEstimate,
  const Vector2d& direct,
  float* targetPatch = NULL,
  bool debugPrint = false);
};
namespace patch_utils{
inline void patchToMat(
  const uint8_t* const patch_data,
  const size_t patch_width,
  cv::Mat* img)
{
  *img = cv::Mat(patch_width, patch_width, CV_8UC1);
  std::memcpy(img->data, patch_data, patch_width*patch_width);
} 
} 
} 
#endif 
