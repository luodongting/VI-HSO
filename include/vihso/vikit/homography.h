#pragma once
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <Eigen/SVD>
#include "vihso/vikit/math_utils.h"
namespace hso {
using namespace Eigen;
using namespace std;
struct HomographyDecomposition
{
  Vector3d t;
  Matrix3d R;
  double   d;
  Vector3d n;
  Sophus::SE3 T; 
  int score;
};
class Homography
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Homography            (const vector<Vector2d, aligned_allocator<Vector2d> >& _fts1,
                         const vector<Vector2d, aligned_allocator<Vector2d> >& _fts2,
                         double _error_multiplier2,
                         double _thresh_in_px);
  void
  calcFromPlaneParams   (const Vector3d & normal,
                         const Vector3d & point_on_plane);
  void
  calcFromMatches       ();
  size_t
  computeMatchesInliers ();
  bool
  computeSE3fromMatches ();
  bool
  decompose             ();
  void
  findBestDecomposition ();
  double thresh;
  double error_multiplier2;
  const vector<Vector2d, aligned_allocator<Vector2d> >& fts_c1; 
  const vector<Vector2d, aligned_allocator<Vector2d> >& fts_c2; 
  vector<bool> inliers;
  SE3 T_c2_from_c1;             
  Matrix3d H_c2_from_c1;                   
  vector<HomographyDecomposition> decompositions;
};
} /* end namespace vk */
