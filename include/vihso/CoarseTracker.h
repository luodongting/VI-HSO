#ifndef COARSETRACKER_H
#define COARSETRACKER_H


#include "vihso/global.h"
#include "vihso/MatrixAccumulator.h"

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/linear_solver_eigen.h>
#include <g2o/solvers/linear_solver_dense.h>
#include <g2o/types/types_six_dof_expmap.h>
#include <g2o/core/sparse_optimizer_terminate_action.h>
#include "G2oTypes.h"
#include "ImuTypes.h"

namespace vihso {

struct Feature;

class CoarseTracker
{
	typedef Eigen::Matrix<double,7,7> Matrix7d;
	typedef Eigen::Matrix<double,7,1> Vector7d;
	typedef Eigen::Matrix<double,6,6> Matrix6d;
	typedef Eigen::Matrix<double,6,1> Vector6d;
	typedef Eigen::Matrix<double,9,1> Vector9d;
	typedef Eigen::Matrix<double,4,4> Matrix4d;

	int pattern_0[13][2] = {{0,-2}, {-1,-1}, {0,-1}, {1,-1}, {-2,0}, {-1,0}, {0,0}, {1,0}, {2,0}, {-1,1}, {0,1}, {1,1}, {0,2}};
	int pattern_L[13][2] = {{-2,-2}, {0,-2}, {2,-2}, {-1,-1}, {1,-1}, {-2,0}, {0,0}, {2,0}, {-1,1}, {1,1}, {-2,2}, {0,2}, {2,2}};

	int staticPattern[8][40][2] = 
	{
		{{0,0}, 	  {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},	
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		{{0,-1},	  {-1,0},	   {0,0},	    {1,0},	     {0,1}, 	  {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},	
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		{{-1,-1},	  {-1,0},	   {-1,1},		{0,-1},		 {0,0},		  {0,1},	   {1,-1},		{1,0},		 {1,1},       {-100,-100},	
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		{{0,-2},	  {-1,-1},	   {1,-1},		{-2,0},		 {0,0},		  {2,0},	   {-1,1},		{1,1},		 {0,2},       {0,-1},	    
		 {-1,0},      {1,0},       {0,1},       {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		{{0,-2},	  {-1,-1},	   {1,-1},		{-2,0},		 {0,0},		  {2,0},	   {-1,1},		{1,1},		 {0,2},       {-2,-2},   
		 {-2,2},      {2,-2},      {2,2},       {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		{{0,-2},	  {-1,-1},	   {1,-1},		{-2,0},		 {0,0},		  {2,0},	   {-1,1},		{1,1},		 {0,2},       {-2,-2},   
		 {-2,2},      {2,-2},      {2,2},       {-3,-1},     {-3,1},      {3,-1}, 	   {3,1},       {1,-3},      {-1,-3},     {1,3},
		 {-1,3},      {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		{{-2,-2},     {-2,-1}, {-2,-0}, {-2,1}, {-2,2}, {-1,-2}, {-1,-1}, {-1,-0}, {-1,1}, {-1,2}, 										
		 {-0,-2},     {-0,-1}, {-0,-0}, {-0,1}, {-0,2}, {+1,-2}, {+1,-1}, {+1,-0}, {+1,1}, {+1,2},
		 {+2,-2}, 	  {+2,-1}, {+2,-0}, {+2,1}, {+2,2}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		{{-4,-4},     {-4,-2}, {-4,-0}, {-4,2}, {-4,4}, {-2,-4}, {-2,-2}, {-2,-0}, {-2,2}, {-2,4}, 										
		 {-0,-4},     {-0,-2}, {-0,-0}, {-0,2}, {-0,4}, {+2,-4}, {+2,-2}, {+2,-0}, {+2,2}, {+2,4},
		 {+4,-4}, 	  {+4,-2}, {+4,-0}, {+4,2}, {+4,4}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200},
		 {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}},
	};
	int staticPatternNum[8] = {1, 5, 9, 13, 13, 21, 25, 25}; 
	int staticPatternPadding[8] = {1, 1, 1, 2, 2, 3, 2, 4};
	int m_pattern_offset = 2;
	int m_offset_all = 0;
	int PATCH_AREA = 13; 
	int HALF_PATCH_SIZE = 2;
	std::vector<double> imu_track_w;

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	CoarseTracker(bool inverse_composition, int max_level, int min_level, int n_iter, bool verbose);
	~CoarseTracker();
	
	size_t run(FramePtr ref_frame, FramePtr cur_frame);
	size_t runForRelocalization(FramePtr ref_frame, FramePtr cur_frame, Sophus::SE3& Tcw_curframe);

	bool m_inverse_composition;		
	int m_max_level, m_min_level;	
	int m_n_iter;					
	bool m_verbose;					

	FramePtr m_ref_frame, m_cur_frame;	
	SE3 m_T_cur_ref;	

	bool bIMUTrack=false;
	
private:
	Vector2d computeResiduals(const SE3& T_cur_ref, float exposure_rat, double cutoff_error, float b=0);
	double calcIMUResAndGS(Matrix6d& H_out, Vector6d& b_out, const SE3& T21, IMU::Preintegrated* IMU_preintegrator, Vector9d &res_RVP, double PointEnergy, double imu_track_weight);
	void addEdgePhotometricG2O(g2o::SparseOptimizer* optimizer, vihso::VertexPhotometricPose* VT21, vihso::VertexExposureRatio* VExpRatio, double cutoff_error);

	void computeGS(Matrix7d& H_out, Vector7d& b_out);
	void computeGS(Matrix6d& H_out, Vector6d& b_out);

	void precomputeReferencePatches();

	void selectRobustFunctionLevel(const SE3& T_cur_ref, float exposure_rat, float b=0);

	void makeDepthRef();

	Vector2f lineFit(vector<float>& X, vector<float>& Y, float a, float b=0);

	Accumulator7 m_acc7;

	float m_exposure_rat, m_b;	

	int m_level;	
	int m_iter;		

	int m_total_terms;		
	int m_saturated_terms;	

	std::vector<bool> m_visible_fts;	
	cv::Mat m_ref_patch_cache;			

	Matrix<double, 6, Dynamic, ColMajor> m_jacobian_cache_true;	
    Matrix<double, 6, Dynamic, ColMajor> m_jacobian_cache_raw;	

    std::vector<Vector7d> m_buf_jacobian;	
    std::vector<double> m_buf_weight;		
    std::vector<double> m_buf_error;		

    std::vector<double> m_pt_ref; 	
    std::vector<Vector3d> m_pt_host;

    std::vector<float> m_color_cur;	
    std::vector<float> m_color_ref;	
	
	float m_huber_thresh;	
    float m_outlier_thresh;	
};
} // namespace vihso

#endif //COARSETRACKER_H