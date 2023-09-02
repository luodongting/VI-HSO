#ifndef VIHSO_INITIALIZATION_H
#define VIHSO_INITIALIZATION_H

#include <vihso/global.h>

namespace vihso {

class FrameHandlerMono;

namespace initialization
{

enum InitResult { FAILURE, NO_KEYFRAME, SUCCESS };

enum class InitializerType {
    kHomography,       
    kTwoPoint,        
    kFivePoint,       
    kOneShot,         
    kStereo,          
    kArrayGeometric,  
    kArrayOptimization 
};

class KltHomographyInit
{

friend class vihso::FrameHandlerMono; 

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    KltHomographyInit(): mbref(false)
    {}
    ~KltHomographyInit()
	{}
    
    FramePtr frame_ref_;   
    bool mbref;            
    cv::Mat cvImgshow;     
    InitResult addFirstFrame(FramePtr frame_ref);
    InitResult addSecondFrame(FramePtr frame_ref);
    void reset();
	int ComputeTwoFrameKLT(FramePtr frame_ref, FramePtr frame_cur, Sophus::SE3 Tcr);

protected:
    vector<cv::Point2f> px_ref_;   
    vector<cv::Point2f> px_cur_;  
    vector<Vector3d> f_ref_;        
    vector<Vector3d> f_cur_;       
    vector<double> disparities_;    
    vector<int> inliers_;          
    vector<Vector3d> xyz_in_cur_;   
    SE3 T_cur_from_ref_;            
    InitializerType init_type_;    
    cv::Mat img_prev_;              
    vector<cv::Point2f> px_prev_;  
    vector<Vector3d> ftr_type_;     
};

void detectFeatures(FramePtr frame, vector<cv::Point2f>& px_vec, vector<Vector3d>& f_vec, vector<Vector3d>& ftr_type);

void trackKlt(  FramePtr frame_ref,
                FramePtr frame_cur,
                vector<cv::Point2f>& px_ref,
                vector<cv::Point2f>& px_cur,
                vector<Vector3d>& f_ref,
                vector<Vector3d>& f_cur,
                vector<double>& disparities,
                cv::Mat& img_prev, 
                vector<cv::Point2f>& px_prev,
                vector<Vector3d>& fts_type);

void computeInitializeMatrix(   const vector<Vector3d>& f_ref,
                                const vector<Vector3d>& f_cur,
                                double focal_length,
                                double reprojection_threshold,
                                vector<int>& inliers,
                                vector<Vector3d>& xyz_in_cur,
                                SE3& T_cur_from_ref);

double computeP3D(  const vector<Vector3d>& vBearingRef,
                    const vector<Vector3d>& vBearingCur,
                    const Matrix3d& R,
                    const Vector3d& t,
                    const double reproj_thresh,
                    double error_multiplier2,
                    vector<Vector3d>& vP3D,
                    vector<int>& inliers);

bool patchCheck(const cv::Mat& imgPre, const cv::Mat& imgCur, const cv::Point2f& pxPre, const cv::Point2f& pxCur);
bool createPatch(const cv::Mat& img, const cv::Point2f& px, float* patch);
bool checkSSD(float* patch1, float* patch2);    

Vector3d distancePointOnce(const Vector3d pointW, Vector3d bearingRef, Vector3d bearingCur, SE3 T_c_r);

} // namespace initialization

} // namespace vihso

#endif // VIHSO_INITIALIZATION_H
