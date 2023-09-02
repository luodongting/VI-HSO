#ifndef VIHSO_FRAME_H_
#define VIHSO_FRAME_H_

#include <sophus/se3.h>
#include <boost/noncopyable.hpp>
#include <boost/thread.hpp>
#include <vihso/global.h>

#include "vihso/camera.h"
#include "ImuTypes.h"

namespace g2o {
class VertexSE3Expmap;
}
typedef g2o::VertexSE3Expmap g2oFrameSE3;

namespace vihso {

class Point;
class PhotomatricCalibration;
struct Feature;
class ConstraintPoseImu;

typedef list<Feature*> Features;
typedef vector<cv::Mat> ImgPyr;

#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class Frame : boost::noncopyable
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static int frame_counter_;  
    static int keyFrameCounter_;

    int id_;                   
    int keyFrameId_;            
    bool mbKF;                  
      
    double mTimeStamp;          
    hso::AbstractCamera* cam_;  
    Sophus::SE3 T_f_w_;         
    ImgPyr img_pyr_;            
    Matrix<double, 6, 6> Cov_;  
    Features fts_;             
	Features temp_fts_;			
    std::mutex mMutexFeatures;  
    vector<Feature*> key_pts_;  
    g2oFrameSE3* v_kf_;                 
    int last_published_ts_;              

    ImgPyr sobelX_, sobelY_; 
    ImgPyr grad_pyr_, canny_;  
    float integralImage_;       
    float gradMean_;            
    std::vector<Frame*> connectedKeyFrames; 
    int lastReprojectFrameId_;              

    PhotomatricCalibration* m_pc=NULL;  
    ImgPyr m_pyr_raw;
    double m_exposure_time = -1;       
    bool m_exposure_finish = false;    
    bool m_added=false;
    bool m_kf_pc=false;
    FramePtr m_last_frame;     
    float m_error_in_px = 1.f; 
    int m_n_inliers;           


    Frame(hso::AbstractCamera* cam, const cv::Mat& img, double timestamp, PhotomatricCalibration* opc=NULL);
    Frame(hso::AbstractCamera* cam, const cv::Mat& img, double timestamp,  FramePtr pPrevF, const IMU::Calib &ImuCalib, PhotomatricCalibration* opc=NULL);
    ~Frame();

    void initFrame(const cv::Mat& img);
    void addFeature(Feature* ftr);
	void addTempFeature(Feature* ftr);
    void getFeaturesCopy(Features& list_copy);
    void setKeyframe();
    void setKeyPoints();
    void checkKeyPoints(Feature* ftr);
    void removeKeyPoint(Feature* ftr);
    void prepareForFeatureDetect();
    void photometricallyCorrectPyramid(const cv::Mat& img_level_0, ImgPyr& pyr_correct, ImgPyr& pyr_raw, int n_levels);
    void finish();
    bool isVisible(const Vector3d& point_w);

    inline size_t nObs()  { return fts_.size(); }		
    inline const cv::Mat& img() { return img_pyr_[0]; }	
    inline bool isKeyframe()  { return mbKF; }			

    Vector2d w2c(const Vector3d& xyz_w);
    Vector3d w2f(const Vector3d& xyz_w);
    Vector3d c2f(const Vector2d& px);
    Vector3d c2f(const double x, const double y);
    Vector3d f2w(const Vector3d& f);
    Vector2d f2c(const Vector3d& f);

    inline static void jacobian_xyz2uv(const Vector3d& xyz_in_f, Matrix<double,2,6>& J)
    {
        const double x = xyz_in_f[0];
        const double y = xyz_in_f[1];
        const double z_inv = 1./xyz_in_f[2];
        const double z_inv_2 = z_inv*z_inv;

        J(0,0) = -z_inv;              
        J(0,1) = 0.0;                 
        J(0,2) = x*z_inv_2;           
        J(0,3) = y*J(0,2);            
        J(0,4) = -(1.0 + x*J(0,2));   
        J(0,5) = y*z_inv;             

        J(1,0) = 0.0;                 
        J(1,1) = -z_inv;              
        J(1,2) = y*z_inv_2;           
        J(1,3) = 1.0 + y*J(1,2);      
        J(1,4) = -J(0,3);             
        J(1,5) = -x*z_inv;            
    }

public:
    
    Eigen::Vector3d mVw;   
    IMU::Bias mImuBias;     
    IMU::Calib mImuCalib;   
    bool mbImuPreintegrated;
    bool bImu;             
    FramePtr mpLastKeyFrame;   
    FramePtr mpNextKeyFrame;    
    IMU::Preintegrated* mpImuPreintegratedFrame;   
    IMU::Preintegrated* mpImuPreintegrated;        

    Eigen::Matrix3d mPredRwb;
    Eigen::Vector3d mPredVwb;
    Eigen::Vector3d mPredtwb;
    IMU::Bias mPredBias;
    ConstraintPoseImu* mpcpi;   

    int mnBALocalForKF;   
    int mnBAFixedForKF;   

    double mProcessTime = 0.0;  

    std::list<pair<double,Eigen::Matrix4d>> mlRelativeFrame;
	std::map<Frame*,int> mConnectedKeyFrameWeights;		
	std::vector<Frame*> mvpOrderedConnectedKeyFrames;	
    std::vector<int> mvOrderedWeights;					

	bool mbFirstConnection;
	Frame* mpParent;
    std::set<Frame*> mspChildrens;

public:
    void SetPose(const Sophus::SE3 &Tcw);
    void SetImuPoseVelocity(const Eigen::Matrix3d &Rwb, const Eigen::Vector3d &twb, const Eigen::Vector3d &Vwb);
  
    void SetNewBias(const IMU::Bias &b);
    void SetVelocity(const Eigen::Vector3d &Vwb);
    void setIntegrated();

    Eigen::Matrix4d GetPose();
    Sophus::SE3 GetPoseSE3();
    Eigen::Matrix4d GetPoseInverse();
	Sophus::SE3 GetPoseInverseSE3();
    Eigen::Vector3d GetCameraCenter();
    Eigen::Vector3d GetImuPosition();
    Eigen::Matrix3d GetImuRotation();
    Eigen::Matrix4d GetImuPose();
    Eigen::Matrix3d GetRotation();
    Eigen::Vector3d GetTranslation();
    Eigen::Vector3d GetVelocity();

    IMU::Bias GetImuBias();
    Eigen::Vector3d GetGyroBias();
    Eigen::Vector3d GetAccBias();

    int GetKeyFrameID(){ return keyFrameId_; }
    inline void resetKFid(){keyFrameCounter_ = 0;}

    std::mutex mMutexPose;
    std::mutex mpMutexImu;
	std::mutex mMutexConnections;

    inline Eigen::Matrix<double, 2, 3> projectJacUV(const Eigen::Vector3d &v3D) 
    {
        Eigen::Matrix<double, 2, 3> Jac;
        double fx = cam_->K()(0,0);
        double fy = cam_->K()(1,1);

        Jac(0, 0) = fx / v3D[2];
        Jac(0, 1) = 0.f;
        Jac(0, 2) = -fx * v3D[0] / (v3D[2] * v3D[2]);
        Jac(1, 0) = 0.f;
        Jac(1, 1) = fy / v3D[2];
        Jac(1, 2) = -fy * v3D[1] / (v3D[2] * v3D[2]);

        return Jac;
    }
    inline Eigen::Matrix<double, 2, 3> projectJac(const Eigen::Vector3d &v3D) 
    {
        Eigen::Matrix<double, 2, 3> Jac;
 
        Jac(0, 0) = 1 / v3D[2];
        Jac(0, 1) = 0.0;
        Jac(0, 2) = -v3D[0] / (v3D[2] * v3D[2]);
        Jac(1, 0) = 0.0;
        Jac(1, 1) = 1 / v3D[2];
        Jac(1, 2) = -v3D[1] / (v3D[2] * v3D[2]);

        return Jac;
    }

    Features GetFeatures();
	void UpdateFeatures();
	double GetMaxDepth();

    void AddConnection(Frame* pKF, const int &weight);
    void EraseConnection(Frame* pKF);
	void UpdateBestCovisibles();
    void UpdateConnections();
	std::set<Frame *> GetConnectedKeyFrames();
    std::vector<Frame* > GetVectorCovisibleKeyFrames();
    std::vector<Frame*> GetBestCovisibilityKeyFrames(const int &N);
    std::vector<Frame*> GetCovisiblesByWeight(const int &w);
    int GetWeight(Frame* pKF);

    void AddChild(Frame* pKF);
    void EraseChild(Frame* pKF);
    void ChangeParent(Frame* pKF);
    std::set<Frame*> GetChilds();
    Frame* GetParent();
    bool hasChild(Frame* pKF);
    void SetFirstConnection(bool bFirst);

};

namespace frame_utils 
{
    void createImgPyramid(const cv::Mat& img_level_0, int n_levels, ImgPyr& pyr);   
    void createImgGrad(const ImgPyr& pyr_img, ImgPyr& scharr, int n_levels);       
    bool getSceneDepth(Frame& frame, double& depth_mean, double& depth_min);  
    bool getSceneDistance(Frame& frame, double& distance_mean);              
    void createIntegralImage(const cv::Mat& image, float& integralImage);         
    bool frameCovisibilityComparator(pair<int, Frame*>& lhs, pair<int, Frame*>& rhs);
	bool weightComp( int a, int b);

} // namespace frame_utils
} // namespace vihso

#endif // VIHSO_FRAME_H_
