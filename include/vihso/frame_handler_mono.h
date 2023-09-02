#ifndef VIHSO_FRAME_HANDLER_H_
#define VIHSO_FRAME_HANDLER_H_

#include <set>
#include <chrono>
#include <boost/thread.hpp>
#include <vihso/frame_handler_base.h>
#include <vihso/reprojector.h>
#include <vihso/initialization.h>

#include "vihso/PhotomatricCalibration.h"
#include "vihso/camera.h"
#include "LocalMapping.h"
#include "vihso/depth_filter.h"

namespace vihso {

class FrameHandlerMono : public FrameHandlerBase
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	FrameHandlerMono(Map* pMap, hso::AbstractCamera* cam, bool _use_pc=false);
	virtual ~FrameHandlerMono();

	void addIMGIMU(const cv::Mat& img, const std::vector<IMU::IMUPoint>& vIMUMeans, double timestamp);  
	FramePtr lastFrame() { return last_frame_; }
	const set<FramePtr>& coreKeyframes() { return core_kfs_; }
	const vector<cv::Point2f>& initFeatureTrackRefPx() const { return klt_homography_init_.px_ref_; }
	const vector<cv::Point2f>& initFeatureTrackCurPx() const { return klt_homography_init_.px_cur_; }
	DepthFilter* GetDepthFilter() const { return mpDepthFilter; }
	hso::AbstractCamera* cam_;  
	Reprojector reprojector_;   
	FramePtr new_frame_;    
	FramePtr last_frame_;   
	FramePtr firstFrame_;   

	set<FramePtr> core_kfs_;                     
	vector< pair<FramePtr,size_t> > overlap_kfs_; 

	initialization::KltHomographyInit klt_homography_init_;
	SE3 motionModel_; 
	bool afterInit_ = false;  
	PhotomatricCalibration* m_photomatric_calib;

	vector<Frame*> vpLocalKeyFrames; 
	virtual void initialize();
	virtual void resetAll();

	set<Frame*> LocalMap_; 
	static bool frameCovisibilityComparator(pair<int, Frame*>& lhs, pair<int, Frame*>& rhs);
	static bool frameCovisibilityComparatorF(pair<float, Frame*>& lhs, pair<float, Frame*>& rhs);
	void createCovisibilityGraph(FramePtr currentFrame, size_t n_closest, bool is_keyframe);

	IMU::Calib *mpImuCalib; 
	IMU::Bias mLastBias;
	std::vector<IMU::IMUPoint> mvImuFromLastFrame;
	bool bInitializing; 
	bool mbMapUpdated;   

	FramePtr mpLastKeyFrame;
	IMU::Preintegrated *mpImuPreintegratedFromLastKF;

	Eigen::Matrix3d mRwg; 
	Eigen::Vector3d mbg;
	Eigen::Vector3d mba;
	double mScale;
	double mFirstTs;  
	double mTinit;    
	int mnScaleRefinement;  
	double mFULLBATime;     

	bool mbMapViewMonitoring=false; 
	string mFILE_NAME; 

	size_t mnMatchesInliers = 0; 

	double mTimeStampLost=0.0;	
	bool mbFramelost=false;		

	LocalMapping* mpLocalMapper;
	DepthFilter* mpDepthFilter;

public:

	void PreintegrateIMU(const std::vector<IMU::IMUPoint>& vIMUMeans);

	void MonocularInitialization();

	FrameHandlerBase::TrackingResult TrackWithFrame();
	FrameHandlerBase::TrackingResult TrackWithFrameAndIMU();
	FrameHandlerMono::TrackingResult RelocalizeWithIMU();

	void InitializeIMU(double priorG = 1e2, double priorA = 1e6, bool bFirst = false);
	void UpdateFrameIMU(const double s, const IMU::Bias &b);
	void ScaleRefinement();
	bool PredictStateIMU();
	bool NeedNewKeyFrame();
	void SetLocalMapper(LocalMapping *pLocalMapper);
	void SetDepthFilter(DepthFilter *pDepthFilter);

	std::vector<FramePtr> vKFsForScale;		
	std::vector<Vector3d> vKFPosesForScale;	
	int mnScaleAdjustmentKFID;
	double mdScaleAdjustmentTime;
	void ScaleDelayCorrection();

	vector< pair<string, double> > m_stamp_et;  
	vector<double> m_grad_mean;               
	vector<Vector3d> m_vig;

	inline double sqr(double arg) { return arg*arg;}
	inline Eigen::Vector3d getEulerangle(Eigen::Matrix3d &R_21)
	{
		double epsilon=1E-12;
		double yaw=0, pitch=0, roll=0;
		pitch = atan2(-R_21(2,0), sqrt( sqr(R_21(0,0))) +sqr(R_21(1 ,0))  );
		if ( fabs(pitch) > (M_PI_2-epsilon) ) {
			yaw = atan2(	-R_21(0,1), R_21(1 ,1) );
			roll  = 0.0 ;
		} else {
			roll  = atan2(R_21(2, 1), R_21(2,2));
			yaw   = atan2(R_21(1 ,0), R_21(0,0));
		}
		Eigen::Vector3d R_ZYX(yaw, roll, pitch);
		return R_ZYX;
	}

};

} // namespace vihso

#endif 
