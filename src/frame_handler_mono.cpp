#include <boost/timer.hpp>
// #include <boost/timer/timer.hpp>
#include <boost/thread.hpp>
#include <vihso/config.h>
#include <vihso/frame_handler_mono.h>
#include <vihso/map.h>
#include <vihso/frame.h>
#include <vihso/feature.h>
#include <vihso/point.h>
#include <vihso/pose_optimizer.h>
#include <vihso/matcher.h>
#include <vihso/feature_alignment.h>
#include <vihso/global.h>
#include <vihso/bundle_adjustment.h>
#include "vihso/CoarseTracker.h"
#include "vihso/camera.h"
#include "vihso/vikit/performance_monitor.h"
namespace vihso {
FrameHandlerMono::FrameHandlerMono(Map* pMap, hso::AbstractCamera* cam, bool _use_pc) :
                                  FrameHandlerBase(pMap), cam_(cam), reprojector_(cam_, mpMap),
                                  new_frame_(NULL), last_frame_(NULL), firstFrame_(NULL), motionModel_(SE3(Matrix3d::Identity(), Vector3d::Zero())),
                                  mpLocalMapper(NULL), mpDepthFilter(NULL)
{    
    if(_use_pc)
        m_photomatric_calib = new PhotomatricCalibration(2, cam_->width(), cam_->height());
    else
        m_photomatric_calib = NULL;
    initialize();
}
void FrameHandlerMono::initialize()
{
    Eigen::Matrix4d Tbc = vTbc[0];
    mpImuCalib = new IMU::Calib(Tbc, GYR_N*IMUFREQ_sqrt, ACC_N*IMUFREQ_sqrt, GYR_W/IMUFREQ_sqrt, ACC_W/IMUFREQ_sqrt);
    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);
    bInitializing = false;
    mbMapUpdated = false;
    mnScaleRefinement = 0;
}
FrameHandlerMono::~FrameHandlerMono()
{
    delete mpDepthFilter;
    if(m_photomatric_calib != NULL) 
        delete m_photomatric_calib;
}
void FrameHandlerMono::resetAll()
{
    resetCommon();
    last_frame_.reset();
    new_frame_.reset();
    core_kfs_.clear();
    overlap_kfs_.clear();
    mpDepthFilter->reset();
}
bool FrameHandlerMono::frameCovisibilityComparator(pair<int, Frame*>& lhs, pair<int, Frame*>& rhs)
{
    if(lhs.first != rhs.first)
        return (lhs.first > rhs.first);
    else
        return (lhs.second->id_ < rhs.second->id_);
}
bool FrameHandlerMono::frameCovisibilityComparatorF(pair<float, Frame*>& lhs, pair<float, Frame*>& rhs)
{
    return lhs.first > rhs.first;
}
void FrameHandlerMono::createCovisibilityGraph(FramePtr currentFrame, size_t n_closest, bool is_keyframe)   
{
    std::map<Frame*, int> KFcounter;    
    int n_linliers = 0;
	Features lfts = currentFrame->GetFeatures();
    for(Features::iterator it = lfts.begin(); it != lfts.end(); ++it)
    {
        if((*it)->point == NULL) 
            continue;
        n_linliers++;
		Point* pMP = (*it)->point;
        list<Feature*> observations = pMP->GetObservations();
        for(auto ite=observations.begin(); ite!=observations.end(); ++ite)
        {
            if((*ite)->frame->id_== currentFrame->id_) continue;
            KFcounter[(*ite)->frame]++;
        }
    }
    if(KFcounter.empty()) return;
    int nmax=0;         
    Frame* pKFmax=NULL; 
    const int th = n_linliers > 30? 5 : 3;
    vector< pair<int, Frame*> > vPairs; 
    vPairs.reserve(KFcounter.size());  
    vpLocalKeyFrames.clear();
    for(std::map<Frame*,int>::iterator mit=KFcounter.begin(), mend=KFcounter.end(); mit!=mend; mit++)
    {
        vpLocalKeyFrames.push_back(mit->first); 
        if(mit->second>nmax) 
        {
            nmax=mit->second;
            pKFmax=mit->first;
        }
        if(mit->second>=th)
            vPairs.push_back(make_pair(mit->second,mit->first));
        if(mit->first->keyFrameId_+10 < currentFrame->keyFrameId_) 
		{
			if(!mit->first->sobelX_.empty())
            {
                mit->first->sobelX_.clear();
                mit->first->sobelY_.clear();
            }
		}
    }
    if(vPairs.empty())
        vPairs.push_back(make_pair(nmax,pKFmax));
    std::partial_sort(vPairs.begin(), vPairs.begin()+vPairs.size(), vPairs.end(), boost::bind(&FrameHandlerMono::frameCovisibilityComparator, _1, _2));
    const size_t nCovisibility = 5;
    size_t k = min(nCovisibility, vPairs.size());
    for(size_t i = 0; i < k; ++i)
        currentFrame->connectedKeyFrames.push_back(vPairs[i].second);
    if(is_keyframe)
    {
        LocalMap_.clear();
        size_t n = min(n_closest, vPairs.size());
        std::for_each(vPairs.begin(), vPairs.begin()+n, [&](pair<int, Frame*>& i){ LocalMap_.insert(i.second); });
        FramePtr LastKF = mpMap->lastKeyframe();
        if(LocalMap_.find(LastKF.get()) == LocalMap_.end())
        {
            LocalMap_.insert(LastKF.get()); 
        }
        assert(LocalMap_.find(currentFrame.get()) == LocalMap_.end());
        LocalMap_.insert(currentFrame.get());
    }
}
void FrameHandlerMono::addIMGIMU(const cv::Mat& img, const std::vector<IMU::IMUPoint>& vIMUMeans, double timestamp)
{
    if(!FrameHandlerBase::startStateCheck(timestamp))
        return;
    overlap_kfs_.clear();
    new_frame_.reset(new Frame(cam_, img.clone(), timestamp, last_frame_, *mpImuCalib, m_photomatric_calib));
	if(mpMap->nKFsInMap == 0)
    {
        new_frame_->keyFrameId_ = 0;
    }
    else
    {
        mbMapUpdated = false;	
        int nCurMapChangeIndex = mpMap->GetMapChangeIndex();
        int nMapChangeIndex = mpMap->GetLastMapChange();
        if(nCurMapChangeIndex>nMapChangeIndex)
        {
            mpMap->SetLastMapChange(nCurMapChangeIndex);
            mbMapUpdated = true;
        }
        new_frame_->keyFrameId_ = mpLastKeyFrame->keyFrameId_;
        new_frame_->mpLastKeyFrame = mpLastKeyFrame;
        if(mbMapUpdated)
            mbMapViewMonitoring = true;
    }
    if(last_frame_.get())
        new_frame_->SetNewBias(last_frame_->GetImuBias());
    PreintegrateIMU(vIMUMeans);
    if(mState == NO_IMAGES_YET) 
    {
        mState = NOT_INITIALIZED;
    }
    if(mState == NOT_INITIALIZED)
    {
        MonocularInitialization();
        last_frame_ = new_frame_;
        new_frame_.reset();
        finishStateCheck();
        return;
    }
    else
    {
        FrameHandlerBase::TrackingResult resTrack = RES_Frame;
        if(mState == ONLY_VISIUAL_INIT)
        {
            resTrack = TrackWithFrame();
            if(resTrack == RES_KF && (!mpMap->isImuInitialized()))  
            {
                InitializeIMU(priorG0, priorA0, true);
            }
        }
        else if(mState == OK)
        {
            resTrack = TrackWithFrameAndIMU();
			if(resTrack == RES_FAILURE)
			{
				if(mpMap->GetNumOfKF()>10)   
				{
					mTimeStampLost = new_frame_->mTimeStamp;
					mState = RECENTLY_LOST;
					mbFramelost = true;
				}
				else
				{
					mState = LOST;
				}
			}
            mTinit += new_frame_->mTimeStamp - last_frame_->mTimeStamp;
            if(mTinit<300.0 && new_frame_->isKeyframe())
            {
                if(mpMap->isImuInitialized() && (resTrack != RES_FAILURE))
                {
                    if(!mpMap->isIniertialBA1())
                    {
                        if (mTinit>5.0f)
                        {
                            InitializeIMU(priorG1, priorA1, true);
                            mpMap->SetIniertialBA1();
                        }
                    }
                    if(!mpMap->isIniertialBA2())
                    {
                        if (mTinit>15.0f)
                        { 
                            InitializeIMU(priorG2, priorA2, true);
                            mpMap->SetIniertialBA2();														
                        }
                    }
                }
            }
        }
		if(mState == RECENTLY_LOST || mState == LOST)
        {
			if (mState == RECENTLY_LOST) 
			{
				cout << "Lost for a short time" << endl;
				resTrack = RES_Frame;
				if(mpMap->isImuInitialized())
					resTrack = RelocalizeWithIMU();
				else
					resTrack = RES_FAILURE;
				if(resTrack == RES_KF)
				{
					mState = OK;
				}
				else if (new_frame_->mTimeStamp - mTimeStampLost > 5.0)
				{
					mState = LOST;
					cout << "Track Lost..." << endl;
					resTrack = RES_FAILURE;
				}
			}
			else if (mState == LOST)  
			{	
				if (mpMap->GetNumOfKF()<10)
				{
					mpMap->reset();
					cout << "Reseting current map..." << endl;
				}
				else
					mpMap->reset();
				if(mpLastKeyFrame)
					mpLastKeyFrame = static_cast<vihso::FramePtr>(NULL);
				last_frame_ = NULL;
        		new_frame_.reset();
				return;
			}
        }
        last_frame_ = new_frame_;
        new_frame_.reset();
        finishStateCheck(last_frame_, cam_->width(), cam_->height());
    }
}
void FrameHandlerMono::PreintegrateIMU(const std::vector<IMU::IMUPoint>& vIMUMeans)
{
    if(!new_frame_->m_last_frame)    
    {
        new_frame_->SetNewBias(IMU::Bias());
        new_frame_->setIntegrated();
        return;
    }
    if(vIMUMeans.size() == 0)   
    {
        new_frame_->setIntegrated();
        return;
    }
    mvImuFromLastFrame.clear();
    mvImuFromLastFrame.reserve(vIMUMeans.size());
    double curtimestamp = new_frame_->mTimeStamp, lasttimestamp = last_frame_->mTimeStamp;
    for(size_t i=0; i<vIMUMeans.size(); i++)
    {
        IMU::IMUPoint m = vIMUMeans[i];
        if(m.t < lasttimestamp-0.001l)   
            continue;
        if(m.t >= curtimestamp+0.001l)
            continue;
        mvImuFromLastFrame.push_back(m);
    }
    IMU::Preintegrated* pImuPreintegratedFromLastFrame = new IMU::Preintegrated(last_frame_->mImuBias, new_frame_->mImuCalib);
    const int n = mvImuFromLastFrame.size()-1; 
    for(int i=0; i<n; i++)
    {    
        double tstep;
        Eigen::Vector3d acc, angVel;
        if((i==0) && (i<(n-1))) 
        {
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            float tini = mvImuFromLastFrame[i].t-lasttimestamp;
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tini/tab))*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tini/tab))*0.5f;
            tstep = mvImuFromLastFrame[i+1].t-lasttimestamp;
        }   
        else if(i<(n-1))  
        {
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a)*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w)*0.5f;
            tstep = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
        }
        else if((i>0) && (i==(n-1)))    
        {    
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            float tend = mvImuFromLastFrame[i+1].t-curtimestamp;
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tend/tab))*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tend/tab))*0.5f;
            tstep = curtimestamp-mvImuFromLastFrame[i].t;
        }
        else if((i==0) && (i==(n-1)))  
        {
            acc = mvImuFromLastFrame[i].a;
            angVel = mvImuFromLastFrame[i].w;
            tstep = curtimestamp-lasttimestamp;
        }
        if (!mpImuPreintegratedFromLastKF)
            cout << "mpImuPreintegratedFromLastKF does not exist" << endl;
        mpImuPreintegratedFromLastKF->IntegrateNewMeasurement(acc,angVel,tstep);
        pImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc,angVel,tstep);
    }
    new_frame_->mpImuPreintegratedFrame = pImuPreintegratedFromLastFrame;
    new_frame_->mpImuPreintegrated = mpImuPreintegratedFromLastKF;
    new_frame_->setIntegrated();
    new_frame_->bImu = mpMap->isImuInitialized();
}
void FrameHandlerMono::MonocularInitialization()
{
    if(!klt_homography_init_.mbref)
    {
        new_frame_->SetPose( SE3(Matrix3d::Identity(), Vector3d::Zero()) );
        initialization::InitResult res = klt_homography_init_.addFirstFrame(new_frame_);
        if(res == initialization::FAILURE)
            return;
        new_frame_->setKeyframe(); 
        mpLastKeyFrame = new_frame_;
        mpMap->addKeyframe(new_frame_);    
        firstFrame_ = new_frame_;
        firstFrame_->m_exposure_time = 1.0;
        firstFrame_->m_exposure_finish = true;
        if(mpImuPreintegratedFromLastKF)
        {
            delete mpImuPreintegratedFromLastKF;
        }
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
        new_frame_->mpImuPreintegrated = mpImuPreintegratedFromLastKF;
        return;
    }
    else 
    {        
        if(new_frame_->mTimeStamp - firstFrame_->mTimeStamp > 1.0)
        {
            klt_homography_init_.reset();
            mpMap->reset();
            mpDepthFilter->reset();
            new_frame_->resetKFid();
            return;
        }     
        initialization::InitResult res = klt_homography_init_.addSecondFrame(new_frame_);
        if(res == initialization::FAILURE || res == initialization::NO_KEYFRAME) return;
        size_t ninitPoints =  klt_homography_init_.inliers_.size();
        klt_homography_init_.reset();
        afterInit_ = true;
        firstFrame_->setKeyPoints();
        new_frame_->setKeyframe(); 
        mpLastKeyFrame = new_frame_;
        createCovisibilityGraph(new_frame_, Config::coreNKfs(), true);
        mpMap->addKeyframe(new_frame_);   
        new_frame_->mpLastKeyFrame = firstFrame_;
        firstFrame_->mpNextKeyFrame = new_frame_;
		firstFrame_->UpdateConnections();
		new_frame_->UpdateConnections();
        new_frame_->mpImuPreintegrated = mpImuPreintegratedFromLastKF;
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(new_frame_->mpImuPreintegrated->GetUpdatedBias(),new_frame_->mImuCalib);
        motionModel_ = SE3(Matrix3d::Identity(), Vector3d::Zero()); 
		double depth_mean, depth_min, distance_mean; 
        frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);
        frame_utils::getSceneDistance(*new_frame_, distance_mean);
        mpDepthFilter->addKeyframe(new_frame_, distance_mean, 0.5*depth_min, 200);
        mState = ONLY_VISIUAL_INIT;
		SE3 T1(firstFrame_->GetPoseInverseSE3()),T2(new_frame_->GetPoseInverseSE3());
        return;
    }
}
FrameHandlerBase::TrackingResult FrameHandlerMono::TrackWithFrame()
{
    new_frame_->SetPose( motionModel_ * last_frame_->GetPoseSE3() );
    new_frame_->m_last_frame = last_frame_;
    new_frame_->mpLastKeyFrame = mpLastKeyFrame;
    if(new_frame_->gradMean_ > last_frame_->gradMean_ + 0.5)
    {
        CoarseTracker Tracker(false, Config::kltMaxLevel(), Config::kltMinLevel()+1, 50, false);
        size_t img_align_n_tracked = Tracker.run(last_frame_, new_frame_);
        VIHSO_DEBUG_STREAM("Img Align:\t Tracked = " << img_align_n_tracked <<"\t \t Cost = "<<align.elapsed()<<"s");
    }
    else   
    {
        CoarseTracker Tracker(true, Config::kltMaxLevel(), Config::kltMinLevel()+1, 50, false);
        size_t img_align_n_tracked = Tracker.run(last_frame_, new_frame_);
        VIHSO_DEBUG_STREAM("Img Align:\t Tracked = " << img_align_n_tracked << "\t \t Cost = "<<align.elapsed()<<"s");
    }
    reprojector_.reprojectMap(new_frame_, overlap_kfs_);            
    const size_t repr_n_new_references = reprojector_.n_matches_;   
    const size_t repr_n_mps = reprojector_.n_trials_;               
    const size_t repr_n_sds = reprojector_.n_seeds_;                
    const size_t repr_n_fis = reprojector_.n_filters_;
    if((int)repr_n_new_references < Config::qualityMinFts())
    {
        VIHSO_WARN_STREAM_THROTTLE(1.0, "Not enough matched features.");
        new_frame_->SetPose(last_frame_->GetPoseSE3());
        tracking_quality_ = TRACKING_INSUFFICIENT;
        mpMap->reset();
        mpDepthFilter->reset();
        new_frame_->resetKFid();
        mState = NOT_INITIALIZED;
        return RES_FAILURE;
    }
    size_t sfba_n_edges_final = 0; 
    double sfba_thresh = 0, sfba_error_init = 0, sfba_error_final = 0;
    pose_optimizer::optimizeLevenbergMarquardt3rd(  Config::poseOptimThresh(), Config::poseOptimNumIter(), false, new_frame_,
                                                    sfba_thresh, sfba_error_init,
                                                    sfba_error_final, sfba_n_edges_final);
    new_frame_->m_n_inliers = sfba_n_edges_final;
    VIHSO_DEBUG_STREAM("PoseOptimizer:\t ErrInit = "<<sfba_error_init<<"px\t thresh = "<<sfba_thresh);
    VIHSO_DEBUG_STREAM("PoseOptimizer:\t ErrFin. = "<<sfba_error_final<<"px\t nObsFin. = "<<sfba_n_edges_final);
    if((int)sfba_n_edges_final < Config::qualityMinFts())   
        return RES_FAILURE;
    core_kfs_.insert(new_frame_);
    setTrackingQuality(sfba_n_edges_final);
    if(tracking_quality_ == TRACKING_INSUFFICIENT)
    {
        new_frame_->SetPose(last_frame_->GetPoseSE3());
        return RES_FAILURE;
    }
    if (!((new_frame_->mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.25))
    {
        createCovisibilityGraph(new_frame_, Config::coreNKfs(), false); 
        mpDepthFilter->addFrame(new_frame_);
        motionModel_ = new_frame_->GetPoseSE3() * last_frame_->GetPoseInverseSE3();
        Eigen::Matrix4d Tbibj = mpLastKeyFrame->GetImuPose().inverse() * new_frame_->GetImuPose();
        mpLastKeyFrame->mlRelativeFrame.push_back(make_pair(new_frame_->mTimeStamp, Tbibj)); 
        return RES_Frame;
    }
    else
    {
        if(afterInit_) 
            afterInit_ = false;
        new_frame_->setKeyframe();
        mpLastKeyFrame->mpNextKeyFrame = new_frame_;
        mpLastKeyFrame = new_frame_;
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(new_frame_->GetImuBias(),new_frame_->mImuCalib);
        for(Features::iterator it=new_frame_->fts_.begin(); it!=new_frame_->fts_.end(); ++it)   
            if((*it)->point != NULL)
                (*it)->point->addFrameRef(*it);
        mpMap->mCandidatesManager.addCandidatePointToFrame(new_frame_);    
        createCovisibilityGraph(new_frame_, Config::coreNKfs(), true);  
		new_frame_->UpdateConnections();
        size_t loba_n_erredges_init, loba_n_erredges_fin;
        double loba_err_init, loba_err_fin;
        ba::VisiualOnlyLocalBA(new_frame_.get(), &LocalMap_, mpMap, loba_n_erredges_init, loba_n_erredges_fin, loba_err_init, loba_err_fin);
        for(auto& kf: overlap_kfs_) kf.first->setKeyPoints();
        double depth_mean, depth_min;   
        frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);
        double distance_mean;           
        frame_utils::getSceneDistance(*new_frame_, distance_mean);
        if(sfba_n_edges_final <= 70)
            mpDepthFilter->addKeyframe(new_frame_, distance_mean, 0.5*depth_min, 100);
        else
            mpDepthFilter->addKeyframe(new_frame_, distance_mean, 0.5*depth_min, 200);  
        mpMap->addKeyframe(new_frame_);
        motionModel_ = new_frame_->GetPoseSE3() * last_frame_->GetPoseInverseSE3();
        return RES_KF;
    }
}
FrameHandlerBase::TrackingResult FrameHandlerMono::TrackWithFrameAndIMU()
{
    new_frame_->m_last_frame = last_frame_;
    new_frame_->mpLastKeyFrame = mpLastKeyFrame;
    PredictStateIMU();
	Sophus::SE3 Tcw_IMUPre = new_frame_->GetPoseSE3();
	size_t nCoarseAlignMatches = 0;
	size_t nReprojectorMatches = 0;
	size_t nIMUBAMatches = 0;
	Sophus::SE3 T21 = Tcw_IMUPre * last_frame_->GetPoseSE3().inverse();
	Eigen::Matrix3d R_21 = T21.rotation_matrix();
	Eigen::Vector3d R_ZYX = getEulerangle(R_21);
	double dr = R_ZYX.norm();
	double avggrad = (last_frame_->gradMean_ + last_frame_->mpLastKeyFrame->gradMean_)/60.0;
	double theta = (dr+avggrad)>0.5?(dr+avggrad):0.5;
	if(new_frame_->gradMean_ > last_frame_->gradMean_ + theta)    
	{
		CoarseTracker Tracker(false, Config::kltMaxLevel(), Config::kltMinLevel()+1, 50, false);   
		Tracker.bIMUTrack = true;
		nCoarseAlignMatches = Tracker.run(last_frame_, new_frame_);
	}
	else   
	{
		CoarseTracker invTracker(true, Config::kltMaxLevel(), Config::kltMinLevel()+1, 50, false);
		invTracker.bIMUTrack = true;
		nCoarseAlignMatches = invTracker.run(last_frame_, new_frame_);
	}
    reprojector_.reprojectMap(new_frame_, overlap_kfs_);
	nReprojectorMatches = reprojector_.n_matches_;
    if((int)nReprojectorMatches < 10)
    {
        tracking_quality_ = TRACKING_INSUFFICIENT;
        return RES_FAILURE;
    }
    if(mbMapUpdated)    
        nIMUBAMatches = ba::PoseInertialOptimizationLastKeyFrame(new_frame_, false);
    else    
        nIMUBAMatches = ba::PoseInertialOptimizationLastFrame(new_frame_, false);
    new_frame_->m_n_inliers = nIMUBAMatches;
    core_kfs_.insert(new_frame_);
    setTrackingQuality(nReprojectorMatches);
    if(tracking_quality_ == TRACKING_INSUFFICIENT)
    {
		createCovisibilityGraph(new_frame_, Config::coreNKfs(), false);     
        mpDepthFilter->addFrame(new_frame_);
        motionModel_ = new_frame_->GetPoseSE3() * last_frame_->GetPoseInverseSE3();
        Eigen::Matrix4d Tbibj = mpLastKeyFrame->GetImuPose().inverse() * new_frame_->GetImuPose();
        mpLastKeyFrame->mlRelativeFrame.push_back(make_pair(new_frame_->mTimeStamp, Tbibj));
        regular_counter_++;
        return RES_FAILURE;
    }
    bool bKF = NeedNewKeyFrame();
    if(!bKF)
    {
        createCovisibilityGraph(new_frame_, Config::coreNKfs(), false);      
        mpDepthFilter->addFrame(new_frame_);
        motionModel_ = new_frame_->GetPoseSE3() * last_frame_->GetPoseInverseSE3();
        Eigen::Matrix4d Tbibj = mpLastKeyFrame->GetImuPose().inverse() * new_frame_->GetImuPose();
        mpLastKeyFrame->mlRelativeFrame.push_back(make_pair(new_frame_->mTimeStamp, Tbibj));
        regular_counter_++; 
        return RES_Frame;
    }
    else
    {
        new_frame_->setKeyframe();
		new_frame_->UpdateFeatures();
		Features lfts = new_frame_->GetFeatures();
		size_t nftsAdd=0;
        for(Features::iterator it=lfts.begin(); it!=lfts.end(); ++it)
        {
			if((*it)==NULL) continue;
			if((*it)->point != NULL)
			{
				(*it)->point->addFrameRef(*it);
				nftsAdd++;
			}	
		}
        mpMap->mCandidatesManager.addCandidatePointToFrame(new_frame_);
        createCovisibilityGraph(new_frame_, Config::coreNKfs(), true);
		new_frame_->UpdateConnections();
        ba::visualImuLocalBundleAdjustment(new_frame_.get(), &LocalMap_, mpMap, false);
        for(auto& kf: overlap_kfs_) kf.first->setKeyPoints();
        double depth_mean, depth_min;
        frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);
        double distance_mean;
        frame_utils::getSceneDistance(*new_frame_, distance_mean);
        if(nIMUBAMatches <= 70)
            mpDepthFilter->addKeyframe(new_frame_, distance_mean, 0.5*depth_min, 100);
        else
            mpDepthFilter->addKeyframe(new_frame_, distance_mean, 0.5*depth_min, 200);  
        mpMap->addKeyframe(new_frame_);
        motionModel_ = new_frame_->GetPoseSE3() * last_frame_->GetPoseInverseSE3();
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(new_frame_->GetImuBias(),new_frame_->mImuCalib);
        mpLastKeyFrame->mpNextKeyFrame = new_frame_;
        mpLastKeyFrame = new_frame_;
        regular_counter_ = 0;
		vKFsForScale.push_back(new_frame_);
		vKFPosesForScale.push_back(new_frame_->GetImuPosition());
        return RES_KF;
    }
}
FrameHandlerMono::TrackingResult FrameHandlerMono::RelocalizeWithIMU()
{
    new_frame_->m_last_frame = last_frame_;
    new_frame_->mpLastKeyFrame = mpLastKeyFrame;
    PredictStateIMU();
	Sophus::SE3 TpreIMU = new_frame_->GetPoseSE3();
	size_t nReprojectorMatches=0;
	if(mbFramelost)
	{	
		mbFramelost=false;
	}
	else
	{
    	reprojector_.reprojectMap(new_frame_, overlap_kfs_);
	}
	nReprojectorMatches = reprojector_.n_matches_;
	if(nReprojectorMatches > 10)
	{
		new_frame_->setKeyframe(); 
		for(Features::iterator it=new_frame_->fts_.begin(); it!=new_frame_->fts_.end(); ++it)
			if((*it)->point != NULL)
				(*it)->point->addFrameRef(*it);
		mpMap->mCandidatesManager.addCandidatePointToFrame(new_frame_);  
		createCovisibilityGraph(new_frame_, Config::coreNKfs(), true); 
		new_frame_->UpdateConnections();
		size_t loba_n_erredges_init, loba_n_erredges_fin;
        double loba_err_init, loba_err_fin;
        ba::VisiualOnlyLocalBA(new_frame_.get(), &LocalMap_, mpMap, loba_n_erredges_init, loba_n_erredges_fin, loba_err_init, loba_err_fin);
		for(auto& kf: overlap_kfs_) kf.first->setKeyPoints();
		double depth_mean, depth_min;  
		frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);
		double distance_mean;          
		frame_utils::getSceneDistance(*new_frame_, distance_mean);
		mpDepthFilter->addKeyframe(new_frame_, distance_mean, 0.5*depth_min, 100);  
		mpMap->addKeyframe(new_frame_);
		motionModel_ = new_frame_->GetPoseSE3() * last_frame_->GetPoseInverseSE3();
		mpImuPreintegratedFromLastKF = new IMU::Preintegrated(new_frame_->GetImuBias(),new_frame_->mImuCalib);
		mpLastKeyFrame->mpNextKeyFrame = new_frame_;
		mpLastKeyFrame = new_frame_;
		regular_counter_ = 0;
		return RES_KF;
	}
	else
	{
        createCovisibilityGraph(new_frame_, Config::coreNKfs(), false);     
        mpDepthFilter->addFrame(new_frame_);
        motionModel_ = new_frame_->GetPoseSE3() * last_frame_->GetPoseInverseSE3();
        Eigen::Matrix4d Tbibj = mpLastKeyFrame->GetImuPose().inverse() * new_frame_->GetImuPose();
        mpLastKeyFrame->mlRelativeFrame.push_back(make_pair(new_frame_->mTimeStamp, Tbibj));
        regular_counter_++; 
        return RES_Frame;
	}
}
void FrameHandlerMono::InitializeIMU(double priorG, double priorA, bool bFIBA)
{
    float minTime = 2.0;    
    size_t nMinKF = 16;     
    if(mpMap->keyframes_.size() < nMinKF)
        return;
    if(!new_frame_->isKeyframe())
        return;
    std::list<FramePtr> lpKF;
    FramePtr pKF = new_frame_;
    while(pKF->mpLastKeyFrame) 
    {
        lpKF.push_front(pKF);
        pKF = pKF->mpLastKeyFrame;
    }
    lpKF.push_front(pKF);  
    std::vector<FramePtr> vpKF(lpKF.begin(),lpKF.end());
    mFirstTs = vpKF.front()->mTimeStamp;
    if(new_frame_->mTimeStamp - mFirstTs < minTime) 
        return;
    bInitializing = true;
    const int N = vpKF.size();
    IMU::Bias b(0,0,0,0,0,0);
    if (!mpMap->isImuInitialized())
    {
        Eigen::Vector3d dirG = Eigen::Vector3d::Zero(); 
        for(vector<FramePtr>::iterator itKF = vpKF.begin(); itKF!=vpKF.end(); itKF++)
        {
            if (!(*itKF)->mpImuPreintegrated) 
                continue;
            if (!(*itKF)->mpLastKeyFrame)       
                continue;
            dirG -= (*itKF)->mpLastKeyFrame->GetImuRotation()*(*itKF)->mpImuPreintegrated->GetUpdatedDeltaVelocity();
            Eigen::Vector3d _vel = ((*itKF)->GetImuPosition() - (*itKF)->mpLastKeyFrame->GetImuPosition())/(*itKF)->mpImuPreintegrated->dT;    
            (*itKF)->SetVelocity(_vel);                 
            (*itKF)->mpLastKeyFrame->SetVelocity(_vel);
        }
        dirG = dirG/dirG.norm();         
        Eigen::Vector3d gI(0.0, 0.0, -1.0); 
        Eigen::Vector3d v = gI.cross(dirG); 
        const double nv = v.norm();         
        const double cosg = gI.dot(dirG);   
        const double ang = acos(cosg);     
        Eigen::Vector3d vzg = v*ang/nv;    
        mRwg = IMU::ExpSO3(vzg);          
        mTinit = 0.0;
    }
    else   
    {
        mRwg = Eigen::Matrix3d::Identity(); 
        mbg = new_frame_->GetGyroBias();
        mba = new_frame_->GetAccBias();
    }
    mScale=1.0;
    vihso::ba::InertialOptimization(mpMap, mRwg, mScale, mbg, mba, false, false, priorG, priorA);
    if (mScale<1e-1)
    {
        cout << "scale too small" << endl;
        bInitializing = false;
        return;
    }
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
    if ((fabs(mScale-1.f)>0.00001) && !mpMap->isImuInitialized()) 
    {
        mpMap->IncreaseChangeIndex();
        mpMap->ApplyScaledRotation(mRwg.transpose(), mScale, true);
        mpDepthFilter->ApplyScaledDepthFilter(mScale);
        UpdateFrameIMU(mScale,vpKF[0]->GetImuBias());
    }
    if(!mpMap->isImuInitialized())
    {   
        for(size_t i=0; i<lpKF.size(); i++)
        {
            FramePtr pKF =  vpKF[i];
            pKF->bImu = true;
        }
        vihso::ba::FullInertialBA(mpMap, 100, false, NULL, true, priorGBA0, priorABA0);
        mpMap->SetImuInitialized();
        mState = OK; 
    }
    else if(!mpMap->isIniertialBA1())
    {
        vihso::ba::FullInertialBA(mpMap, 100, false, NULL, true, priorGBA1, priorABA1);
    }
    else if(!mpMap->isIniertialBA2())
    {
        vihso::ba::FullInertialBA(mpMap, 100, false, NULL, false, priorGBA2, priorABA2);
    }
    mFULLBATime = mTinit;  
}
void FrameHandlerMono::UpdateFrameIMU(const double s, const IMU::Bias &b)
{
    mLastBias = b;
    last_frame_->SetNewBias(mLastBias);
    new_frame_->SetNewBias(mLastBias);
    Eigen::Vector3d Gz(0, 0, -IMU::GRAVITY_VALUE);
    Eigen::Vector3d twb1;
    Eigen::Matrix3d Rwb1;
    Eigen::Vector3d Vwb1;
    double t12;
    if(!last_frame_->isKeyframe()) 
    {
        twb1 = last_frame_->mpLastKeyFrame->GetImuPosition();
        Rwb1 = last_frame_->mpLastKeyFrame->GetImuRotation();
        Vwb1 = last_frame_->mpLastKeyFrame->GetVelocity();
        t12 = last_frame_->mpImuPreintegrated->dT;
        Eigen::Matrix3d Rwb2 = Rwb1 * last_frame_->mpImuPreintegrated->GetUpdatedDeltaRotation();
        Eigen::Vector3d twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*last_frame_->mpImuPreintegrated->GetUpdatedDeltaPosition();
        Eigen::Vector3d Vwb2 = Vwb1 + Gz*t12 + Rwb1*last_frame_->mpImuPreintegrated->GetUpdatedDeltaVelocity();
        last_frame_->SetImuPoseVelocity(Rwb2, twb2, Vwb2);
    }
    motionModel_ = new_frame_->GetPoseSE3() * last_frame_->GetPoseInverseSE3();
}
void FrameHandlerMono::ScaleRefinement()
{
    mRwg = Eigen::Matrix3d::Identity();
    mScale=1.0;
    vihso::ba::InertialOptimization(mpMap, mRwg, mScale);
    if (mScale<1e-1)
    {
        cout << "scale too small" << endl;
        bInitializing=false;
        return;
    }
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
    if ((fabs(mScale-1.f)>0.00001))
    {
        mpMap->IncreaseChangeIndex();
        mpMap->ApplyScaledRotation(mRwg.transpose(), mScale, true);
        mpDepthFilter->ApplyScaledDepthFilter(mScale);
        UpdateFrameIMU(mScale,new_frame_->GetImuBias());
    }
    return;
}
void FrameHandlerMono::ScaleDelayCorrection()
{
	if(vKFPosesForScale.size()<50)
		return;
    mRwg = Eigen::Matrix3d::Identity();
    mScale=1.0;
	vihso::ba::ScaleDelayOptimization(vKFsForScale, mRwg, mScale);
	vKFsForScale.clear();
	vKFPosesForScale.clear();
}
bool FrameHandlerMono::PredictStateIMU()
{
    if(!new_frame_->m_last_frame)
    {
        cout << "No last frame" << endl;
        return false;
    }
    if(mbMapUpdated && mpLastKeyFrame)
    {   
        const Eigen::Vector3d twb1 = mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3d Rwb1 = mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3d Vwb1 = mpLastKeyFrame->GetVelocity();
        const Eigen::Vector3d Gz(0,0,-IMU::GRAVITY_VALUE);
        const double t12 = mpImuPreintegratedFromLastKF->dT;
        Eigen::Matrix3d Rwb2 = IMU::NormalizeRotation(Rwb1*mpImuPreintegratedFromLastKF->GetDeltaRotation(mpLastKeyFrame->GetImuBias()));
        Eigen::Vector3d twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mpImuPreintegratedFromLastKF->GetDeltaPosition(mpLastKeyFrame->GetImuBias());
        Eigen::Vector3d Vwb2 = Vwb1 + t12*Gz + Rwb1*mpImuPreintegratedFromLastKF->GetDeltaVelocity(mpLastKeyFrame->GetImuBias());
        new_frame_->SetImuPoseVelocity(Rwb2,twb2,Vwb2);
        return true;
    }
    else if(!mbMapUpdated)
    {
        const Eigen::Vector3d twb1 = last_frame_->GetImuPosition();
        const Eigen::Matrix3d Rwb1 = last_frame_->GetImuRotation();
        const Eigen::Vector3d Vwb1 = last_frame_->mVw;
        const Eigen::Vector3d Gz = Eigen::Vector3d(0,0,-IMU::GRAVITY_VALUE);
        const float t12 = new_frame_->mpImuPreintegratedFrame->dT;
        Eigen::Matrix3d Rwb2 = IMU::NormalizeRotation(Rwb1*new_frame_->mpImuPreintegratedFrame->GetDeltaRotation(last_frame_->mImuBias));
        Eigen::Vector3d twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*new_frame_->mpImuPreintegratedFrame->GetDeltaPosition(last_frame_->mImuBias);
        Eigen::Vector3d Vwb2 = Vwb1 + t12*Gz + Rwb1*new_frame_->mpImuPreintegratedFrame->GetDeltaVelocity(last_frame_->mImuBias);
        new_frame_->SetImuPoseVelocity(Rwb2,twb2,Vwb2);
        return true;
    }
    else
        cout << "not IMU prediction!!" << endl;
    return false;
}
bool FrameHandlerMono::NeedNewKeyFrame()
{
	const bool c1 = regular_counter_ < 5;
    const bool c2 = new_frame_->id_ >= (mpLastKeyFrame->id_ + 10);    
    int nRefMatches = last_frame_->m_n_inliers;
	mnMatchesInliers = new_frame_->m_n_inliers;
    float thRefRatio = 0.90f;
	if(mnMatchesInliers >180)
		thRefRatio = 0.75f;
	else
		thRefRatio = 0.90f;
    const bool c3 = (1.0*mnMatchesInliers<nRefMatches*thRefRatio);  
    bool c4 = false;
	bool c5 = false;
	float DSO = 0.0f;
	if(1)
    {
        const SE3 T_c_r_full(new_frame_->GetPoseSE3() * mpLastKeyFrame->GetPoseInverseSE3()); 
        const SE3 T_c_r_nR(Matrix3d::Identity(), T_c_r_full.translation());     
        float optical_flow_full = 0;   
        float optical_flow_nR = 0;    
        float optical_flow_nt = 0;
        size_t optical_flow_num = 0;    
		std::list<vihso::Feature*> lfts = mpLastKeyFrame->GetFeatures();
        for(auto pft:lfts)
        {
            if(pft->point == NULL) continue;
            Vector3d P_w = pft->point->GetWorldPos();
            Vector3d O_l = mpLastKeyFrame->GetCameraCenter();
            Vector3d p_ref(pft->f * (P_w - O_l).norm());
            Vector3d p_cur_full(T_c_r_full * p_ref);
            Vector3d p_cur_nR(T_c_r_nR * p_ref);
            Vector2d uv_cur_full(new_frame_->cam_->world2cam(p_cur_full));
            Vector2d uv_cur_nR(new_frame_->cam_->world2cam(p_cur_nR));
            optical_flow_full += (uv_cur_full - pft->px).squaredNorm();
            optical_flow_nR += (uv_cur_nR - pft->px).squaredNorm();
            optical_flow_num++;
        }
        optical_flow_full /= optical_flow_num; 
        if(optical_flow_full < 133)	
		{
			c5 = true;
		}
        optical_flow_full = sqrtf(optical_flow_full);
        optical_flow_nR /= optical_flow_num; 
        optical_flow_nR = sqrtf(optical_flow_nR);
		DSO = 0.75 * (0.02*optical_flow_full + 0.04*optical_flow_nR);
        c4 = DSO>1.0f;
    }
	else
	{
		const SE3 T_c_r_full(new_frame_->GetPoseSE3() * mpLastKeyFrame->GetPoseInverseSE3());   
		float optical_flow_full = 0; 
        size_t optical_flow_num = 0;
		std::list<vihso::Feature*> lfts = mpLastKeyFrame->GetFeatures();
        for(auto pft:lfts)
        {
            if(pft->point == NULL) continue;
            Vector3d P_w = pft->point->GetWorldPos();
            Vector3d O_l = mpLastKeyFrame->GetCameraCenter();
            Vector3d p_ref(pft->f * (P_w - O_l).norm());
            Vector3d p_cur_full(T_c_r_full * p_ref);
            Vector2d uv_cur_full(new_frame_->cam_->world2cam(p_cur_full));
            optical_flow_full += (uv_cur_full - pft->px).norm();
            optical_flow_num++;
        }
        optical_flow_full /= optical_flow_num;
		Eigen::Vector3d vecAngle = T_c_r_full.unit_quaternion().matrix().eulerAngles(2,1,0).transpose()*180.0/vihso::PI;
		double x=vecAngle.x(),y=vecAngle.y(),z=vecAngle.z();
		if(vecAngle.x()>100)	x = 180-vecAngle.x();
		if(vecAngle.x()<-100)	x= 180+vecAngle.x();
		if(vecAngle.y()>100)	y = 180-vecAngle.y();
		if(vecAngle.y()<-100)	y= 180+vecAngle.y();
		if(vecAngle.z()>100)	z = 180-vecAngle.z();
		if(vecAngle.z()<-100)	z= 180+vecAngle.z();
		vecAngle = Vector3d(x,y,z);
        c4 = optical_flow_full>40.0f;	
    }
	if(c2)
	{
		return true;
	}
	if(c3)
	{
		return true;
	}
	if(c4)
	{
		return true;
	}
    return false;
}
void FrameHandlerMono::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper = pLocalMapper;
}
void FrameHandlerMono::SetDepthFilter(DepthFilter *pDepthFilter)
{
    mpDepthFilter = pDepthFilter;
    reprojector_.depth_filter_ = pDepthFilter;
}
} // namespace vihso
