#include <stdexcept>
#include <vihso/frame.h>
#include <vihso/feature.h>
#include <vihso/point.h>
#include <vihso/config.h>
#include <boost/bind.hpp>
#include <fast/fast.h>
#include "vihso/PhotomatricCalibration.h"
#include "vihso/vikit/math_utils.h"
#include "vihso/vikit/vision.h"

using namespace cv;
namespace vihso {

int Frame::frame_counter_ = 0;
int Frame::keyFrameCounter_ = 0;

Frame::Frame(hso::AbstractCamera* cam, const cv::Mat& img, double timestamp, PhotomatricCalibration* opc) :
             id_(frame_counter_++), mbKF(false), mTimeStamp(timestamp), 
             cam_(cam), key_pts_(5), v_kf_(NULL), gradMean_(0)
{
    if(opc != NULL) m_pc = opc;
    initFrame(img);
}

Frame::Frame(hso::AbstractCamera* cam, const cv::Mat& img, double timestamp,  FramePtr pPrevF, const IMU::Calib &ImuCalib, PhotomatricCalibration* opc) :
            id_(frame_counter_++), mbKF(false), mTimeStamp(timestamp), cam_(cam), key_pts_(5), v_kf_(NULL), gradMean_(0),
            mbImuPreintegrated(false), bImu(false), m_last_frame(pPrevF), mpLastKeyFrame(NULL),mpNextKeyFrame(NULL), mpImuPreintegratedFrame(NULL),mpImuPreintegrated(NULL),
            mpcpi(NULL), mnBALocalForKF(0), mnBAFixedForKF(0),mbFirstConnection(true), mpParent(NULL)
{
    if(opc != NULL) m_pc = opc;
    initFrame(img);
    mImuCalib = ImuCalib;   
}

Frame::~Frame()
{
    std::for_each(fts_.begin(), fts_.end(), [&](Feature* i)
    {
        if(i->m_prev_feature != NULL)
        {
            assert(i->m_prev_feature->frame->isKeyframe());
            i->m_prev_feature->m_next_feature = NULL;
            i->m_prev_feature = NULL;
        }
        if(i->m_next_feature != NULL)
        {
            i->m_next_feature->m_prev_feature = NULL;
            i->m_next_feature = NULL;
        }
        delete i; i=NULL;
    });
    img_pyr_.clear();
    grad_pyr_.clear();
    sobelX_.clear();
    sobelY_.clear();
    canny_.clear();
    m_pyr_raw.clear();
}
void Frame::initFrame(const cv::Mat& img)   
{
    cv::Mat imgdist = img.clone();
    if(img.empty() || img.type() != CV_8UC1 )
        throw std::runtime_error("Frame: provided image is empty or type error");
    if(img.cols != cam_->width() || img.rows != cam_->height())
    {
        resize(img, imgdist, Size(cam_->width(), cam_->height()));
    }
    std::for_each(key_pts_.begin(), key_pts_.end(), [&](Feature* ftr){ ftr=NULL; });
    frame_utils::createImgPyramid(imgdist, max(Config::nPyrLevels(), Config::kltMaxLevel()+1), img_pyr_);   
    prepareForFeatureDetect();    
}
void Frame::setKeyframe()
{
    mbKF = true;
    setKeyPoints();
    keyFrameId_ = keyFrameCounter_++;
}
void Frame::addFeature(Feature* ftr)
{
    std::unique_lock<mutex> lock(mMutexFeatures);
    fts_.push_back(ftr);
}
void Frame::addTempFeature(Feature* ftr)
{
    std::unique_lock<mutex> lock(mMutexFeatures);
    temp_fts_.push_back(ftr);
}
void Frame::getFeaturesCopy(Features& list_copy)
{
    std::unique_lock<mutex> lock(mMutexFeatures);
    for(auto it = fts_.begin(); it != fts_.end(); ++it)
        list_copy.push_back(*it);
}
void Frame::setKeyPoints()
{
  for(size_t i = 0; i < 5; ++i) 
    if(key_pts_[i] != NULL)
      if(key_pts_[i]->point == NULL)
        key_pts_[i] = NULL;
  std::for_each(fts_.begin(), fts_.end(), [&](Feature* ftr){ if(ftr->point != NULL) checkKeyPoints(ftr); });    
}
void Frame::checkKeyPoints(Feature* ftr)
{
  const int cu = cam_->width()/2;   
  const int cv = cam_->height()/2;
  const Vector2d uv = ftr->px;      
  if(key_pts_[0] == NULL)
    key_pts_[0] = ftr;
  else if(std::max(std::fabs(ftr->px[0]-cu), std::fabs(ftr->px[1]-cv))  
          < std::max(std::fabs(key_pts_[0]->px[0]-cu), std::fabs(key_pts_[0]->px[1]-cv)))
    key_pts_[0] = ftr;
  if(uv[0] >= cu && uv[1] >= cv)
  {
    if(key_pts_[1] == NULL)
      key_pts_[1] = ftr;
    else if((uv[0] - cu) * (uv[1] - cv)
        >(key_pts_[1]->px[0] - cu) * (key_pts_[1]->px[1] - cv))
      key_pts_[1] = ftr;
  }
  if(uv[0] >= cu && uv[1] < cv)
  {
    if(key_pts_[2] == NULL)
      key_pts_[2] = ftr;
    else if((uv[0] - cu) * -(uv[1] - cv)
        >(key_pts_[2]->px[0] - cu) * -(key_pts_[2]->px[1] - cv))
      key_pts_[2] = ftr;
  }
  if(uv[0] < cu && uv[1] >= cv)
  {
    if(key_pts_[3] == NULL)
      key_pts_[3] = ftr;
    else if(-(uv[0] - cu) * (uv[1] - cv)
        >-(key_pts_[3]->px[0] - cu) * (key_pts_[3]->px[1] - cv))
      key_pts_[3] = ftr;
  }
  if(uv[0] < cu && uv[1] < cv)
  {
    if(key_pts_[4] == NULL)
      key_pts_[4] = ftr;
    else if(-(uv[0] - cu) * -(uv[1] - cv)
        >-(key_pts_[4]->px[0] - cu) * -(key_pts_[4]->px[1] - cv))
      key_pts_[4] = ftr;
  }
}
void Frame::removeKeyPoint(Feature* ftr)
{
    bool found = false;
    std::for_each(key_pts_.begin(), key_pts_.end(), [&](Feature*& i){
    if(i == ftr) {
        i = NULL;
        found = true;
    }
    });
    if(found) setKeyPoints();
}
bool Frame::isVisible(const Vector3d& point_w)
{
    Sophus::SE3 Tcw = GetPoseSE3();
    Vector3d xyz_f = Tcw * point_w;
    if(xyz_f.z() < 0.0) return false; 
    Vector2d px = f2c(xyz_f);
    if(px[0] >= 0.0 && px[1] >= 0.0 && px[0] < cam_->width() && px[1] < cam_->height())
        return true;
    return false;
}
void Frame::prepareForFeatureDetect()
{   
    sobelX_.resize(Config::nPyrLevels());
    sobelY_.resize(Config::nPyrLevels());
    assert(Config::nPyrLevels() == 3);
    for(int i = 0; i < 3; ++i)  
    {
        cv::Sobel(img_pyr_[i], sobelX_[i], CV_16S, 1, 0, 5, 1, 0, BORDER_REPLICATE);    
        cv::Sobel(img_pyr_[i], sobelY_[i], CV_16S, 0, 1, 5, 1, 0, BORDER_REPLICATE);    
    }
    float intSum = 0, gradSum = 0;
    int sum = 0;
    for(int y=16;y<img_pyr_[0].rows-16;y++)
        for(int x=16;x<img_pyr_[0].cols-16;x++) 
        {
            sum++;
            float gradx = sobelX_[0].at<short>(y,x);
            float grady = sobelY_[0].at<short>(y,x);
            gradSum += sqrtf(gradx*gradx + grady*grady);
            intSum += img_pyr_[0].ptr<uchar>(y)[x]; 
        }
    integralImage_ = intSum/sum;    
    gradMean_ = gradSum/sum;        
    gradMean_ /= 30;
    if(gradMean_ > 20) gradMean_ = 20;
    if(gradMean_ < 7)  gradMean_ = 7;
}

void Frame::finish()
{
    grad_pyr_.clear();
    canny_.clear();   
}
Vector2d Frame::w2c(const Vector3d& xyz_w)  
{
    unique_lock<mutex> lock(mMutexPose); 
    return cam_->world2cam( T_f_w_ * xyz_w ); 
}
Vector3d Frame::w2f(const Vector3d& xyz_w)
{ 
    unique_lock<mutex> lock(mMutexPose); 
    return T_f_w_ * xyz_w; 
}
Vector3d Frame::c2f(const Vector2d& px)
{ 
    return cam_->cam2world(px[0], px[1]); 
}
Vector3d Frame::c2f(const double x, const double y)
{
    return cam_->cam2world(x, y); 
}
Vector3d Frame::f2w(const Vector3d& f)
{ 
    unique_lock<mutex> lock(mMutexPose); 
    return T_f_w_.inverse() * f; 
}
Vector2d Frame::f2c(const Vector3d& f)
{ 
    return cam_->world2cam( f ); 
}
void Frame::SetNewBias(const IMU::Bias &b)
{
    mImuBias = b;
    if(mpImuPreintegrated)
        mpImuPreintegrated->SetNewBias(b);
}
void Frame::SetVelocity(const Eigen::Vector3d &Vwb)
{
    mVw = Vwb;
}
void Frame::setIntegrated()
{
    unique_lock<std::mutex> lock(mpMutexImu);
    mbImuPreintegrated = true;
}
Eigen::Matrix4d Frame::GetPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return T_f_w_.matrix();
}
Sophus::SE3 Frame::GetPoseSE3()
{
    unique_lock<mutex> lock(mMutexPose);
    return T_f_w_;
}
Eigen::Matrix4d Frame::GetPoseInverse()
{
    unique_lock<mutex> lock(mMutexPose);
    return T_f_w_.inverse().matrix();
}
Sophus::SE3 Frame::GetPoseInverseSE3()
{
    unique_lock<mutex> lock(mMutexPose);
    return T_f_w_.inverse();
}
Eigen::Vector3d Frame::GetCameraCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return T_f_w_.inverse().translation();
}
Eigen::Vector3d Frame::GetImuPosition()
{
    unique_lock<mutex> lock(mMutexPose);
    return (mImuCalib.SE3_Tbc * T_f_w_).inverse().translation();
}
Eigen::Matrix3d Frame::GetImuRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return (mImuCalib.SE3_Tbc * T_f_w_).inverse().rotation_matrix();
}
Eigen::Matrix4d Frame::GetImuPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return (mImuCalib.SE3_Tbc * T_f_w_).inverse().matrix();
}
Eigen::Matrix3d Frame::GetRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return T_f_w_.rotation_matrix();
}
Eigen::Vector3d Frame::GetTranslation()
{
    unique_lock<mutex> lock(mMutexPose);
    return T_f_w_.translation();
}
Eigen::Vector3d Frame::GetVelocity()
{
    unique_lock<mutex> lock(mMutexPose);
    return mVw;
}
void Frame::SetPose(const Sophus::SE3 &Tcw)
{
    unique_lock<mutex> lock(mMutexPose);
    T_f_w_ = Tcw;
}
void Frame::SetImuPoseVelocity(const Eigen::Matrix3d &Rwb, const Eigen::Vector3d &twb, const Eigen::Vector3d &Vwb)
{
    mVw = Vwb;
    Sophus::SE3 Twb(Rwb, twb);
    Sophus::SE3 Tbw = Twb.inverse();
    T_f_w_ = mImuCalib.SE3_Tcb * Tbw;
}
IMU::Bias Frame::GetImuBias()
{
    unique_lock<mutex> lock(mMutexPose);
    return mImuBias;
}
Eigen::Vector3d Frame::GetGyroBias()
{
    unique_lock<mutex> lock(mMutexPose);
    return Eigen::Vector3d(mImuBias.bwx, mImuBias.bwy, mImuBias.bwz);
}
Eigen::Vector3d Frame::GetAccBias()
{
    unique_lock<mutex> lock(mMutexPose);
    return Eigen::Vector3d(mImuBias.bax, mImuBias.bay, mImuBias.baz);
}
Features Frame::GetFeatures()
{
    std::unique_lock<mutex> lock(mMutexFeatures);
	if(temp_fts_.empty())
    	return fts_;
	else
	{
		Features allfts;
		for(Features::iterator it=fts_.begin(); it!=fts_.end(); it++)
			allfts.push_back(*it);
		for(Features::iterator it=temp_fts_.begin(); it!=temp_fts_.end(); it++)
			allfts.push_back(*it);
		return allfts;
	}
}
void Frame::UpdateFeatures()
{
}
double Frame::GetMaxDepth()
{
	double MaxDepth=0.0;
	list<vihso::Feature *> lfts = GetFeatures();
	for(list<Feature*>::iterator lit=lfts.begin(), litend=lfts.end(); lit!=litend; lit++)
	{
		if((*lit)->point == NULL) continue;
		if((*lit)->point->isBad_) continue;
		double depth = (*lit)->point->GetWorldPos().norm();
		if(depth > MaxDepth)
			MaxDepth = depth;
	}
	return MaxDepth;
}
void Frame::AddConnection(Frame *pKF, const int &weight)
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(!mConnectedKeyFrameWeights.count(pKF)) 
            mConnectedKeyFrameWeights[pKF]=weight;
        else if(mConnectedKeyFrameWeights[pKF]!=weight) 
            mConnectedKeyFrameWeights[pKF]=weight;
        else
            return;
    }
    UpdateBestCovisibles();
}
void Frame::UpdateBestCovisibles()
{
    unique_lock<mutex> lock(mMutexConnections);
    vector<pair<int,Frame*> > vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    for(map<Frame*,int>::iterator mit=mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
       vPairs.push_back(make_pair(mit->second,mit->first));
    sort(vPairs.begin(),vPairs.end());
    list<Frame*> lKFs; 	
    list<int> lWs; 		
    for(size_t i=0, iend=vPairs.size(); i<iend;i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }
    mvpOrderedConnectedKeyFrames = vector<Frame*>(lKFs.begin(),lKFs.end());
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());
}
void Frame::UpdateConnections()
{   
    std::map<Frame*, int> KFcounter;
    Features lfts = GetFeatures();
    int nPoints = 0;
    for(Features::iterator it = lfts.begin(); it != lfts.end(); ++it)
    {
        if((*it)->point == NULL) 
            continue;
        list<Feature*> observations = (*it)->point->GetObservations();
        for(auto ite = observations.begin(); ite != observations.end(); ++ite)
        {
            if((*ite)->frame->id_== id_) continue;
            KFcounter[(*ite)->frame]++;
        }
        nPoints++;
    }
    if(KFcounter.empty()) return;
    int nmax=0;         
    Frame* pKFmax=NULL; 
    const int th = nPoints > 30? 5 : 3;
    vector< pair<int, Frame*> > vPairs; 
    vPairs.reserve(KFcounter.size());  
    for(std::map<Frame*,int>::iterator mit=KFcounter.begin(), mend=KFcounter.end(); mit!=mend; mit++)
    {
        if(mit->second>nmax) 
        {
            nmax=mit->second;
            pKFmax=mit->first;
        }
        if(mit->second>=th)
		{
			vPairs.push_back(make_pair(mit->second,mit->first));
			(mit->first)->AddConnection(this,mit->second);
		}
        if(mit->first->keyFrameId_+5 < keyFrameId_)   
        {
            if(!mit->first->sobelX_.empty())
            {
                mit->first->sobelX_.clear();
                mit->first->sobelY_.clear();
            }
        }	
    }
    if(vPairs.empty())  
	{
		vPairs.push_back(make_pair(nmax,pKFmax));
		pKFmax->AddConnection(this,nmax);
    }
    sort(vPairs.begin(),vPairs.end());
    list<Frame*> lKFs;
    list<int> lWs;
    for(size_t i=0; i<vPairs.size();i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }
	{
        unique_lock<mutex> lockCon(mMutexConnections);
        mConnectedKeyFrameWeights = KFcounter;
        mvpOrderedConnectedKeyFrames = vector<Frame*>(lKFs.begin(),lKFs.end());
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());
        if(mbFirstConnection && keyFrameId_!=0)
        {
            mpParent = mvpOrderedConnectedKeyFrames.front();
            mpParent->AddChild(this);
            mbFirstConnection = false;
        }
    }
}
set<Frame*> Frame::GetConnectedKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    set<Frame*> s;
    for(map<Frame*,int>::iterator mit=mConnectedKeyFrameWeights.begin();mit!=mConnectedKeyFrameWeights.end();mit++)
        s.insert(mit->first);
    return s;
}
vector<Frame*> Frame::GetVectorCovisibleKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}
vector<Frame*> Frame::GetBestCovisibilityKeyFrames(const int &N)
{
    unique_lock<mutex> lock(mMutexConnections);
    if((int)mvpOrderedConnectedKeyFrames.size()<N)
        return mvpOrderedConnectedKeyFrames;
    else
        return vector<Frame*>(mvpOrderedConnectedKeyFrames.begin(),mvpOrderedConnectedKeyFrames.begin()+N);
}
vector<Frame*> Frame::GetCovisiblesByWeight(const int &w)
{
    unique_lock<mutex> lock(mMutexConnections);
    if(mvpOrderedConnectedKeyFrames.empty())
    {
        return vector<Frame*>();
    }
    vector<int>::iterator it = upper_bound( mvOrderedWeights.begin(),   
                                            mvOrderedWeights.end(),     
                                            w,                          
                                            frame_utils::weightComp);	
    if(it==mvOrderedWeights.end() && mvOrderedWeights.back() < w)
    {
        return vector<Frame*>();
    }
    else
    {
        int n = it-mvOrderedWeights.begin();
        return vector<Frame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin()+n);
    }
}
int Frame::GetWeight(Frame *pKF)
{
    unique_lock<mutex> lock(mMutexConnections);
    if(mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}
void Frame::AddChild(Frame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}
void Frame::EraseChild(Frame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}
void Frame::ChangeParent(Frame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    if(pKF == this)
    {
        cout << "ERROR: Change parent KF, the parent and child are the same KF" << endl;
        throw std::invalid_argument("The parent and child can not be the same");
    }
    mpParent = pKF;
    pKF->AddChild(this);
}
set<Frame*> Frame::GetChilds()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens;
}
Frame* Frame::GetParent()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mpParent;
}
bool Frame::hasChild(Frame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}
void Frame::SetFirstConnection(bool bFirst)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbFirstConnection = bFirst;
}
namespace frame_utils 
{
void createImgPyramid(const cv::Mat& img_level_0, int n_levels, ImgPyr& pyr)
{
    pyr.resize(n_levels);
    pyr[0] = img_level_0;   
    for(int i=1; i<n_levels; ++i)
    {
        if(img_level_0.cols % 16 == 0 && img_level_0.rows % 16 == 0)    
        {
            pyr[i] = cv::Mat(pyr[i-1].rows/2, pyr[i-1].cols/2, CV_8U);
            hso::halfSample(pyr[i-1], pyr[i]);  
        }
        else
        {
            float scale = 1.0/(1<<i);   
            cv::Size sz(cvRound((float)img_level_0.cols*scale), cvRound((float)img_level_0.rows*scale));
            cv::resize(pyr[i-1], pyr[i], sz, 0, 0, cv::INTER_LINEAR);
        }     
    }
}
void createImgGrad(const ImgPyr& pyr_img, ImgPyr& scharr, int n_levels)
{ 
    scharr.resize(n_levels);
    for(int i = 0; i < n_levels; ++i)
        hso::calcSharrDeriv(pyr_img[i], scharr[i]); 
}
bool getSceneDepth(Frame& frame, double& depth_mean, double& depth_min)
{
    vector<double> depth_vec;
    depth_vec.reserve(frame.fts_.size());
    depth_min = std::numeric_limits<double>::max(); 
    for(auto it=frame.fts_.begin(), ite=frame.fts_.end(); it!=ite; ++it)    
    {
        if((*it)->point != NULL)
        {
            const Vector3d pose = (*it)->point->GetWorldPos();
            const double z = frame.w2f(pose).z();
            depth_vec.push_back(z);
            depth_min = fmin(z, depth_min);
        }
    }
    if(depth_vec.empty())
    {
        VIHSO_WARN_STREAM("Cannot set scene depth. Frame has no point-observations!");
        return false;
    }
    depth_mean = hso::getMedian(depth_vec);
    return true;
}
bool getSceneDistance(Frame& frame, double& distance_mean)
{
    vector<double> distance_vec;
    distance_vec.reserve(frame.fts_.size());
    for(auto& ft: frame.fts_)
    {
        if(ft->point == NULL) continue;
        const Vector3d pose = ft->point->GetWorldPos();
        const double distance = frame.w2f(pose).norm();
        distance_vec.push_back(distance);
    }
    if(distance_vec.empty())
    {
        VIHSO_WARN_STREAM("Cannot set scene distance. Frame has no point-observations!");
        return false;
    }
    distance_mean = hso::getMedian(distance_vec);
    return true;
}
void createIntegralImage(const cv::Mat& image, float& integralImage)
{
    float sum = 0;
    int num = 0;
    int height = image.rows;
    int weight = image.cols;
    for(int y=8;y<height-8;y++)
        for(int x=8;x<weight-8;x++)
        {
            sum += image.ptr<uchar>(y)[x];
            num++; 
        }
    integralImage = sum/num;
}
bool frameCovisibilityComparator(pair<int, Frame*>& lhs, pair<int, Frame*>& rhs)
{
    if(lhs.first != rhs.first)
        return (lhs.first > rhs.first);
    else
        return (lhs.second->id_ < rhs.second->id_);
}
bool weightComp( int a, int b)
{
	return a>b;
}
} 
} 
