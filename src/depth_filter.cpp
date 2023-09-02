#include <algorithm>
#include <numeric>
#include <boost/bind.hpp>
#include <boost/math/distributions/normal.hpp>
#include <vihso/global.h>
#include <vihso/depth_filter.h>
#include <vihso/frame.h>
#include <vihso/point.h>
#include <vihso/feature.h>
#include <vihso/matcher.h>
#include <vihso/config.h>
#include <vihso/feature_detection.h>
#include <vihso/IndexThreadReduce.h>
#include <vihso/matcher.h>
#include <vihso/feature_alignment.h>
#include <vihso/bundle_adjustment.h>
#include "vihso/vikit/robust_cost.h"
#include "vihso/vikit/math_utils.h"
#include <time.h>
#include <cmath>
namespace vihso {
int Seed::batch_counter = 0;    
int Seed::seed_counter = 0;     
Seed::Seed(Feature* ftr, float depth_mean, float depth_min, float converge_threshold) :
    batch_id(batch_counter),
    id(seed_counter++),
    ftr(ftr),
    a(10),
    b(10),
    mu(1.0/depth_mean),
    z_range(1.0/depth_min),
    sigma2(z_range*z_range/36),
    isValid(true),
    eplStart(Vector2i(0,0)),
    eplEnd(Vector2i(0,0)),
    haveReprojected(false),
    temp(NULL)
{
    vec_distance.push_back(depth_mean);
    vec_sigma2.push_back(sigma2);
    converge_thresh = converge_threshold;
}
DepthFilter::DepthFilter( feature_detection::FeatureExtractor* featureExtractor, callback_t seed_converged_cb) :
    featureExtractor_(featureExtractor),
    seed_converged_cb_(seed_converged_cb),
    seeds_updating_halt_(false),
    thread_(NULL),
    new_keyframe_set_(false),
    new_keyframe_min_depth_(0.0),
    new_keyframe_mean_depth_(0.0),
    px_error_angle_(-1)
{
    frame_prior_.resize(100000);
    threadReducer_ = new lsd_slam::IndexThreadReduce();
    runningStats_ = new RunningStats();
    n_update_last_ = 100;
    n_pre_update_ = 0;
    n_pre_try_ = 0;
    nPonits = 1;
    nSkipFrame = 0;
    nMeanConvergeFrame_ = 6;
    convergence_sigma2_thresh_ = 200;
}
DepthFilter::~DepthFilter()
{
    stopThread();
}
void DepthFilter::startThread()
{
    thread_ = new boost::thread(&DepthFilter::updateSeedsLoop, this);
}
void DepthFilter::stopThread()
{
    if(thread_ != NULL)
    {
        seeds_updating_halt_ = true;
        thread_->interrupt();
        thread_->join();
        thread_ = NULL;
    }
    delete threadReducer_;
    delete runningStats_;
}
void DepthFilter::addFrame(FramePtr frame)
{
    if(thread_ != NULL)
    {
        {
            lock_t lock(frame_queue_mut_);
            if(frame_queue_.size() > 2) 
                frame_queue_.pop();
            frame_queue_.push(frame);
        }
        seeds_updating_halt_ = false;   
        frame_queue_cond_.notify_one();
    }
    else
        updateSeeds(frame);
}
void DepthFilter::addKeyframe(FramePtr frame, double depth_mean, double depth_min, float converge_thresh)
{
    new_keyframe_min_depth_ = depth_min;           
    new_keyframe_mean_depth_ = depth_mean;         
    convergence_sigma2_thresh_ = converge_thresh;  
    if(thread_ != NULL)
    {
        new_keyframe_ = frame;
        new_keyframe_set_ = true;
        seeds_updating_halt_ = true;    
        frame_queue_cond_.notify_one(); 
    }
    else
        initializeSeeds(frame);
}
void DepthFilter::removeKeyframe(FramePtr frame)
{
  seeds_updating_halt_ = true;
  lock_t lock(seeds_mut_);
  std::list<Seed>::iterator it=seeds_.begin();
  size_t n_removed = 0;
  while(it!=seeds_.end())
  {
    if(it->ftr->frame == frame.get())
    {
      it = seeds_.erase(it);
      ++n_removed;
    }
    else
      ++it;
  }
  seeds_updating_halt_ = false;
}
void DepthFilter::reset()
{
  seeds_updating_halt_ = true;
  {
    lock_t lock(seeds_mut_);
    seeds_.clear();
  }
  lock_t lock();
  while(!frame_queue_.empty())
    frame_queue_.pop();
  seeds_updating_halt_ = false;
}
void DepthFilter::initializeSeeds(FramePtr frame)
{
    boost::unique_lock<boost::mutex> lock_c(detector_mut_);
    Features new_features;
    featureExtractor_->setExistingFeatures(frame->fts_);
	int nExtractorFts=0;
	if(frame->m_n_inliers < 100) 
		nExtractorFts = 700;
	else if(frame->m_n_inliers < 150)
		nExtractorFts = 500;
    featureExtractor_->detect(frame.get(), 20, frame->gradMean_, new_features, nExtractorFts, frame->m_last_frame.get());  
    lock_c.unlock();
    seeds_updating_halt_ = true;
    lock_t lock(seeds_mut_); 
    ++Seed::batch_counter;
    std::for_each(new_features.begin(), new_features.end(), [&](Feature* ftr)
    {
        Seed seed(ftr, new_keyframe_mean_depth_, new_keyframe_min_depth_, convergence_sigma2_thresh_);
        for(auto it = frame_prior_[Seed::batch_counter-1].begin(); it != frame_prior_[Seed::batch_counter-1].end(); ++it)
        {
            seed.pre_frames.push_back(*it);
        }
        seeds_.push_back(seed);
    });
    seeds_updating_halt_ = false;
    frame->finish();
}
void DepthFilter::updateSeedsLoop()
{
    while(!boost::this_thread::interruption_requested())    
    {
        FramePtr frame;
        {
            lock_t lock(frame_queue_mut_);
            if(seeds_.empty())
            {
                while(frame_queue_.empty() && new_keyframe_set_ == false)
                    frame_queue_cond_.wait(lock);
            }
            else
            {
                std::list<Seed>::iterator it=seeds_.begin();
                while(frame_queue_.empty() && new_keyframe_set_ == false && it != seeds_.end())
                {
                    observeDepthWithPreviousFrameOnce(it);
                    it++;
                }
            }
            if(!frame_queue_.empty() || new_keyframe_set_)
            {
                if(new_keyframe_set_)  
                {
                    new_keyframe_set_ = false;
                    seeds_updating_halt_ = false;
                    clearFrameQueue();
                    frame = new_keyframe_;
                }
                else   
                {
                    frame = frame_queue_.front();
                    frame_queue_.pop();
                }
            }
            else 
            {
                while(frame_queue_.empty() && new_keyframe_set_ == false)
                    frame_queue_cond_.wait(lock);
                if(new_keyframe_set_)
                {
                    new_keyframe_set_ = false;
                    seeds_updating_halt_ = false;
                    clearFrameQueue();
                    frame = new_keyframe_;
                }
                else
                {
                    frame = frame_queue_.front();
                    frame_queue_.pop();
                }
            }
            n_pre_update_ = 0;
            n_pre_try_ = 0;
        }
        updateSeeds(frame);
        if(frame->isKeyframe())
        {
            initializeSeeds(frame); 
			if(frame->m_n_inliers < 100)
			{
				FramePtr pF = frame->mpLastKeyFrame;
				if(pF == NULL) break;
				if(pF->sobelX_.empty()) break;
				int m = 3;
				int avglier = 0;
				double Mcrid = 0.0;
				for(int i=0; i<m; i++)
				{
					int n0 = pF->nObs(); 
					if(n0>200) n0=200;
					avglier += n0;
					Mcrid += n0/200.0;
					pF = pF->mpLastKeyFrame;
					if(pF == NULL) break;
					if(pF->sobelX_.empty()) break;
				}
				avglier = avglier/3;
				if(avglier < 120)
				{
					Mcrid = Mcrid/3.0; 
					int nf = floor((Mcrid)*2.82);
					pF = frame->mpLastKeyFrame;
					for(int i=0; i<nf; i++)
					{
						updateSeeds(pF, false);
						pF = pF->mpLastKeyFrame;
						if(pF == NULL) break;
						if(pF->sobelX_.empty()) break;
					}	
				}	
			}								
        }
	}
}
void DepthFilter::updateSeeds(FramePtr frame, bool bProcessFramePrior)
{
	if(bProcessFramePrior)
	{
		if(!frame->isKeyframe())
			frame_prior_[Seed::batch_counter].push_front(frame);  
		else
			frame_prior_[Seed::batch_counter+1].push_front(frame); 
		if(Seed::batch_counter > 5 && frame->isKeyframe() && frame->m_pc == NULL)
		{
			list<FramePtr>::iterator it = frame_prior_[Seed::batch_counter-5].begin();
			while(it != frame_prior_[Seed::batch_counter-5].end())
			{
				Frame* dframe = (*it).get();
				if(!dframe->isKeyframe())
				{
					delete dframe;
					dframe = NULL;
				}
				++it;
			}
		}
	}
    active_frame_ = frame;
    lock_t lock(seeds_mut_);
    if(this->px_error_angle_ == -1)
    {
        const double focal_length = frame->cam_->errorMultiplier2();    
        double px_noise = 1.0;                                          
        double px_error_angle = atan(px_noise/(2.0*focal_length))*2.0;
        this->px_error_angle_ = px_error_angle;
    }
	std::list<Seed>::iterator it=seeds_.begin();
	if(bProcessFramePrior)
	{
		while(it!=seeds_.end())
		{
			if(seeds_updating_halt_)
				return;
			if((Seed::batch_counter - it->batch_id) > 3)	
			{
				assert(it->ftr->point == NULL);    
				if(it->temp != NULL && it->haveReprojected) 
				{   
					it->temp->seedStates_ = -1;
				}
				else
				{
					delete it->ftr;
					it->ftr = NULL;
				}
				it->pre_frames.clear();
				it->optFrames_P.clear();
				it->optFrames_A.clear();
				it = seeds_.erase(it);
				continue;
			}
			it++;
		}
	}
    observeDepth();
    it = seeds_.begin();
	int nConvergeSeeds=0, nInvalidSeeds=0;
    while(it!=seeds_.end())
    {
        if(seeds_updating_halt_)
            return;
        if(sqrt(it->sigma2) < it->z_range/it->converge_thresh)
        {
            assert(it->ftr->point == NULL); 
            bool isValid = true;
            bool isEnough = false;  
            if(activatePoint(*it, isValid, isEnough))
                it->mu = it->opt_id;       
            Vector3d pHost = it->ftr->f * (1.0/it->mu);
            if(it->mu < 1e-10 || pHost[2] < 1e-10)  isValid = false;
            if(!isValid)
            {   
                if(it->temp != NULL && it->haveReprojected)
                    it->temp->seedStates_ = -1;
                it = seeds_.erase(it);
                continue;
            }
            {
                if(m_v_n_converge.size() > 1000)
                    m_v_n_converge.erase(m_v_n_converge.begin());
                m_v_n_converge.push_back(it->vec_distance.size());
            }
            if(isEnough)
            {
                Vector3d xyz_world = it->ftr->frame->GetPoseInverseSE3() * pHost;
                Point* point = new Point(xyz_world, it->ftr);   
                point->SetIdist(it->mu);
                point->hostFeature_ = it->ftr;
                point->color_ = it->ftr->frame->img_pyr_[0].at<uchar>((int)it->ftr->px[1], (int)it->ftr->px[0]);   
                if(it->ftr->type == Feature::EDGELET)
                    point->ftr_type_ = Point::FEATURE_EDGELET;
                else if(it->ftr->type == Feature::CORNER)
                    point->ftr_type_ = Point::FEATURE_CORNER;
                else
                    point->ftr_type_ = Point::FEATURE_GRADIENT;
                it->ftr->point = point;
                if(it->temp != NULL && it->haveReprojected)
                    it->temp->seedStates_ = 1;
                else
                    assert(it->temp == NULL && !it->haveReprojected);
                it->pre_frames.clear();
                it->optFrames_P.clear();
                it->optFrames_A.clear();
                seed_converged_cb_(point, it->sigma2); 
                it = seeds_.erase(it);
				nConvergeSeeds++;
            }
            else
                ++it;
        }
        else if(!it->isValid)
        {
            VIHSO_WARN_STREAM("z_min is NaN");
            it = seeds_.erase(it);
			nInvalidSeeds++;
        }
        else
            ++it;
    }
    lock_t lock_converge(mean_mutex_);
    if(m_v_n_converge.size() > 500)
        nMeanConvergeFrame_ = std::accumulate(m_v_n_converge.begin(), m_v_n_converge.end(), 0) / m_v_n_converge.size();
    else
        nMeanConvergeFrame_ = 6;
}
void DepthFilter::clearFrameQueue()
{
  while(!frame_queue_.empty())
    frame_queue_.pop();
}
void DepthFilter::getSeedsCopy(const FramePtr& frame, std::list<Seed>& seeds)
{
  lock_t lock(seeds_mut_);
  for(std::list<Seed>::iterator it=seeds_.begin(); it!=seeds_.end(); ++it)
  {
    if (it->ftr->frame == frame.get())
      seeds.push_back(*it);
  }
}
#define UNZERO(val) (val < 0 ? (val > -1e-10 ? -1e-10 : val) : (val < 1e-10 ? 1e-10 : val))
void DepthFilter::updateSeed(const float x, const float tau2, Seed* seed)
{
    float id_var = seed->sigma2*1.01f;  
    float w = tau2 / (tau2 + id_var);  
    float new_idepth = (1-w)*x + w*seed->mu;   
    seed->mu = UNZERO(new_idepth);
    id_var *= w;   
    if(id_var < seed->sigma2) seed->sigma2 = id_var;
}
double DepthFilter::computeTau(	const SE3& T_ref_cur,
								const Vector3d& f,
								const double z,
								const double px_error_angle)
{
    Vector3d t(T_ref_cur.translation());
    Vector3d a = f*z-t;
    double t_norm = t.norm();
    double a_norm = a.norm();
    double alpha = acos(f.dot(t)/t_norm);
    double beta = acos(a.dot(-t)/(t_norm*a_norm)); 
    double beta_plus = beta + px_error_angle;
    double gamma_plus = PI-alpha-beta_plus; 
    double z_plus = t_norm*sin(beta_plus)/sin(gamma_plus); 
    return (z_plus - z); 
}
void DepthFilter::observeDepth()
{
	threadReducer_->reduce(boost::bind(&DepthFilter::observeDepthRow, this, _1, _2, _3), 0, (int)seeds_.size(), runningStats_, 10);
	runningStats_->n_seeds = seeds_.size();
	n_update_last_ = runningStats_->n_updates;
	runningStats_->setZero(); 
}
void DepthFilter::observeDepthRow(int yMin, int yMax, RunningStats* stats)
{
    if(seeds_updating_halt_) return;
    std::list<Seed>::iterator it=seeds_.begin();
    for(int i = 0; i < yMin; ++i) it++;
    for(int idx = yMin; idx < yMax; ++idx, ++it)
    {
        if(seeds_updating_halt_) return;
		SE3 T1w = it->ftr->frame->GetPoseSE3();
		SE3 Tw2 = active_frame_->GetPoseInverseSE3();
        SE3 T_ref_cur =  T1w * Tw2; 
        Vector3d xyz_f = T_ref_cur.inverse() * (1.0/it->mu * it->ftr->f);
        if(xyz_f.z() < 0.0) 
        {
            stats->n_out_views++;
            it->is_update = false;
            continue;
        }
        if(!active_frame_->cam_->isInFrame(active_frame_->f2c(xyz_f).cast<int>())) 
        {
            stats->n_out_views++;
            it->is_update = false;
            continue;
        }
        it->is_update = true;
        if((it->optFrames_A.size()<15) && (active_frame_->id_>it->ftr->frame->id_))	
        {
			it->optFrames_A.push_back(active_frame_);
		}
        float z_inv_min = it->mu + 2*sqrt(it->sigma2);                      
        float z_inv_max = max(it->mu - 2*sqrt(it->sigma2), 0.00000001f);    
        if(isnan(z_inv_min)) it->isValid = false;
        Matcher matcher;
        double z;    
        int res = matcher.doLineStereo(*it->ftr->frame, 
                                       *active_frame_, 
                                       *it->ftr,
                                        1.0/z_inv_min, 
                                        1.0/it->mu, 
                                        1.0/z_inv_max, 
                                        z, 
                                        it->eplStart, 
                                        it->eplEnd);
        if(res != 1)
        {
            it->b++; 
            it->eplStart = Vector2i(0,0);
            it->eplEnd   = Vector2i(0,0);
            stats->n_failed_matches++;
            if(res == -1)
                stats->n_fail_lsd++;
            else if(res == -2)
                stats->n_fail_triangulation++;
            else if(res == -3)
                stats->n_fail_alignment++;
            else if(res == -4)
                stats->n_fail_score++;
            continue;
        }
        double tau = computeTau(T_ref_cur, it->ftr->f, z, this->px_error_angle_);  
        double tau_inverse = 0.5 * (1.0/max(0.0000001, z-tau) - 1.0/(z+tau));       
        updateSeed(1./z, tau_inverse*tau_inverse, &*it);
        it->vec_distance.push_back(1.0/it->mu);
        it->vec_sigma2.push_back(it->sigma2);
        it->last_update_frame = active_frame_;
        it->last_matched_px = matcher.px_cur_;
        it->last_matched_level = matcher.search_level_;
        stats->n_updates++;
        if(active_frame_->isKeyframe())
        {
            boost::unique_lock<boost::mutex> lock(detector_mut_);
            featureExtractor_->setGridOccpuancy(matcher.px_cur_, it->ftr);
        }
    }
}
bool DepthFilter::observeDepthWithPreviousFrameOnce(std::list<Seed>::iterator& ite)
{
    if(ite->pre_frames.empty() || this->px_error_angle_ == -1)
        return false;
    FramePtr preFrame = *(ite->pre_frames.begin());
    assert(preFrame->id_ < ite->ftr->frame->id_);
    n_pre_try_++;
	SE3 Trw = ite->ftr->frame->GetPoseSE3();
	SE3 Twc = preFrame->GetPoseInverseSE3();
    SE3 T_ref_cur = Trw * Twc;
    Vector3d xyz_f = T_ref_cur.inverse()*(1.0/ite->mu * ite->ftr->f);
    if(xyz_f.z() < 0.0) 
    {
        ite->pre_frames.erase(ite->pre_frames.begin());
        return false;
    }
    if(!preFrame->cam_->isInFrame(preFrame->f2c(xyz_f).cast<int>()))
    {
        ite->pre_frames.erase(ite->pre_frames.begin());
        return false;
    }
    if(ite->optFrames_P.size() < 15)
        ite->optFrames_P.push_back(preFrame);
    float z_inv_min = ite->mu + 2*sqrt(ite->sigma2);
    float z_inv_max = max(ite->mu - 2*sqrt(ite->sigma2), 0.00000001f);
    double z;
    if(!matcher_.findEpipolarMatchPrevious(*ite->ftr->frame, *preFrame, *ite->ftr, 1.0/ite->mu, 1.0/z_inv_min, 1.0/z_inv_max, z))
    {
        ite->pre_frames.erase(ite->pre_frames.begin());
        return false;
    }
    n_pre_update_++;
    double tau = computeTau(T_ref_cur, ite->ftr->f, z, this->px_error_angle_);
    double tau_inverse = 0.5 * (1.0/max(0.0000001, z-tau) - 1.0/(z+tau));
    updateSeed(1./z, tau_inverse*tau_inverse, &*ite);
    ite->pre_frames.erase(ite->pre_frames.begin()); 
	return true;
}
bool DepthFilter::activatePoint(Seed& seed, bool& isValid, bool& isEnough)
{
    seed.opt_id = seed.mu;
    const int halfPatchSize = 4;
    const int patchSize = halfPatchSize*2;
    const int patchArea = patchSize*patchSize;
	const float ratioFactor = 1.5f*2.0;	
    Frame* host = seed.ftr->frame;            
    Vector3d pHost = seed.ftr->f*(1.0/seed.mu);
	Vector3d x3D = host->GetPoseInverseSE3() * pHost;	
    vector< pair<FramePtr, Vector2d> > targets; 
    targets.reserve(seed.optFrames_P.size()+seed.optFrames_A.size());
    for(size_t i = 0; i < seed.optFrames_P.size(); ++i)
    {
        FramePtr target = seed.optFrames_P[i];
		SE3 Ttw = target->GetPoseSE3();
		SE3 Twh = host->GetPoseInverseSE3();
        SE3 Tth = Ttw * Twh;
        Vector3d pTarget = Tth*pHost;           
        if(pTarget[2] < 0.0001) continue;
        Vector2d px(target->cam_->world2cam(pTarget));
        if(!target->cam_->isInFrame(px.cast<int>(), 8)) 
            continue;
        targets.push_back(make_pair(target, px));  
    }
    for(size_t i = 0; i < seed.optFrames_A.size(); ++i)
    {
        FramePtr target = seed.optFrames_A[i];
		SE3 Ttw = target->GetPoseSE3();
		SE3 Twh = host->GetPoseInverseSE3();
        SE3 Tth = Ttw * Twh;
        Vector3d pTarget = Tth*pHost;
        if(pTarget[2] < 0.0001) continue;
        Vector2d px(target->cam_->world2cam(pTarget));
        if(!target->cam_->isInFrame(px.cast<int>(), 8)) 
            continue;
        targets.push_back(make_pair(target, px));	
    }
    float n_frame_thresh = nMeanConvergeFrame_*0.7;
    if(n_frame_thresh > 8)  n_frame_thresh = 8;
    if(n_frame_thresh < 3)  n_frame_thresh = 3;
    if(targets.size() < n_frame_thresh)
    {
        return false;
    }
    double distMean = 0;   
    vector< pair<FramePtr, Vector2d> > targetResult;   
    targetResult.reserve(targets.size()); 
    vector<Vector2d> targetNormal; targetNormal.reserve(targets.size());
	double dist1 = (x3D - host->GetCameraCenter()).norm();	
    for(size_t i = 0; i < targets.size(); ++i)
    {
        Vector2d beforePx(targets[i].second);
        Matcher matcher;
		bool is_matched = matcher.findMatchSeed(seed, *(targets[i].first.get()), targets[i].second, 0.65);
		double dist2 = (x3D - targets[i].first->GetCameraCenter()).norm();
		if(dist1==0 || dist2==0)
            is_matched = false;
		const float ratioDist = dist2/dist1;	
		const float ratioOctave = (1<<seed.ftr->level) / (1<<matcher.search_level_);	
		if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
			is_matched = false;
        if(is_matched) 
        {
            Vector2d afterPx(targets[i].second);
            if(seed.ftr->type != Feature::EDGELET)
            {
                double err = (beforePx-afterPx).norm();
                err /= (1<<matcher.search_level_);
                distMean += err;
            }
            else
            {
                Vector2d normal(matcher.A_cur_ref_*seed.ftr->grad);
                normal.normalize();
                targetNormal.push_back(normal);
                double err = fabs(normal.transpose()*(beforePx-afterPx));
                err /= (1<<matcher.search_level_);
                distMean += err;
            }
            Vector3d f(targets[i].first->cam_->cam2world(targets[i].second));
            Vector2d obs(hso::project2d(f));
            targetResult.push_back(make_pair(targets[i].first, obs));
        }
    }
    if(targetResult.size() < n_frame_thresh)    
    {
        return false;
    }    
    distMean /= targetResult.size();
    if(seed.ftr->type != Feature::EDGELET && distMean > 3.2)    
    {
        isValid = false;
        return false;
    }
    if(seed.ftr->type == Feature::EDGELET && distMean > 2.5)
    {
        isValid = false;
        return false;
    }
    isValid = true;
    isEnough = false;
    if(seed.ftr->type != Feature::EDGELET && distMean > 2.5)    
    {
        return false;
    }
    if(seed.ftr->type == Feature::EDGELET && distMean > 2.0)
    {
        return false;
    }   
    #ifdef ACTIVATE_DBUG
        cout << "======================" << endl;
    #endif
    seedOptimizer(seed, targetResult, targetNormal);
    isEnough = true;
    return true;
}
void DepthFilter::seedOptimizer(Seed& seed, const vector<pair<FramePtr, Vector2d> >& targets, const vector<Vector2d>& normals)
{
    if(seed.ftr->type == Feature::EDGELET)
        assert(targets.size() == normals.size());
    double oldEnergy = 0.0, rho = 0, mu = 0.1, nu = 2.0;
    bool stop = false;
    int n_trials = 0;
    const int n_trials_max = 5;
	Frame* host = seed.ftr->frame;	
    double old_id = seed.mu;		
    vector<SE3> Tths; Tths.resize(targets.size());
    Vector3d pHost(seed.ftr->f * (1.0/old_id)); 
    vector<float> errors; errors.reserve(targets.size());
    for(size_t i = 0; i < targets.size(); ++i)
    {
        FramePtr target = targets[i].first;
		SE3 Ttw = target->GetPoseSE3();
		SE3 Twh = host->GetPoseInverseSE3();
        SE3 Tth = Ttw * Twh;
        Vector2d residual = targets[i].second-hso::project2d(Tth*pHost);
        if(seed.ftr->type == Feature::EDGELET)
            errors.push_back(fabs(normals[i].transpose()*residual));
        else
            errors.push_back(residual.norm());
    }
    hso::robust_cost::MADScaleEstimator mad_estimator;
    const double huberTH = mad_estimator.compute(errors); 
    for(size_t i = 0; i < targets.size(); ++i)
    {
        FramePtr target = targets[i].first;
        SE3 Ttw = target->GetPoseSE3();
		SE3 Twh = host->GetPoseInverseSE3();
        SE3 Tth = Ttw * Twh;
        Vector2d residual = targets[i].second-hso::project2d(Tth*pHost);
        if(seed.ftr->type == Feature::EDGELET)
        {
            double resEdgelet = normals[i].transpose()*residual;
            double hw = fabsf(resEdgelet) < huberTH ? 1 : huberTH / fabsf(resEdgelet);  
            oldEnergy += resEdgelet*resEdgelet * hw;
        }
        else
        {
            double res_dist = residual.norm();
            double hw = res_dist < huberTH ? 1 : huberTH / res_dist;
            oldEnergy += res_dist*res_dist * hw;
        }
        Tths[i] = Tth;
    }
    double H = 0, b = 0;
    for(int iter = 0; iter < 5; ++iter)
    {
        n_trials = 0;
        do
        {
            double new_id = old_id;
            double newEnergy = 0;
            H = b = 0;
            pHost = seed.ftr->f * (1.0/old_id);
            for(size_t i = 0; i < targets.size(); ++i)
            {
                FramePtr target = targets[i].first;
                SE3 Tth = Tths[i];
                Vector3d pTarget = Tth*pHost;
                Vector2d residual = targets[i].second-hso::project2d(pTarget);
                if(seed.ftr->type == Feature::EDGELET)
                {
                    double resEdgelet = normals[i].transpose()*residual;
                    double hw = fabsf(resEdgelet) < huberTH ? 1 : huberTH / fabsf(resEdgelet);
                    Vector2d Jxidd;
                    Point::jacobian_id2uv(pTarget, Tth, old_id, seed.ftr->f, Jxidd);
                    double JEdge = normals[i].transpose()*Jxidd;
                    H += JEdge*JEdge*hw;
                    b -= JEdge*resEdgelet*hw;
                }
                else
                {
                    double res_dist = residual.norm();
                    double hw = res_dist < huberTH ? 1 : huberTH / res_dist;
                    Vector2d Jxidd;
                    Point::jacobian_id2uv(pTarget, Tth, old_id, seed.ftr->f, Jxidd);
                    H += (Jxidd[0]*Jxidd[0] + Jxidd[1]*Jxidd[1])*hw;
                    b -= (Jxidd[0]*residual[0] + Jxidd[1]*residual[1])*hw;
                }
            }
            H *= 1.0+mu;
            double step = b/H;
            if(!(bool)std::isnan(step))
            {
                new_id = old_id+step;
                pHost = seed.ftr->f * (1.0/new_id);
                for(size_t i = 0; i < targets.size(); ++i)
                {
                    FramePtr target = targets[i].first;
                    SE3 Tth = Tths[i];
                    Vector2d residual = targets[i].second-hso::project2d(Tth*pHost);
                    if(seed.ftr->type == Feature::EDGELET)
                    {
                        double resEdgelet = normals[i].transpose()*residual;
                        double hw = fabsf(resEdgelet) < huberTH ? 1 : huberTH / fabsf(resEdgelet);
                        newEnergy += resEdgelet*resEdgelet * hw;
                    }
                    else
                    {
                        double res_dist = residual.norm();
                        double hw = res_dist < huberTH ? 1 : huberTH / res_dist;
                        newEnergy += res_dist*res_dist * hw;
                    }
                }
                rho = oldEnergy - newEnergy;
            }
            else
            {
                #ifdef ACTIVATE_DBUG
                    cout << "Matrix is close to singular!" << endl;
                    cout << "H = " << H << endl;
                    cout << "b = " << b << endl;
                #endif
                rho = -1;
            }
            if(rho > 0)
            {
                #ifdef ACTIVATE_DBUG
                    if(seed.ftr->type == Feature::EDGELET)
                        cout<< "EDGELET:  ";
                    else
                        cout<< "CORNER:  ";
                    cout<< "It. " << iter
                        << "\t Trial " << n_trials
                        << "\t Succ"
                        << "\t old Energy = " << oldEnergy
                        << "\t new Energy = " << newEnergy
                        << "\t lambda = " << mu
                        << endl;
                #endif
                oldEnergy = newEnergy;
                old_id = new_id;
                seed.opt_id = new_id;
                stop = fabsf(step) < 0.00001*new_id;
                mu *= std::max(1./3., std::min(1.-std::pow(2*rho-1,3), 2./3.));
                nu = 2.;
            }
            else
            {
                mu *= nu;
                nu *= 2.;
                ++n_trials;
                if (n_trials >= n_trials_max) stop = true;
                #ifdef ACTIVATE_DBUG
                    if(seed.ftr->type == Feature::EDGELET)
                        cout<< "EDGELET:  ";
                    else
                        cout<< "CORNER:  ";
                    cout<< "It. " << iter
                        << "\t Trial " << n_trials
                        << "\t Fail"
                        << "\t old Energy = " << oldEnergy
                        << "\t new Energy = " << newEnergy
                        << "\t lambda = " << mu
                        << endl;
                #endif
            }
        }while(!(rho>0 || stop));
        if(stop) break;
    }
}
void DepthFilter::ApplyScaledDepthFilter(const double s)
{
    seeds_updating_halt_ = true;
    {
        lock_t lock(seeds_mut_);
        std::list<Seed>::iterator it=seeds_.begin();
        int nreAdjust_seeds = 0;
        while(it!=seeds_.end())
        {
            it->mu = it->mu/s;          
            it->z_range = it->z_range/s;
            it->sigma2 = it->sigma2/(s*s);
            it++;
            nreAdjust_seeds++;
        } 
    }
    seeds_updating_halt_ = false;
}
void DepthFilter::SetTracker(FrameHandlerMono *pTracker)
{
    mpTracker = pTracker;
}
void DepthFilter::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper = pLocalMapper;
}
} // namespace vihso
