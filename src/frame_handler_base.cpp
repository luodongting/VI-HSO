#include <stdlib.h>
#include <Eigen/StdVector>
#include <boost/bind.hpp>
#include <fstream>
#include <vihso/frame_handler_base.h>
#include <vihso/config.h>
#include <vihso/feature.h>
#include <vihso/matcher.h>
#include <vihso/map.h>
#include <vihso/point.h>

namespace vihso
{

#ifdef VIHSO_TRACE
hso::PerformanceMonitor* g_permon = NULL;
#endif


FrameHandlerBase::FrameHandlerBase(Map* pMap) :
	stage_(STAGE_PAUSED),
	set_reset_(false),
	set_start_(false),
	mpMap(pMap),
	acc_frame_timings_(10),
	acc_num_obs_(10),
	num_obs_last_(0),
	tracking_quality_(TRACKING_INSUFFICIENT),
	regular_counter_(0),
	mState(SYSTEM_NOT_READY),
	lasttimestamp(0.0),
	mbStep(false),
	bStepByStep(false)
{
	#ifdef VIHSO_TRACE
		g_permon = new hso::PerformanceMonitor();
		g_permon->addTimer("pyramid_creation");
		g_permon->addTimer("sparse_img_align");
		g_permon->addTimer("reproject");
		g_permon->addTimer("reproject_kfs");
		g_permon->addTimer("reproject_candidates");
		g_permon->addTimer("feature_align");
		g_permon->addTimer("pose_optimizer");
		g_permon->addTimer("point_optimizer");
		g_permon->addTimer("local_ba");
		g_permon->addTimer("tot_time");
		g_permon->addLog("timestamp");
		g_permon->addLog("img_align_n_tracked");
		g_permon->addLog("repr_n_mps");
		g_permon->addLog("repr_n_new_references");
		g_permon->addLog("sfba_thresh");
		g_permon->addLog("sfba_error_init");
		g_permon->addLog("sfba_error_final");
		g_permon->addLog("sfba_n_edges_final");
		g_permon->addLog("loba_n_erredges_init");
		g_permon->addLog("loba_n_erredges_fin");
		g_permon->addLog("loba_err_init");
		g_permon->addLog("loba_err_fin");
		g_permon->addLog("n_candidates");
		g_permon->addLog("dropout");
		g_permon->init(Config::traceName(), Config::traceDir());
	#endif
}

FrameHandlerBase::~FrameHandlerBase()
{
	#ifdef VIHSO_TRACE
		delete g_permon;
	#endif
}

bool FrameHandlerBase::startFrameProcessingCommon(const double timestamp)
{
	if(set_start_)  
	{
		resetAll();                
		stage_ = STAGE_FIRST_FRAME; 
	}

	if(stage_ == STAGE_PAUSED)  
		return false;

	VIHSO_LOG(timestamp);
	VIHSO_START_TIMER("tot_time");
	timer_.start();
	gettimeofday(&fps_start,NULL);
	mpMap->emptyTrash();
	return true;
}

int FrameHandlerBase::finishFrameProcessingCommon(const size_t update_id, const UpdateResult dropout, const size_t num_observations)
{
	#ifdef OUT_FPS_OBS
	b_fpsAndobs = true;
	gettimeofday(&fps_end,NULL);
	double time_use = (fps_end.tv_sec-fps_start.tv_sec)*1000000+(fps_end.tv_usec-fps_start.tv_usec); 
	double fps_cur = 1000000.0 / time_use;
	v_Fps_.push_back(fps_cur);
	v_obs_.push_back(num_observations);
	v_FrameID.push_back(update_id);

	#endif
	VIHSO_LOG(dropout);
	acc_frame_timings_.push_back(timer_.stop());  
	if(stage_ == STAGE_DEFAULT_FRAME)
		acc_num_obs_.push_back(num_observations);
	num_obs_last_ = num_observations;            
	VIHSO_STOP_TIMER("tot_time");

	#ifdef VIHSO_TRACE
		g_permon->writeToFile();
		{
		boost::unique_lock<boost::mutex> lock(mpMap->mCandidatesManager.mut_);
		size_t n_candidates = mpMap->mCandidatesManager.mlCandidatePoints.size();
		VIHSO_LOG(n_candidates);
		}
	#endif

	if(dropout == RESULT_FAILURE &&
		(stage_ == STAGE_DEFAULT_FRAME || stage_ == STAGE_RELOCALIZING ))
	{
		stage_ = STAGE_RELOCALIZING;
		tracking_quality_ = TRACKING_INSUFFICIENT;
	}
	else if (dropout == RESULT_FAILURE)
		resetAll();
	if(set_reset_)
		resetAll();

	return 0;
}

void FrameHandlerBase::resetCommon()
{
	mpMap->reset();
	stage_ = STAGE_PAUSED;
	mState = SYSTEM_NOT_READY;
	set_reset_ = false;
	set_start_ = false;
	tracking_quality_ = TRACKING_INSUFFICIENT;
	num_obs_last_ = 0;
}

void FrameHandlerBase::setTrackingQuality(const size_t num_observations)
{

	tracking_quality_ = TRACKING_GOOD;

	if(num_observations < (size_t)Config::qualityMinFts())  
	{
		VIHSO_WARN_STREAM_THROTTLE(0.5, "After cam-imuBA, Tracking "<< num_observations << " less than "<< Config::qualityMinFts() <<" features!");
		tracking_quality_ = TRACKING_INSUFFICIENT;
	}

	const int feature_drop = static_cast<int>(std::min((int)num_obs_last_, Config::maxFts())) - num_observations;  
	if(feature_drop > Config::qualityMaxFtsDrop())
	{
		VIHSO_WARN_STREAM("Lost "<< feature_drop <<" features!");
		tracking_quality_ = TRACKING_BAD;
	}
}

bool ptLastOptimComparator(Point* lhs, Point* rhs)
{
  	return (lhs->last_structure_optim_ < rhs->last_structure_optim_);
}

void FrameHandlerBase::optimizeStructure(FramePtr frame, size_t max_n_pts, int max_iter)  //20 3
{
    deque<Point*> pts; 
    for(Features::iterator it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
    {
        if((*it)->point == NULL) continue;
        size_t nobs = (*it)->point->GetNumofObs();
        if(nobs > 1) 
          pts.push_back((*it)->point);
    }

    max_n_pts = min(max_n_pts, pts.size());
    nth_element(pts.begin(), pts.begin() + max_n_pts, pts.end(), ptLastOptimComparator);
    for(deque<Point*>::iterator it=pts.begin(); it!=pts.begin()+max_n_pts; ++it)
    {
        (*it)->optimize(max_iter); 
        (*it)->last_structure_optim_ = frame->id_;  
    }
}


bool FrameHandlerBase::startStateCheck(const double timestamp)
{
	if(set_start_) 
	{
		resetAll();
		mState = NO_IMAGES_YET;
		lasttimestamp = timestamp;
	}

	if(timestamp < lasttimestamp) 
		mState = SYSTEM_NOT_READY;
	
	if(mState == SYSTEM_NOT_READY)  
		return false;

	mpMap->emptyTrash(); 

	if(bStepByStep)
	{
		while(!mbStep)
		usleep(500);
		mbStep = false;
	}

	return true;
}

bool FrameHandlerBase::finishStateCheck()
{
  	return true;
}
bool FrameHandlerBase::finishStateCheck(FramePtr cur,  double width, double height)
{
	clock_t start2 = clock();
	int iDistribution[10] = {};
	Features fts = cur->GetFeatures();
	for(Features::iterator it=fts.begin(); it!=fts.end(); it++)
	{
		Vector2d px = (*it)->px;
		float x = px.x();	
		float y = px.y();	

		if(y < height/2.0)
			iDistribution[0]++;
		else
			iDistribution[1]++;

		if(x < width/2.0)
			iDistribution[2]++;
		else
			iDistribution[3]++;

		if(y < height -height/width * x)
			iDistribution[4]++;
		else
			iDistribution[5]++;
		
		if(y > height/width * x)
			iDistribution[6]++;
		else
			iDistribution[7]++;

		if(y<0.75*height && y>0.25*height && x<0.75*width && x>0.25*width)
			iDistribution[8]++;
		else
			iDistribution[9]++;

	}
	clock_t end2  = clock();
	double programTimes2 = ((double) end2 -start2) / CLOCKS_PER_SEC * 1000.0;
	
	double dDistribution[10]={};
	double dLogRatio = -10.0;
	getProportion(iDistribution, 10, dDistribution);

	double variance =  getVariance(dDistribution, 10);
	double standardDeviation = sqrt(variance);
	double distribute = dLogRatio*log(variance);

	TestRecord testrecord(cur->id_, variance, standardDeviation, distribute);
	vTestRecord.push_back(testrecord);

  	return true;
}

void FrameHandlerBase::SetStepByStep(bool bSet)
{
  	bStepByStep = bSet;
}

void getProportion(int* array, int len, double* res_array)
{
	double sum = 0;
	for(int i=0;i<len; i++)
	{
		sum = sum + array[i];
	}
	for(int i=0;i<len; i++)
	{
		res_array[i] = (double)array[i]/sum;
	}
}

double getVariance(const double* array, int len)
{
	double sum = 0;
	for(int i=0;i<len; i++)
	{
		sum = sum + array[i];
	}
    double mean =  sum / len;
    
	double variance  = 0.0;
    for (int i = 0 ; i < len ; i++)
    {
        variance = variance + pow(array[i]-mean,2);
    }
    variance = variance/len;
    return variance;
}

} // namespace vihso
