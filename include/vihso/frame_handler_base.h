#ifndef VIHSO_FRAME_HANDLER_BASE_H_
#define VIHSO_FRAME_HANDLER_BASE_H_

#include <queue>
#include <boost/noncopyable.hpp>
#include <boost/function.hpp>
#include <boost/thread.hpp>
#include <vihso/global.h>
#include <vihso/map.h>

#include "vihso/vikit/timer.h"
#include "vihso/vikit/ringbuffer.h"

#include <time.h>
#include<sys/time.h>

#include "ImuTypes.h"

namespace hso {
class PerformanceMonitor;}

namespace vihso
{
class Point;
class Matcher;
class DepthFilter;
class IMUPoint;
class Map;

struct TestRecord
{
	int Frame_ID;

	double fts_Variance;
	double fts_StandardDeviation;
	double fts_Distribute;

	TestRecord(int id=0, double var=0, double standar=0, double dist=0)		
	{
		this->Frame_ID = id;
		this->fts_Variance = var;
		this->fts_StandardDeviation = standar;
		this->fts_Distribute = dist;
	}
};

class FrameHandlerBase : boost::noncopyable
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 public:
	enum Stage              
	{
		STAGE_PAUSED,           
		STAGE_FIRST_FRAME,      
		STAGE_SECOND_FRAME,     
		STAGE_DEFAULT_FRAME,    
		STAGE_RELOCALIZING      
	};

	enum TrackingQuality    
	{
		TRACKING_INSUFFICIENT,  
		TRACKING_BAD,         
		TRACKING_GOOD          
	};

	enum UpdateResult     
	{
		RESULT_NO_KEYFRAME,  
		RESULT_IS_KEYFRAME,  
		RESULT_FAILURE        
	};


	enum eTrackingState
	{
		SYSTEM_NOT_READY=-1,            
		NO_IMAGES_YET=0,               
		NOT_INITIALIZED=1,              
		ONLY_VISIUAL_INIT=2,            
		OK=3,                           
		RECENTLY_LOST=4,                
		LOST=5,                         
		OK_KLT=6
	};
	enum TrackingResult       
	{
		RES_KF,
		RES_Frame,
		RES_FAILURE
	};

	FrameHandlerBase(Map* pMap);
	virtual ~FrameHandlerBase();

	const Map& GetConstMap() const { return *mpMap; } 
	Map* GetpMap()  { return mpMap; }      
	void reset() { set_reset_ = true; }    
	void start() { set_start_ = true; }    
	Stage stage() const { return stage_; } 
	TrackingQuality trackingQuality() const { return tracking_quality_; } 
	double lastProcessingTime() const { return timer_.getTime(); }        
	size_t lastNumObservations() const { return num_obs_last_; }          


 public: 
	Stage stage_;               
	bool set_reset_;           
	bool set_start_;           
	Map* mpMap;
	hso::Timer timer_;         
	hso::RingBuffer<double> acc_frame_timings_;  
	hso::RingBuffer<size_t> acc_num_obs_;    
	size_t num_obs_last_;               
	TrackingQuality tracking_quality_;            
	size_t regular_counter_;                      

	bool startFrameProcessingCommon(const double timestamp); 
	int finishFrameProcessingCommon(
		const size_t update_id,
		const UpdateResult dropout,
		const size_t num_observations);                               

	void resetCommon();                        
	virtual void resetAll() {cout << "BaseResetALL" << endl; resetCommon(); } 

	virtual void setTrackingQuality(const size_t num_observations);
	virtual void optimizeStructure(FramePtr frame, size_t max_n_pts, int max_iter); 

 public: 
	bool b_fpsAndobs = false;
	struct timeval fps_start; 
	struct timeval fps_end;   
	vector<float> v_Fps_;  
	vector<float> v_obs_;
	vector<size_t> v_FrameID;

	bool b_KFPose = true;
	double timeall=0; 

	vector<TestRecord> vTestRecord;

 public: 
	eTrackingState mState; 
	double lasttimestamp;

	bool mbStep;
	bool bStepByStep;
	void SetStepByStep(bool bSet);

	bool startStateCheck(const double timestamp);
	bool finishStateCheck();
	bool finishStateCheck(FramePtr cur,  double width, double height);

};

void getProportion(int* array, int len, double* res_array);
double getVariance(const double* array, int len);

} // namespace nslam

#endif  
