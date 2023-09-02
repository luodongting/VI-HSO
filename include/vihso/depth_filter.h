#ifndef VIHSO_DEPTH_FILTER_H_
#define VIHSO_DEPTH_FILTER_H_

#include <queue>
#include <boost/thread.hpp>
#include <boost/function.hpp>
#include <vihso/global.h>
#include <vihso/feature_detection.h>
#include <vihso/reprojector.h>
#include <vihso/matcher.h>
#include <vihso/IndexThreadReduce.h>

#include "vihso/vikit/performance_monitor.h"
#include <vihso/frame_handler_mono.h>
#include "LocalMapping.h"

namespace vihso {

class Frame;
class Feature;
class Point;
class LocalMapping;

struct Seed
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static int batch_counter;     
    static int seed_counter;      
    int batch_id;                 
    int id;                       
    Feature* ftr;                 
    float a;                      
    float b;                      
    float mu;                     
    float z_range;                
    float sigma2;                 
    Matrix2d patch_cov;           
    vector<FramePtr> pre_frames;  

    bool isValid;

    Vector2i eplStart;
    Vector2i eplEnd;

    bool haveReprojected; 
    Point* temp;         

    std::vector<float> vec_distance; 
    std::vector<float> vec_sigma2;    

    std::vector<FramePtr> optFrames_P;  
    std::vector<FramePtr> optFrames_A;  
    float opt_id;

    FramePtr last_update_frame; 
    Vector2d last_matched_px;   
    int last_matched_level;     

    float converge_thresh;    

    bool is_update;

    Seed(Feature* ftr, float depth_mean, float depth_min, float converge_threshold=200);
};


class DepthFilter{

friend class vihso::Reprojector;
friend class vihso::FrameHandlerMono;

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	typedef boost::unique_lock<boost::mutex> lock_t;
	typedef boost::function<void ( Point*, double )> callback_t;

	struct Options
	{
		bool check_ftr_angle;                       
		bool epi_search_1d;                        
		bool verbose;                              
		bool use_photometric_disparity_error;      
		int max_n_kfs;                              
		double sigma_i_sq;                          
		double seed_convergence_sigma2_thresh;    
		Options() : check_ftr_angle(false), epi_search_1d(false), verbose(false), use_photometric_disparity_error(false),
					max_n_kfs(3), sigma_i_sq(5e-4), seed_convergence_sigma2_thresh(200.0)
		{}
	} options_;

	boost::mutex stats_mut_;
	RunningStats* runningStats_; 
	int n_update_last_;         

	FramePtr active_frame_;     	
	
	boost::mutex mean_mutex_;
	size_t nMeanConvergeFrame_ = 6; 

	DepthFilter(feature_detection::FeatureExtractor* featureExtractor, callback_t seed_converged_cb);
	virtual ~DepthFilter();

	void startThread();
	void stopThread();
	void reset();

	void addFrame(FramePtr frame);
	void addKeyframe(FramePtr frame, double depth_mean, double depth_min, float converge_thresh=200.0);
	void removeKeyframe(FramePtr frame);
	static void updateSeed(const float x, const float tau2, Seed* seed);  
	static double computeTau( const SE3& T_ref_cur, const Vector3d& f, const double z, const double px_error_angle);  

	void getSeedsCopy(const FramePtr& frame, std::list<Seed>& seeds);
	std::list<Seed>& getSeeds() { return seeds_; }                    

	void directPromotionFeature();

protected:
	boost::mutex detector_mut_;

	feature_detection::FeatureExtractor* featureExtractor_;
	callback_t seed_converged_cb_; 
	std::list<Seed> seeds_;         
	boost::mutex seeds_mut_;       

	boost::mutex m_converge_seed_mut;
	std::list<Seed> m_converge_seed;

	bool seeds_updating_halt_;                    
	boost::thread* thread_;                       
	std::queue<FramePtr> frame_queue_;           
	boost::mutex frame_queue_mut_;               
	boost::condition_variable frame_queue_cond_;  

	FramePtr new_keyframe_;              
	bool new_keyframe_set_;               
	double new_keyframe_min_depth_;       
	double new_keyframe_mean_depth_;     
	hso::PerformanceMonitor permon_;      
	Matcher matcher_;                     
	float convergence_sigma2_thresh_ = 200.0;
	lsd_slam::IndexThreadReduce* threadReducer_;
	double px_error_angle_; 
	

	void initializeSeeds(FramePtr frame);
	virtual void updateSeeds(FramePtr frame, bool bProcessFramePrior = true);
	// virtual void UpdateSeedsWithKFs(FramePtr frame);
	void clearFrameQueue();
	void updateSeedsLoop();
	void observeDepth();
	void observeDepthRow(int yMin, int yMax, RunningStats* stats);
	bool observeDepthWithPreviousFrameOnce(std::list<Seed>::iterator& ite);

private:
	std::vector< list<FramePtr> > frame_prior_; 
	size_t n_pre_update_, n_pre_try_;
	size_t nPonits, nSkipFrame;
	vector<size_t> m_v_n_converge; 
	bool activatePoint(Seed& seed, bool& isValid, bool& isEnough);
	void seedOptimizer(Seed& seed, const vector<pair<FramePtr, Vector2d> >& targets, const vector<Vector2d>& normals);

public:
	void ApplyScaledDepthFilter(const double s);
	FrameHandlerMono* mpTracker;
	LocalMapping* mpLocalMapper;
	void SetTracker(FrameHandlerMono* pTracker);
	void SetLocalMapper(LocalMapping *pLocalMapper);
};

} // namespace vihso

#endif // VIHSO_DEPTH_FILTER_H_
