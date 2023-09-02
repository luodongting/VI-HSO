#ifndef VIHSO_CONFIG_H_
#define VIHSO_CONFIG_H_

#include <string>
#include <stdint.h>
#include <stdio.h>

namespace vihso {

using std::string;

class Config
{
public:
	static Config& getInstance();
	static string& traceName() { return getInstance().trace_name; }
	static string& traceDir() { return getInstance().trace_dir; }
	static int& nPyrLevels() { return getInstance().n_pyr_levels; }
	static bool& useImu() { return getInstance().use_imu; }
	static int& coreNKfs() { return getInstance().core_n_kfs; }
	static double& mapScale() { return getInstance().map_scale; }
	static int& gridSize() { return getInstance().grid_size; }
	static double& initMinDisparity() { return getInstance().init_min_disparity; }
	static int& initMinTracked() { return getInstance().init_min_tracked; }
	static int& initMinInliers() { return getInstance().init_min_inliers; }
	static int& kltMaxLevel() { return getInstance().klt_max_level; }
	static int& kltMinLevel() { return getInstance().klt_min_level; }
	static double& poseOptimThresh() { return getInstance().poseoptim_thresh; }
	static int& poseOptimNumIter() { return getInstance().poseoptim_num_iter; }
	static double& lobaCorThresh() { return getInstance().loba_CornorThresh; }
	static double& lobaEdgeThresh() { return getInstance().loba_EdgeLetThresh; }
	static double& lobaRobustHuberWidth() { return getInstance().loba_robust_huber_width; }
	static int& lobaNumIter() { return getInstance().loba_num_iter; }
	static double& SelsectKFsMinDist() { return getInstance().kfselect_mindist; }
	static int& maxNKfs() { return getInstance().max_n_kfs; }
	static double& imgImuDelay() { return getInstance().img_imu_delay; }
	static int& maxFts() { return getInstance().Track_Max_fts; }
	static void setmaxFts(const int& max_nfts) { getInstance().Track_Max_fts = max_nfts; }
	static int& qualityMinFts() { return getInstance().Track_Min_fts; }
	static int& qualityMaxFtsDrop() { return getInstance().Track_MaxDrop_fts; }
	static double& edgeLetCosAngle() { return getInstance().edgelet_angle; }
	static int& maxDropKeyframe() { return getInstance().n_max_drop_keyframe; }
	static void setMaxKLTLevel(const int& max_level) {  getInstance().klt_max_level = max_level; }

private:
	Config();
	Config(Config const&);
	void operator=(Config const&);
	string trace_name;
	string trace_dir;
	int n_pyr_levels;
	bool use_imu;
	int core_n_kfs;
	double map_scale;
	int grid_size;
	double init_min_disparity;
	int init_min_tracked;
	int init_min_inliers;
	int klt_max_level;
	int klt_min_level;
	double poseoptim_thresh;
	int poseoptim_num_iter;
	double loba_CornorThresh;
	double loba_EdgeLetThresh;
	double loba_robust_huber_width;
	int loba_num_iter;
	double kfselect_mindist;
	int max_n_kfs;
	double img_imu_delay;
	int Track_Max_fts;
	int Track_Min_fts;
	int Track_MaxDrop_fts;
	double edgelet_angle;

	int n_max_drop_keyframe;
};

} // namespace vihso

#endif // VIHSO_CONFIG_H_
