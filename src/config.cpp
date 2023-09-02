#include <vihso/config.h>

#include <fstream>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <iomanip>

namespace vihso {

Config::Config() 
{
	cv::FileStorage fsConfig("../include/Config.yaml", cv::FileStorage::READ);  //Config.yaml
    if(!fsConfig.isOpened())
    {
        std::cerr << "Failed to open settings file at: " << "/include/Config.yaml" << std::endl;
        exit(-1);
    }

    trace_name = (string)fsConfig["Track_Name"];
	trace_dir = (string)fsConfig["Trace_Dir"];
	
	int USE_IMU = fsConfig["IMU.used"];
	if(USE_IMU==1)
		use_imu = true;
	else
		use_imu = false;

	n_pyr_levels = fsConfig["Initialization.nPyramidLevels"];
    map_scale = fsConfig["Initialization.MapScale"];
    init_min_disparity = fsConfig["Initialization.MinThDisparity"];
    init_min_tracked = fsConfig["Initialization.MinThTrackFeatures"];
    init_min_inliers = fsConfig["Initialization.MinRANSACInliers"];

	grid_size = fsConfig["FeatureExtractor.GridSize"];

    klt_max_level = fsConfig["LucasKanade.MaxLevel"];
    klt_min_level = fsConfig["LucasKanade.MinLevel"];

    poseoptim_thresh = fsConfig["VisiualOptimize.ThError"];
    poseoptim_num_iter = fsConfig["VisiualOptimize.nIterator"];

	core_n_kfs = fsConfig["LocalBA.nNeighborKFs"];
    loba_CornorThresh = fsConfig["LocalBA.ThCornorError"];
	loba_EdgeLetThresh = fsConfig["LocalBA.ThEdgeLetError"];
    loba_robust_huber_width = fsConfig["ThRobustHuberWidth"];
    loba_num_iter = fsConfig["LocalBA.nIterator"];
	
    kfselect_mindist = fsConfig["Map.MinDistTwoKF"];
    
    max_n_kfs = fsConfig["Map.MaxKFs"];
    img_imu_delay = fsConfig["IMU.Delay"];

    Track_Max_fts = fsConfig["Track.MaxFeatures"];
    Track_Min_fts = fsConfig["Track.MinFeatures"];
    Track_MaxDrop_fts = fsConfig["Track.MaxDropFeatures"];

    edgelet_angle = fsConfig["Feature.EdgeLetCosAngle"];
    n_max_drop_keyframe = fsConfig["MaxDropKFs"];
}

Config& Config::getInstance()
{
  static Config instance;
  return instance;
}

} // namespace vihso

