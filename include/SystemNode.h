/*
 * @Project: HSO_IMU
 * @Remark: 
 * @Author: YWZ
 * @Date: 2022-05-07 14:41:08
 * @LastEditors: YWZ
 * @LastEditTime: 2023-09-01 15:31:54
 * @FilePath: /hso_imu/include/SystemNode.h
 */
#ifndef SYSTEMNODE_H
#define SYSTEMNODE_H

#include<algorithm>
#include <sstream>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <sys/time.h>
#include <boost/thread.hpp>
#include <sophus/se3.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>


#include <vihso/config.h>
#include <vihso/frame_handler_mono.h>
#include <vihso/frame_handler_base.h>
#include <vihso/map.h>
#include <vihso/frame.h>
#include <vihso/feature.h>
#include <vihso/point.h>
#include <vihso/viewer.h>
#include <vihso/depth_filter.h>
#include "vihso/PhotomatricCalibration.h"
#include "vihso/camera.h"
#include "vihso/ImageReader.h"
#include "vihso/depth_filter.h"

#include "SettingParameters.h"
#include "ImuTypes.h"
#include "LocalMapping.h"

namespace vihso
{

class SystemNode 
{

public:
   SystemNode(const std::string &strSettingsFile, bool g_use_pc=false, bool g_show_pc=false, int g_start=0, int g_end=10000);
   ~SystemNode();

   void TrackMonoInertial(cv::Mat &img, double dframestamp, vector<IMU::IMUPoint>& vIMUData);
   void saveResult();
   void SaveKeyFrameTrajectoryEuRoC(const string &fFile, const string &kfFile, const string &sPointCloudFile=NULL);

public:

  Map* mpMap;

  hso::AbstractCamera* cam_;  

  vihso::FrameHandlerMono* vo_;

  vihso::DepthFilter* depth_filter_;

  vihso::LocalMapping* LocalMapper_;

  hso::Viewer* viewer_;
  boost::thread * viewer_thread_;

 public:
   bool mb_usePCC, mb_showPCC;  
   int mi_IMGstart, mi_IMGend;  
   const int G_MAX_RESOLUTION = 848*800; 

};

} 

#endif 
