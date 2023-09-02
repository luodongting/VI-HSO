
#include <SystemNode.h>
#include "SettingParameters.h"
#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace cv;

namespace vihso
{

SystemNode::SystemNode( const std::string &strSettingsFile, bool g_use_pc, bool g_show_pc, int g_start, int g_end):
                        mb_usePCC(g_use_pc), mb_showPCC(g_show_pc), mi_IMGstart(g_start), mi_IMGend(g_end)
{
    readParameters(strSettingsFile);

	if(CAMTYPE == "PinHole")
	{
		int width_i = COL, height_i = ROW;
		cam_ = new hso::PinholeCamera(width_i, height_i, fx, fy, cx, cy, k1, k2, p1, p2);
		cout << "Camera: " << "Pinhole\t" << "Width=" << COL << "\tHeight=" << ROW << endl;
	}
	else if(CAMTYPE == "EquiDistant")
	{
		int width_i = COL, height_i = ROW;
		if(COL*ROW > G_MAX_RESOLUTION + 0.00000001)
		{
			double resize_rate = sqrt(COL*ROW/G_MAX_RESOLUTION);
			width_i  = int(COL/resize_rate);
			height_i = int(ROW/resize_rate);
			resize_rate = sqrt(COL*ROW/width_i*height_i);
			fx /= resize_rate;
			fy /= resize_rate;
			cx /= resize_rate;
			cy /= resize_rate;
		}

		cam_ = new hso::EquidistantCamera(width_i, height_i, fx, fy, cx, cy, k1, k2, p1, p2);
		cout << "Camera: " << "Equidistant\t" << "Width=" << COL << "\tHeight=" << ROW << endl;
	}
	else if(CAMTYPE == "FOV")
	{

		int width_i = COL, height_i = ROW;
		if(COL*ROW > G_MAX_RESOLUTION + 0.00000001)
		{
			double resize_rate = sqrt(COL*ROW/G_MAX_RESOLUTION);
			width_i  = int(COL/resize_rate);
			height_i = int(ROW/resize_rate);
			resize_rate = sqrt(COL*ROW/width_i*height_i);

			if(cx > 1 && cy > 1)
			{
				fx /= resize_rate;
				fy /= resize_rate;
				cx /= resize_rate;
				cy /= resize_rate;
			}
		}

        if(1)
            cam_ = new hso::FOVCamera(width_i, height_i, fx, fy, cx, cy, k1, true);
        else
            cam_ = new hso::FOVCamera(width_i, height_i, fx, fy, cx, cy, k1, false);
        cout << "Camera: " << "FOV\t" << "Width=" << COL << "\tHeight=" << ROW << endl;
    }
    else
        cout << "Calibtation file error." << endl;
	

    mpMap = new Map();
    vo_ = new vihso::FrameHandlerMono(mpMap, cam_, mb_usePCC);
    vo_->start();
    feature_detection::FeatureExtractor* featureExt = new feature_detection::FeatureExtractor(cam_->width(), cam_->height(), Config::gridSize(), Config::nPyrLevels()); 
    DepthFilter::callback_t depth_filter_cb = boost::bind(&CandidatesManager::newCandidatePoint, &mpMap->mCandidatesManager, _1, _2);
    depth_filter_ = new vihso::DepthFilter(featureExt, depth_filter_cb);
    depth_filter_->startThread();
    viewer_ = new hso::Viewer(vo_, mpMap);
    viewer_thread_ = new boost::thread(&hso::Viewer::run, viewer_);
    viewer_thread_->detach();
    vo_->SetDepthFilter(depth_filter_);
    depth_filter_->SetTracker(vo_);
}

SystemNode::~SystemNode()
{
}


void SystemNode::TrackMonoInertial(cv::Mat &img, double dframestamp, vector<IMU::IMUPoint>& vIMUData)
{
        if(cam_->getUndistort())
        {
            cam_->undistortImage(img, img);
        }

        vo_->addIMGIMU(img, vIMUData, dframestamp);

        cv::waitKey (1);
    
}

void SystemNode::saveResult()
{
    std::ofstream okt("KeyFrameTrajectory.csv");
    for(auto it = vo_->mpMap->keyframes_.begin(); it != vo_->mpMap->keyframes_.end(); ++it) 
    {
        Sophus::SE3 Tinv = (*it)->GetPoseInverseSE3();
        std::string stimestamp = std::to_string((*it)->mTimeStamp);

        okt << setprecision(9)
            << (*it)->mTimeStamp << " " 
            << Tinv.translation()[0] << " " 
            << Tinv.translation()[1] << " " 
            << Tinv.translation()[2] << " "
            << Tinv.unit_quaternion().x() << " " 
            << Tinv.unit_quaternion().y() << " "
            << Tinv.unit_quaternion().z() << " "
            << Tinv.unit_quaternion().w() << endl;
    }
    okt.close();
    cout << "KF Result has been saved." << endl;

    std::ofstream kft("EurocTrajectory.csv");
    for(auto it = vo_->mpMap->keyframes_.begin(); it != vo_->mpMap->keyframes_.end(); ++it)
    {
        Sophus::SE3 Tinv = (*it)->GetPoseInverseSE3(); 
        std::string stimestamp = std::to_string((*it)->mTimeStamp*1e9);

        kft << stimestamp << ","    
            << Tinv.translation()[0] << "," 
            << Tinv.translation()[1] << "," 
            << Tinv.translation()[2] << ","
            << Tinv.unit_quaternion().x() << "," 
            << Tinv.unit_quaternion().y() << ","
            << Tinv.unit_quaternion().z() << ","
            << Tinv.unit_quaternion().w() << endl;
    }
    kft.close();
    cout << "Euroc Result has been saved." << endl;
}

void SystemNode::SaveKeyFrameTrajectoryEuRoC(const string &fFile, const string &kfFile, const string &sPointCloudFile)
{
    std::ofstream okfDate(kfFile);
    okfDate << fixed;
    Eigen::Matrix4d Twb0 = Eigen::Matrix4d::Identity();
    for(auto it = vo_->mpMap->keyframes_.begin(); it != vo_->mpMap->keyframes_.end(); ++it)
    {
        std::string stimestamp = std::to_string((*it)->mTimeStamp);
        Eigen::Matrix4d Twbi = (*it)->GetImuPose();
        Eigen::Matrix4d Tb0bi = Twb0.inverse() * Twbi;
        Eigen::Vector3d twb = Tb0bi.block<3,1>(0,3);
        Eigen::Matrix3d Rwb = Tb0bi.block<3,3>(0,0);
        Eigen::Quaterniond qwb(Rwb);

        okfDate  << setprecision(6) << stimestamp << " "     
                << setprecision(9)
                << twb[0] << " " 
                << twb[1] << " " 
                << twb[2] << " "
                << qwb.x() << " " 
                << qwb.y() << " "
                << qwb.z() << " "
                << qwb.w() << endl;
    }
    okfDate.close();
    cout << vo_->mFILE_NAME << " KFs Trajectory has been saved." << endl;

    std::ofstream ofDate(fFile);
    ofDate << fixed;
    for(auto it = vo_->mpMap->keyframes_.begin(); it != vo_->mpMap->keyframes_.end(); ++it) 
    {
        std::string stimestamp = std::to_string((*it)->mTimeStamp);

        Eigen::Matrix4d Twbi = (*it)->GetImuPose();
        Eigen::Matrix4d Tb0bi = Twb0.inverse() * Twbi;
        Eigen::Vector3d twb = Tb0bi.block<3,1>(0,3);
        Eigen::Matrix3d Rwb = Tb0bi.block<3,3>(0,0);
        Eigen::Quaterniond qwb(Rwb);

        ofDate  << setprecision(6) << stimestamp << " "     
                << setprecision(9)
                << twb[0] << " " 
                << twb[1] << " " 
                << twb[2] << " "
                << qwb.x() << " " 
                << qwb.y() << " "
                << qwb.z() << " "
                << qwb.w() << endl;

        if(!(*it)->mlRelativeFrame.empty())
        {
            for(std::list<std::pair<double, Eigen::Matrix4d>>::iterator lit=(*it)->mlRelativeFrame.begin();
                lit!=(*it)->mlRelativeFrame.end(); lit++)
            {
                std::string stimestamp_f = std::to_string((*lit).first);
                Eigen::Matrix4d Tbibj = (*lit).second;
                Eigen::Matrix4d Tb0bj = Tb0bi*Tbibj;

                Eigen::Vector3d twb_f = Tb0bj.block<3,1>(0,3);
                Eigen::Matrix3d Rwb_f = Tb0bj.block<3,3>(0,0);
                Eigen::Quaterniond qwb_f(Rwb_f);

                ofDate  << setprecision(6) << stimestamp_f << " "     
                        << setprecision(9)
                        << twb_f[0] << " " 
                        << twb_f[1] << " " 
                        << twb_f[2] << " "
                        << qwb_f.x() << " " 
                        << qwb_f.y() << " "
                        << qwb_f.z() << " "
                        << qwb_f.w() << endl;
            }
        }
    }
    ofDate.close();
    cout << vo_->mFILE_NAME << " Frames Trajectory has been saved." << endl;
}

}   //namespace vihso