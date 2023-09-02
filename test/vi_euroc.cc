#include<algorithm>
#include <sstream>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <sys/time.h>
#include <boost/thread.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <sophus/se3.h>
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
#include "ImuTypes.h"
#include "SystemNode.h"
#include "SettingParameters.h"
using namespace std;
using namespace cv;
void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps);
void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<Eigen::Vector3f> &vAcc, vector<Eigen::Vector3f> &vGyro);
double ttrack_total = 0;  
int main(int argc, char *argv[])
{
    if(argc < 4)
    {
        cerr << endl << "Usage: ./vi_euroc  settings_Path sequence1_Path time1_Path (sequenceN_Path timeN_Path) file_Name" << endl;
        return 1;
    }
    const int num_seq = (argc-2)/2;
    cout << "Num of Sequences = " << num_seq << endl;
    bool bFileName= (((argc-2) % 2) == 1);  
    string file_name;
    if (bFileName)
    {
        file_name = string(argv[argc-1]);
        cout << "File Name: " << file_name << endl;
    }
    int seq;
    vector< vector<string> > vstrImageFilenames;    
    vector< vector<double> > vTimestampsCam;        
    vector< vector<Eigen::Vector3f> > vAcc, vGyro;  
    vector< vector<double> > vTimestampsImu;        
    vector<int> nImages;    
    vector<int> nImu;       
    vector<int> first_imu(num_seq,0);               
    int tot_images = 0;                             
    vstrImageFilenames.resize(num_seq);
    vTimestampsCam.resize(num_seq);
    vAcc.resize(num_seq);
    vGyro.resize(num_seq);
    vTimestampsImu.resize(num_seq);
    nImages.resize(num_seq);
    nImu.resize(num_seq);
    for (seq = 0; seq<num_seq; seq++)
    {
        cout << "Loading images for sequence " << seq << "...";
        string pathSeq(argv[(2*seq) + 2]);                  
        string pathCam0 = pathSeq + "/mav0/cam0/data";      
        string pathImu = pathSeq + "/mav0/imu0/data.csv";   
        string pathTimeStamps(argv[(2*seq) + 3]);           
        LoadImages(pathCam0, pathTimeStamps, vstrImageFilenames[seq], vTimestampsCam[seq]);
        cout << "LOADED!" << endl;
        cout << "Loading IMU for sequence " << seq << "..." ;
        LoadIMU(pathImu, vTimestampsImu[seq], vAcc[seq], vGyro[seq]);
        cout << "LOADED!" << endl;
        nImages[seq] = vstrImageFilenames[seq].size();
        tot_images += nImages[seq];
        nImu[seq] = vTimestampsImu[seq].size();
        if((nImages[seq]<=0)||(nImu[seq]<=0))   
        {
            cerr << "ERROR: Failed to load images or IMU for sequence" << seq << endl;
            return 1;
        }
        if(vTimestampsImu[seq][first_imu[seq]]<=vTimestampsCam[seq][0])
        {
            while(vTimestampsImu[seq][first_imu[seq]]<=vTimestampsCam[seq][0])
                first_imu[seq]++;
            first_imu[seq]--; 
        }
    }
    vector<float> vTimesTrack;  
    vTimesTrack.resize(tot_images);
    cout.precision(17); 
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages[0] << endl;
    cout << "IMU data in the sequence: " << nImu[0] << endl << endl;
    vihso::SystemNode* SLAM = new vihso::SystemNode(argv[1], false, false);
    if (bFileName) SLAM->vo_->mFILE_NAME = file_name;

    int proccIm=0;
    for (seq = 0; seq<num_seq; seq++)
    {
        cv::Mat im;
        vector<IMU::IMUPoint> vImuMeas;
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        proccIm = 0;
        for(int ni=0; ni<nImages[seq]; ni++, proccIm++)
        {
            im = cv::imread(vstrImageFilenames[seq][ni], cv::IMREAD_UNCHANGED); 
            double tframe = vTimestampsCam[seq][ni];                            
            if(im.empty())
            {
                cerr << endl << "Failed to load image at: "
                     <<  vstrImageFilenames[seq][ni] << endl;
                return 1;
            }
            vImuMeas.clear();
            if(ni>0)    
            {
                while(vTimestampsImu[seq][first_imu[seq]]<=vTimestampsCam[seq][ni]) 
                {
                    vImuMeas.push_back(IMU::IMUPoint(vAcc[seq][first_imu[seq]].x(),vAcc[seq][first_imu[seq]].y(),vAcc[seq][first_imu[seq]].z(),
                                                    vGyro[seq][first_imu[seq]].x(),vGyro[seq][first_imu[seq]].y(),vGyro[seq][first_imu[seq]].z(),
                                                    vTimestampsImu[seq][first_imu[seq]]));
                    first_imu[seq]++;
                }
                first_imu[seq]--;   
            }
            struct timeval Track_start, Track_end;
            gettimeofday(&Track_start,NULL); 
                SLAM->TrackMonoInertial(im, tframe, vImuMeas);
            gettimeofday(&Track_end,NULL);
            double track_once = (Track_end.tv_sec-Track_start.tv_sec)*1000000+(Track_end.tv_usec-Track_start.tv_usec);  
            ttrack_total += track_once;
            vTimesTrack[ni] = track_once;
            SLAM->vo_->last_frame_->mProcessTime = track_once/1000;
            double T=0;
            if(ni<nImages[seq]-1)
                T = vTimestampsCam[seq][ni+1]-tframe;
            else if(ni>0)
                T = tframe-vTimestampsCam[seq][ni-1];
            T = T*1e6;  
            if(track_once<T)
                usleep(T-track_once);
        }   
    }   

    if (bFileName)
    {
        const string fFile =  "./f_" + string(argv[argc-1]) + ".tum";
        const string kfFile =  "./kf_" + string(argv[argc-1]) + ".tum";
        const string PointCloudFile =  "./pointclouds_" + string(argv[argc-1]) + ".txt";
        SLAM->SaveKeyFrameTrajectoryEuRoC(fFile, kfFile, PointCloudFile);
    }
    else
    {
        SLAM->SaveKeyFrameTrajectoryEuRoC("./fFile.tum", "./kfFile.tum");
    }
    delete SLAM;
	SLAM = NULL;
    return 0;
}
void LoadImages(const string &strImagePath, const string &strPathTimes,     
                vector<string> &vstrImages, vector<double> &vTimeStamps)    
{
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImages.reserve(5000);
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            vstrImages.push_back(strImagePath + "/" + ss.str() + ".png");
            double t;
            ss >> t;
            vTimeStamps.push_back(t/1e9);
        }
    }
    fTimes.close();
}
void LoadIMU(const string &strImuPath,                                                              
             vector<double> &vTimeStamps, vector<Eigen::Vector3f> &vAcc, vector<Eigen::Vector3f> &vGyro)    
{
    ifstream fImu;
    fImu.open(strImuPath.c_str());
    vTimeStamps.reserve(5000);  
    vAcc.reserve(5000);
    vGyro.reserve(5000);
    while(!fImu.eof())
    {
        string s;
        getline(fImu,s);
        if (s[0] == '#')
            continue;
        if(!s.empty())
        {
            string item;
            size_t pos = 0;
            double data[7];
            int count = 0;
            while ((pos = s.find(',')) != string::npos) 
            {
                item = s.substr(0, pos);    
                data[count++] = stod(item);
                s.erase(0, pos + 1);        
            }
            item = s.substr(0, pos);
            data[6] = stod(item);
            vTimeStamps.push_back(data[0]/1e9);
            vAcc.push_back(Eigen::Vector3f(data[4],data[5],data[6]));
            vGyro.push_back(Eigen::Vector3f(data[1],data[2],data[3]));
        }
    }
    fImu.close();
}
