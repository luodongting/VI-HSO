#include "SettingParameters.h"

	string CAMTYPE;
    double ROW, COL;
    int FREQ;
    double fx, fy, cx, cy;
    double k1, k2, p1, p2;

    vector<Eigen::Matrix3d> vRbc;
    vector<Eigen::Vector3d> vtbc;
    vector<Eigen::Matrix4d> vTbc;

    double ACC_N, ACC_W;
    double GYR_N, GYR_W;
    double IMUFREQ;
    double IMUFREQ_sqrt;
    Eigen::Vector3d G = Eigen::Vector3d(0.0, 0.0, 9.8);

    double  priorG0, priorA0, priorG1, priorA1, priorG2, priorA2;
    double  priorGBA0, priorABA0, priorGBA1, priorABA1, priorGBA2, priorABA2;

void readParameters(const string& strSettingsFile)
{
    cv::FileStorage fsSettings(strSettingsFile.c_str(), 	
                            cv::FileStorage::READ);         
    if(!fsSettings.isOpened())
    {
        cerr << "Failed to open settings file at: " << strSettingsFile << endl;
        exit(-1);
    }

	CAMTYPE = (string)fsSettings["Camera.type"];
    ROW = fsSettings["Camera.height"];
    COL = fsSettings["Camera.width"];
    FREQ = fsSettings["Camera.fps"];

    ACC_N = fsSettings["IMU.NoiseAcc"];
    ACC_W = fsSettings["IMU.AccWalk"];
    GYR_N = fsSettings["IMU.NoiseGyro"];
    GYR_W = fsSettings["IMU.GyroWalk"];
    G.z() = fsSettings["g_norm"];
    IMUFREQ = fsSettings["IMU.Frequency"];
    IMUFREQ_sqrt = sqrt(IMUFREQ);

    cv::Mat cv_T;
    fsSettings["Tbc"] >> cv_T;
	if(cv_T.at<float>(0,0) != 0.0f && cv_T.at<float>(1,1) != 0.0f)
	{
		Eigen::Matrix4d eigen_T;
		cv::cv2eigen(cv_T, eigen_T);
		Eigen::Matrix3d eigen_R = eigen_T.topLeftCorner(3,3);
		Eigen::Vector3d eigen_t = eigen_T.topRightCorner(3,1);
		Eigen::Quaterniond Q(eigen_R);
		eigen_R = Q.normalized();
		vRbc.push_back(eigen_R);
		vtbc.push_back(eigen_t);
		vTbc.push_back(eigen_T);
	}
	else
	{
		cv::Mat cv_q,cv_t;
		fsSettings["Quaternions"] >> cv_q;
		fsSettings["translation"] >> cv_t;
		Eigen::Vector4d eigen_q;
		cv::cv2eigen(cv_q, eigen_q);
		Eigen::Quaterniond Q(eigen_q[3],eigen_q[0],eigen_q[1],eigen_q[2]);
		Eigen::Matrix3d eigen_R = Q.normalized().toRotationMatrix();
		Eigen::Vector3d eigen_t;
		cv::cv2eigen(cv_t, eigen_t);
		Eigen::Matrix4d eigen_T = Eigen::Matrix4d::Identity();
		eigen_T.topLeftCorner(3,3) = eigen_R;
		eigen_T.topRightCorner(3,1) = eigen_t;

		vRbc.push_back(eigen_R);
		vtbc.push_back(eigen_t);
		vTbc.push_back(eigen_T);
	}
    
    fx = fsSettings["Camera.fx"];
    fy = fsSettings["Camera.fy"];
    cx = fsSettings["Camera.cx"];
    cy = fsSettings["Camera.cy"];
    k1 = fsSettings["Camera.k1"];
    k2 = fsSettings["Camera.k2"];
    p1 = fsSettings["Camera.p1"];
    p2 = fsSettings["Camera.p2"];
    
    cout << "Setting Parameters:  "
        <<  "\n  ROW:"<<ROW
        <<  "\n  COL:"<<COL
        <<  "\n  FREQ:"<<FREQ
        <<  "\n  IMUFREQ:"<<IMUFREQ
        << setprecision(6)
        <<  "\n  ACC_N: " <<ACC_N
        <<  "\n  ACC_W: " <<ACC_W
        <<  "\n  GYR_N: " <<GYR_N
        <<  "\n  GYR_W: " <<GYR_W
        <<  "\n  fx:" <<fx <<  " fy:" <<fy<<  " cx:" << cx <<  " cy:" << cy        
        <<  "\n  k1:" <<k1 <<  " k2:" <<k2 <<  " p1:" <<p1 <<  " p2:" <<p2
        <<  "\n  RIC:   " << vRbc[0]
        <<  "\n  TIC:   " << vtbc[0].transpose()
        <<  "\n  G:     " <<G.transpose()
        << endl;

    priorG0 = fsSettings["priorG0"];
    priorA0 = fsSettings["priorA0"];
    priorG1 = fsSettings["priorG1"];
    priorA1 = fsSettings["priorA1"];
    priorG2 = fsSettings["priorG2"];
    priorA2 = fsSettings["priorA2"];

    priorGBA0 = fsSettings["priorGBA0"];
    priorABA0 = fsSettings["priorABA0"];
    priorGBA1 = fsSettings["priorGBA1"];
    priorABA1 = fsSettings["priorABA1"];
    priorGBA2 = fsSettings["priorGBA2"];
    priorABA2 = fsSettings["priorABA2"];

	cout << "System starting ..." << endl;

    fsSettings.release();
}                   
