#ifndef SETPA_H_
#define SETPA_H_

#include "transformation.h"
#include <fstream>
#include <iostream>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iomanip>

using namespace std;
using namespace cv;

	extern string CAMTYPE;
    extern double ROW, COL;
    extern int FREQ;
    extern double fx, fy, cx, cy;
    extern double k1, k2, p1, p2;

    extern vector<Eigen::Matrix3d> vRbc;
    extern vector<Eigen::Vector3d> vtbc;
    extern vector<Eigen::Matrix4d> vTbc;

    extern double ACC_N, ACC_W;
    extern double GYR_N, GYR_W;
    extern double IMUFREQ;
    extern double IMUFREQ_sqrt;
    extern Eigen::Vector3d G;

    void readParameters(const string& strSettingsFile);

    
    enum SIZE_PARAMETERIZATION
    {
        SIZE_POSE = 7,
        SIZE_SPEEDBIAS = 9,
        SIZE_FEATURE = 1
    };

    enum StateOrder
    {
        O_P = 0,
        O_R = 3,
        O_V = 6,
        O_BA = 9,
        O_BG = 12
    };

    enum NoiseOrder
    {
        O_AN = 0,
        O_GN = 3,
        O_AW = 6,
        O_GW = 9
    };

    extern double  priorG0, priorA0, priorG1, priorA1, priorG2, priorA2;   
    extern double  priorGBA0, priorABA0, priorGBA1, priorABA1, priorGBA2, priorABA2;

#endif  //SETPA_H_