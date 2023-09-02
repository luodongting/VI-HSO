#ifndef IMUTYPES_H_
#define IMUTYPES_H_
#include<vector>
#include<utility>
#include<opencv2/core/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <mutex>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include "transformation.h"
#include "SettingParameters.h"
#include <sophus/se3.h>

using namespace Eigen;

namespace IMU
{
const double GRAVITY_VALUE=9.81;
class IMUPoint
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    IMUPoint(const double &acc_x, const double &acc_y, const double &acc_z,
             const double &ang_vel_x, const double &ang_vel_y, const double &ang_vel_z,
             const double &timestamp): a(acc_x,acc_y,acc_z), w(ang_vel_x,ang_vel_y,ang_vel_z), t(timestamp){}
    IMUPoint(const Eigen::Vector3d Acc, const Eigen::Vector3d Gyro, const double &timestamp):
                a(Acc.x(),Acc.y(),Acc.z()), w(Gyro.x(),Gyro.y(),Gyro.z()), t(timestamp){}
public:
    Eigen::Vector3d a;
    Eigen::Vector3d w;
    double t;   
};
class Bias
{
 public:
   Bias():bax(0),bay(0),baz(0),bwx(0),bwy(0),bwz(0){}
   Bias(const double &b_acc_x, const double &b_acc_y, const double &b_acc_z,
         const double &b_ang_vel_x, const double &b_ang_vel_y, const double &b_ang_vel_z):
         bax(b_acc_x), bay(b_acc_y), baz(b_acc_z), bwx(b_ang_vel_x), bwy(b_ang_vel_y), bwz(b_ang_vel_z){}
   void CopyFrom(Bias &b); 
   friend std::ostream& operator<< (std::ostream &out, const Bias &b);
 public:
   double bax, bay, baz;    
   double bwx, bwy, bwz;    
};
class Calib
{
 public:
   EIGEN_MAKE_ALIGNED_OPERATOR_NEW
   Calib(const Eigen::Matrix4d &Tbc_, const double &ng, const double &na, const double &ngw, const double &naw);
   Calib(const Calib &calib);
   Calib(){};
   void Set(const Eigen::Matrix4d &Tbc_, const double &ng, const double &na, const double &ngw, const double &naw);
 public:
   Eigen::Matrix4d Tbc;            
   Eigen::Matrix4d Tcb;            
   Sophus::SE3 SE3_Tbc;
   Sophus::SE3 SE3_Tcb;
   Eigen::Matrix<double,6,6> Cov, CovWalk;   
};
class IntegratedRotation
{
 public:
   EIGEN_MAKE_ALIGNED_OPERATOR_NEW
   IntegratedRotation(){}
   IntegratedRotation(const Eigen::Vector3d &angVel, const Bias &imuBias, const double &time);
 public:
   double deltaT;   
   Eigen::Matrix3d deltaR; 
   Eigen::Matrix3d rightJ; 
};
class Preintegrated
{
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Preintegrated(const Bias &b_, const Calib &calib);
    Preintegrated(Preintegrated* pImuPre);
    Preintegrated() {}
    ~Preintegrated() {}
    void CopyFrom(Preintegrated* pImuPre);
    void Initialize(const Bias &b_);
    void IntegrateNewMeasurement(const Eigen::Vector3d &acceleration, const Eigen::Vector3d &angVel, const double &dt);
    void Reintegrate();
    void MergePrevious(Preintegrated* pPrev);
    void SetNewBias(const Bias &bu_);
    IMU::Bias GetDeltaBias(const Bias &b_);
    Eigen::Matrix3d GetDeltaRotation(const Bias &b_);
    Eigen::Vector3d GetDeltaVelocity(const Bias &b_);
    Eigen::Vector3d GetDeltaPosition(const Bias &b_);
    Eigen::Matrix3d GetUpdatedDeltaRotation();
    Eigen::Vector3d GetUpdatedDeltaVelocity();
    Eigen::Vector3d GetUpdatedDeltaPosition();
    Eigen::Matrix3d GetOriginalDeltaRotation();
    Eigen::Vector3d GetOriginalDeltaVelocity();
    Eigen::Vector3d GetOriginalDeltaPosition();
    Eigen::Matrix<double,15,15> GetInformationMatrix();
    Eigen::Matrix<double,6,1> GetDeltaBias();
    Bias GetOriginalBias();
    Bias GetUpdatedBias();
 public:
    double dT;                                   
    Eigen::Matrix<double, 15, 15> C;             
    Eigen::Matrix<double, 15, 15> Info;          
    Eigen::Matrix<double, 6, 6> Nga, NgaWalk;    
    Bias b;                 
    Eigen::Matrix3d dR;     
    Eigen::Vector3d dV;     
    Eigen::Vector3d dP;     
    Eigen::Matrix3d JRg, JVg, JVa, JPg, JPa;    
    Eigen::Vector3d avgA;           
    Eigen::Vector3d avgW;           
 public:
    Bias bu;    
    Eigen::Matrix<double,6,1> db; 
    struct integrable
    {
        integrable(const Eigen::Vector3d &a_, const Eigen::Vector3d &w_ , const double &t_):a(a_),w(w_),t(t_){}
        Eigen::Vector3d a;
        Eigen::Vector3d w;
        double t;
    };
    std::vector<integrable> mvMeasurements; 
    std::mutex mMutex;
};
Eigen::Matrix3d ExpSO3(const double &x, const double &y, const double &z);
Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &v);
Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R);
Eigen::Matrix3d RightJacobianSO3(const double &x, const double &y, const double &z);
Eigen::Matrix3d RightJacobianSO3(const Eigen::Vector3d &v);
Eigen::Matrix3d InverseRightJacobianSO3(const double &x, const double &y, const double &z);
Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d &v);
Eigen::Matrix3d Skew(const Eigen::Vector3d &v);
Eigen::Matrix3d NormalizeRotation(const Eigen::Matrix3d &R);
}
#endif 
