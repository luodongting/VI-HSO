#include "ImuTypes.h"
#include<iostream>
namespace IMU
{
const double eps = 1e-4;
Eigen::Matrix3d NormalizeRotation(const Eigen::Matrix3d &R)
{
    Eigen::Quaterniond Quat(R);
    Quat.normalize();
    Eigen::Matrix3d Rn = Quat.toRotationMatrix();
    return Rn;
}
Eigen::Matrix3d Skew(const Eigen::Vector3d &v)
{
    const double x = v(0,0);
    const double y = v(1,0);
    const double z = v(2,0);
    Eigen::Matrix3d W;
    W << 0, -z, y,
         z, 0, -x,
         -y, x, 0;
    return W;
}
Eigen::Matrix3d ExpSO3(const double &x, const double &y, const double &z)
{
    Eigen::Matrix3d I = Eigen::MatrixXd::Identity(3,3);
    const double d2 = x*x+y*y+z*z;
    const double d = sqrt(d2);
    Eigen::Matrix3d W;
    W(0,0) = 0;
    W(0,1) = -z;
    W(0,2) = y;
    W(1,0) = z;
    W(1,1) = 0;
    W(1,2) = -x;
    W(2,0) = -y;
    W(2,1) = x;
    W(2,2) = 0;
    if(d<eps)
        return (I + W + 0.5*W*W);
    else
        return (I + W*sin(d)/d + W*W*(1.0-cos(d))/d2);
}
Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &v)
{
    return ExpSO3(v(0,0),v(1,0),v(2,0));
}
Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R)
{
    const double tr = R(0,0)+R(1,1)+R(2,2);
    Eigen::Vector3d w;
    w <<(R(2,1)-R(1,2))/2, (R(0,2)-R(2,0))/2, (R(1,0)-R(0,1))/2;
    const double costheta = (tr-1.0f)*0.5f;
    if(costheta>1 || costheta<-1)
        return w;
    const double theta = acos(costheta);
    const double s = sin(theta);
    if(fabs(s)<eps)
        return w;
    else
        return theta*w/s;
}
Eigen::Matrix3d RightJacobianSO3(const double &x, const double &y, const double &z)
{
    Eigen::Matrix3d I = Eigen::MatrixXd::Identity(3,3);
    const double d2 = x*x+y*y+z*z;
    const double d = sqrt(d2);
    Eigen::Matrix3d W;
    W << 0, -z, y,
         z, 0, -x,
         -y, x, 0;
    if(d<eps)
    {
        return Eigen::MatrixXd::Identity(3,3);
    }
    else
    {
        return I - W*(1.0f-cos(d))/d2 + W*W*(d-sin(d))/(d2*d);
    }
}
Eigen::Matrix3d RightJacobianSO3(const Eigen::Vector3d &v)
{
    return RightJacobianSO3(v(0,0),v(1,0),v(2,0));
}
Eigen::Matrix3d InverseRightJacobianSO3(const double &x, const double &y, const double &z)
{
    Eigen::Matrix3d I = Eigen::MatrixXd::Identity(3,3);
    const double d2 = x*x+y*y+z*z;
    const double d = sqrt(d2);
    Eigen::Matrix3d W;
    W << 0, -z, y,
         z, 0, -x,
         -y, x, 0;
    if(d<eps)
    {
        return Eigen::MatrixXd::Identity(3,3);
    }
    else
    {
        return I + W/2 + W*W*(1.0f/d2 - (1.0f+cos(d))/(2.0f*d*sin(d)));
    }
}
Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d &v)
{
    return InverseRightJacobianSO3(v(0,0),v(1,0),v(2,0));
}
IntegratedRotation::IntegratedRotation(const Eigen::Vector3d &angVel, const Bias &imuBias, const double &time) :deltaT(time)
{
    const double x = (angVel.x()-imuBias.bwx)*time;
    const double y = (angVel.y()-imuBias.bwy)*time;
    const double z = (angVel.z()-imuBias.bwz)*time;
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    const double d2 = x*x+y*y+z*z;
    const double d = sqrt(d2);
    Eigen::Matrix3d W;
    W << 0, -z, y,
         z, 0, -x,
         -y, x, 0;
    if(d<eps)
    {
        deltaR = I + W;
        rightJ = Eigen::Matrix3d::Identity();
    }
    else
    {
        deltaR = I + W*sin(d)/d + W*W*(1.0f-cos(d))/d2;
        rightJ = I - W*(1.0f-cos(d))/d2 + W*W*(d-sin(d))/(d2*d);
    }
}
Preintegrated::Preintegrated(const Bias &b_, const Calib &calib)
{
    Nga = calib.Cov;            
    NgaWalk = calib.CovWalk;    
    Initialize(b_);             
}
Preintegrated::Preintegrated(Preintegrated* pImuPre): dT(pImuPre->dT), C(pImuPre->C), Info(pImuPre->Info),
    Nga(pImuPre->Nga), NgaWalk(pImuPre->NgaWalk), b(pImuPre->b), dR(pImuPre->dR), dV(pImuPre->dV),
    dP(pImuPre->dP), JRg(pImuPre->JRg), JVg(pImuPre->JVg), JVa(pImuPre->JVa), JPg(pImuPre->JPg),
    JPa(pImuPre->JPa), avgA(pImuPre->avgA), avgW(pImuPre->avgW), bu(pImuPre->bu), db(pImuPre->db), mvMeasurements(pImuPre->mvMeasurements)
{
}
void Preintegrated::CopyFrom(Preintegrated* pImuPre)
{
    std::cout << "Preintegrated: start clone" << std::endl;
    dT = pImuPre->dT;
    C = pImuPre->C;
    Info = pImuPre->Info;
    Nga = pImuPre->Nga;
    NgaWalk = pImuPre->NgaWalk;
    std::cout << "Preintegrated: first clone" << std::endl;
    b.CopyFrom(pImuPre->b);
    dR = pImuPre->dR;
    dV = pImuPre->dV;
    dP = pImuPre->dP;
    JRg = pImuPre->JRg;
    JVg = pImuPre->JVg;
    JVa = pImuPre->JVa;
    JPg = pImuPre->JPg;
    JPa = pImuPre->JPa;
    avgA = pImuPre->avgA;
    avgW = pImuPre->avgW;
    std::cout << "Preintegrated: second clone" << std::endl;
    bu.CopyFrom(pImuPre->bu);
    db = pImuPre->db;
    std::cout << "Preintegrated: third clone" << std::endl;
    mvMeasurements = pImuPre->mvMeasurements;
    std::cout << "Preintegrated: end clone" << std::endl;
}
void Preintegrated::Initialize(const Bias &b_)
{
    dR = Eigen::Matrix3d::Identity();   
    dV = Eigen::Vector3d::Zero();       
    dP = Eigen::Vector3d::Zero();       
    JRg = Eigen::Matrix3d::Zero();      
    JVg = Eigen::Matrix3d::Zero();      
    JVa = Eigen::Matrix3d::Zero();      
    JPg = Eigen::Matrix3d::Zero();      
    JPa = Eigen::Matrix3d::Zero();      
    C = Eigen::Matrix<double,15,15>::Zero();    
    Info = Eigen::Matrix<double,15,15>::Zero(); 
    db = Eigen::Matrix<double,6,1>::Zero();     
    b=b_;
    bu=b_;                              
    avgA = Eigen::Vector3d::Zero();     
    avgW = Eigen::Vector3d::Zero();     
    dT=0.0f;                            
    mvMeasurements.clear();
}
void Preintegrated::Reintegrate()
{
    std::unique_lock<std::mutex> lock(mMutex);
    const std::vector<integrable> aux = mvMeasurements;
    Initialize(bu);
    for(size_t i=0;i<aux.size();i++)
        IntegrateNewMeasurement(aux[i].a,aux[i].w,aux[i].t);
}
void Preintegrated::IntegrateNewMeasurement(const Eigen::Vector3d &acceleration, const Eigen::Vector3d &angVel, const double &dt)
{
    mvMeasurements.push_back(integrable(acceleration,angVel,dt));
    Eigen::Matrix<double,9,9> A = Eigen::Matrix<double,9,9>::Identity();
    Eigen::Matrix<double,9,6> B = Eigen::Matrix<double,9,6>::Zero();    
    Eigen::Vector3d acc(acceleration.x()-b.bax, acceleration.y()-b.bay, acceleration.z()-b.baz);
    Eigen::Vector3d accW(angVel.x()-b.bwx, angVel.y()-b.bwy, angVel.z()-b.bwz);
    avgA = (dT*avgA + dR*acc*dt)/(dT+dt);
    avgW = (dT*avgW + accW*dt)/(dT+dt);
    dP = dP + dV*dt + 0.5f*dR*acc*dt*dt;
    dV = dV + dR*acc*dt;
    Eigen::Matrix3d Wacc;
    Wacc << 0, -acc(2,0), acc(1,0),
            acc(2,0), 0, -acc(0,0),
            -acc(1,0), acc(0,0), 0;   
    A.block<3,3>(3,0) = -dR*dt*Wacc;
    A.block<3,3>(6,0) = -0.5f*dR*dt*dt*Wacc;
    A.block<3,3>(6,3) = Eigen::Matrix3d::Identity()*dt;
    B.block<3,3>(3,3) = dR*dt;
    B.block<3,3>(6,3) = 0.5f*dR*dt*dt;
    JPa = JPa + JVa*dt -0.5f*dR*dt*dt;
    JPg = JPg + JVg*dt -0.5f*dR*dt*dt*Wacc*JRg;
    JVa = JVa - dR*dt;
    JVg = JVg - dR*dt*Wacc*JRg;
    IntegratedRotation dRi(angVel,b,dt);    
    dR = NormalizeRotation(dR*dRi.deltaR);  
    A.block<3,3>(0,0) = dRi.deltaR.transpose();
    B.block<3,3>(0,0) = dRi.rightJ*dt;
    C.block<9,9>(0,0) = A*C.block<9,9>(0,0)*A.transpose() + B*Nga*B.transpose();
    C.block<6,6>(9,9) = C.block<6,6>(9,9) + NgaWalk;                            
    JRg = dRi.deltaR.transpose()*JRg - dRi.rightJ*dt;
    dT += dt;
}
void Preintegrated::MergePrevious(Preintegrated* pPrev)
{
    if (pPrev==this)
        return;
    std::unique_lock<std::mutex> lock1(mMutex);
    std::unique_lock<std::mutex> lock2(pPrev->mMutex);
    Bias bav;
    bav.bwx = bu.bwx;
    bav.bwy = bu.bwy;
    bav.bwz = bu.bwz;
    bav.bax = bu.bax;
    bav.bay = bu.bay;
    bav.baz = bu.baz;
    const std::vector<integrable > aux1 = pPrev->mvMeasurements;
    const std::vector<integrable> aux2 = mvMeasurements;
    Initialize(bav);
    for(size_t i=0;i<aux1.size();i++)
        IntegrateNewMeasurement(aux1[i].a,aux1[i].w,aux1[i].t);
    for(size_t i=0;i<aux2.size();i++)
        IntegrateNewMeasurement(aux2[i].a,aux2[i].w,aux2[i].t);
}
void Preintegrated::SetNewBias(const Bias &bu_)
{
    std::unique_lock<std::mutex> lock(mMutex);
    bu = bu_;
    db(0,0) = bu_.bwx-b.bwx;
    db(1,0) = bu_.bwy-b.bwy;
    db(2,0) = bu_.bwz-b.bwz;
    db(3,0) = bu_.bax-b.bax;
    db(4,0) = bu_.bay-b.bay;
    db(5,0) = bu_.baz-b.baz;
}
IMU::Bias Preintegrated::GetDeltaBias(const Bias &b_)
{
    std::unique_lock<std::mutex> lock(mMutex);
    return IMU::Bias(b_.bax-b.bax,b_.bay-b.bay,b_.baz-b.baz,b_.bwx-b.bwx,b_.bwy-b.bwy,b_.bwz-b.bwz);
}
Eigen::Matrix3d Preintegrated::GetDeltaRotation(const Bias &b_)
{    
    std::unique_lock<std::mutex> lock(mMutex);
    Eigen::Vector3d dbg;
    dbg << b_.bwx-b.bwx, b_.bwy-b.bwy, b_.bwz-b.bwz;    
    return NormalizeRotation(dR*ExpSO3(JRg*dbg));   
}
Eigen::Vector3d Preintegrated::GetDeltaVelocity(const Bias &b_)
{
    std::unique_lock<std::mutex> lock(mMutex);
    Eigen::Vector3d dbg;
    dbg << b_.bwx-b.bwx, b_.bwy-b.bwy, b_.bwz-b.bwz;
    Eigen::Vector3d dba;
    dba << b_.bax-b.bax, b_.bay-b.bay, b_.baz-b.baz;
    return dV + JVg*dbg + JVa*dba;
}
Eigen::Vector3d Preintegrated::GetDeltaPosition(const Bias &b_)
{
    std::unique_lock<std::mutex> lock(mMutex);
    Eigen::Vector3d dbg;
    dbg << b_.bwx-b.bwx, b_.bwy-b.bwy, b_.bwz-b.bwz;
    Eigen::Vector3d dba;
    dba << b_.bax-b.bax, b_.bay-b.bay, b_.baz-b.baz;
    return dP + JPg*dbg + JPa*dba;
}
Eigen::Matrix3d Preintegrated::GetUpdatedDeltaRotation()
{
    std::unique_lock<std::mutex> lock(mMutex);
    return NormalizeRotation(dR*ExpSO3(JRg*db.segment(0,3)));
}
Eigen::Vector3d Preintegrated::GetUpdatedDeltaVelocity()
{
    std::unique_lock<std::mutex> lock(mMutex);
    return dV + JVg*db.segment(0,3) + JVa*db.segment(3,3);
}
Eigen::Vector3d Preintegrated::GetUpdatedDeltaPosition()
{
    std::unique_lock<std::mutex> lock(mMutex);
    return dP + JPg*db.segment(0,3) + JPa*db.segment(3,3);
}
Eigen::Matrix3d Preintegrated::GetOriginalDeltaRotation()
{
    std::unique_lock<std::mutex> lock(mMutex);
    return dR;
}
Eigen::Vector3d Preintegrated::GetOriginalDeltaVelocity()
{
    std::unique_lock<std::mutex> lock(mMutex);
    return dV;
}
Eigen::Vector3d Preintegrated::GetOriginalDeltaPosition()
{
    std::unique_lock<std::mutex> lock(mMutex);
    return dP;
}
Bias Preintegrated::GetOriginalBias()
{
    std::unique_lock<std::mutex> lock(mMutex);
    return b;
}
Bias Preintegrated::GetUpdatedBias()
{
    std::unique_lock<std::mutex> lock(mMutex);
    return bu;
}
Eigen::Matrix<double,6,1> Preintegrated::GetDeltaBias()
{
    std::unique_lock<std::mutex> lock(mMutex);
    return db;
}
Eigen::Matrix<double,15,15> Preintegrated::GetInformationMatrix()
{
    std::unique_lock<std::mutex> lock(mMutex);
    Info = Eigen::Matrix<double,15,15>::Zero();
    Info.block<9,9>(0,0)= C.block<9,9>(0,0).inverse();
    for(int i=9;i<15;i++)
        Info(i,i)=1.0f/C(i,i);
    Eigen::Matrix<double,15,15> EI;
    for(int i=0;i<15;i++)
        for(int j=0;j<15;j++)
            EI(i,j)=Info(i,j);
    return EI;
}
void Bias::CopyFrom(Bias &b)
{
    bax = b.bax;
    bay = b.bay;
    baz = b.baz;
    bwx = b.bwx;
    bwy = b.bwy;
    bwz = b.bwz;
}
std::ostream& operator<< (std::ostream &out, const Bias &b)
{
    if(b.bwx>0)
        out << " ";
    out << b.bwx << ",";
    if(b.bwy>0)
        out << " ";
    out << b.bwy << ",";
    if(b.bwz>0)
        out << " ";
    out << b.bwz << ",";
    if(b.bax>0)
        out << " ";
    out << b.bax << ",";
    if(b.bay>0)
        out << " ";
    out << b.bay << ",";
    if(b.baz>0)
        out << " ";
    out << b.baz;
    return out;
}
Calib::Calib(const Eigen::Matrix4d &Tbc_, const double &ng, const double &na, const double &ngw, const double &naw)
{
    Set(Tbc_,ng,na,ngw,naw);
}
void Calib::Set(const Eigen::Matrix4d &Tbc_, const double &ng, const double &na, const double &ngw, const double &naw)
{
    Tbc = Tbc_;
    Tcb = Tbc.inverse();
    SE3_Tbc = Sophus::SE3(Tbc.block<3,3>(0,0), Tbc.block<3,1>(0,3));
    SE3_Tcb = Sophus::SE3(Tcb.block<3,3>(0,0), Tcb.block<3,1>(0,3));
    Cov = Eigen::Matrix<double, 6, 6>::Identity();
    const double ng2 = ng*ng;
    const double na2 = na*na;
    Cov(0,0) = ng2;
    Cov(1,1) = ng2;
    Cov(2,2) = ng2;
    Cov(3,3) = na2;
    Cov(4,4) = na2;
    Cov(5,5) = na2;
    CovWalk = Eigen::Matrix<double, 6, 6>::Identity();
    const double ngw2 = ngw*ngw;
    const double naw2 = naw*naw;
    CovWalk(0,0) = ngw2;
    CovWalk(1,1) = ngw2;
    CovWalk(2,2) = ngw2;
    CovWalk(3,3) = naw2;
    CovWalk(4,4) = naw2;
    CovWalk(5,5) = naw2;
}
Calib::Calib(const Calib &calib)
{
    Tbc = calib.Tbc;
    Tcb = calib.Tcb;
    Cov = calib.Cov;
    CovWalk = calib.CovWalk;
}
} 
