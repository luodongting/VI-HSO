#include "G2oTypes.h"
#include "ImuTypes.h"
#include "vihso/vikit/math_utils.h"
namespace vihso
{
ImuCamPose::ImuCamPose(FramePtr pF):its(0)
{
    twb = pF->GetImuPosition();
    Rwb = pF->GetImuRotation();
    tcw = pF->GetTranslation();
    Rcw = pF->GetRotation();
    tcb = pF->mImuCalib.SE3_Tcb.translation();
    Rcb = pF->mImuCalib.SE3_Tcb.rotation_matrix();
    Rbc = Rcb.transpose();
    tbc = pF->mImuCalib.SE3_Tbc.translation();
    Rwb0 = Rwb;
    DR.setIdentity();
    mpF = pF.get();
}
ImuCamPose::ImuCamPose(Frame* pF):its(0)
{
    twb = pF->GetImuPosition();
    Rwb = pF->GetImuRotation();
    tcw = pF->GetTranslation();
    Rcw = pF->GetRotation();
    tcb = pF->mImuCalib.SE3_Tcb.translation();
    Rcb = pF->mImuCalib.SE3_Tcb.rotation_matrix();
    Rbc = Rcb.transpose();
    tbc = pF->mImuCalib.SE3_Tbc.translation();
    Rwb0 = Rwb;
    DR.setIdentity();
    mpF = pF;
}
ImuCamPose::ImuCamPose(Eigen::Matrix3d &_Rwc, Eigen::Vector3d &_twc, FramePtr pF): its(0)
{
    tcb = pF->mImuCalib.SE3_Tcb.translation();
    Rcb = pF->mImuCalib.SE3_Tcb.rotation_matrix();
    Rbc = Rcb.transpose();
    tbc = pF->mImuCalib.SE3_Tbc.translation();
    twb = _Rwc*tcb+_twc;
    Rwb = _Rwc*Rcb;
    Rcw = _Rwc.transpose();
    tcw = -Rcw*_twc;
    Rwb0 = Rwb;
    DR.setIdentity();
    mpF = pF.get();
}
void ImuCamPose::SetParam(const Eigen::Matrix3d &_Rcw, const Eigen::Vector3d &_tcw, 
                          const Eigen::Matrix3d &_Rbc, const Eigen::Vector3d &_tbc)
{
    Rbc = _Rbc;
    tbc = _tbc;
    Rcw = _Rcw;
    tcw = _tcw;
    Rcb = Rbc.transpose();
    tcb = -Rcb*tbc;
    Rwb = Rcw.transpose()*Rcb;
    twb = Rcw.transpose()*(tcb-tcw);
}
Eigen::Vector2d ImuCamPose::Project(const Eigen::Vector3d &Xw) const
{
    Eigen::Vector3d Xc = Rcw*Xw + tcw;
    return mpF->f2c(Xc);
}
Eigen::Vector2d ImuCamPose::Project2d(const Eigen::Vector3d &Xw) const
{
    Eigen::Vector3d Xc = Rcw*Xw + tcw;
    return hso::project2d(Xc);
}
bool ImuCamPose::isDepthPositive(const Eigen::Vector3d &Xw) const
{
    return (Rcw.row(2)*Xw+tcw(2))>0.0;
}
void ImuCamPose::Update(const double *pu)
{
    Eigen::Vector3d ur, ut;
    ur << pu[0], pu[1], pu[2];
    ut << pu[3], pu[4], pu[5];
    twb += Rwb*ut;
    Rwb = Rwb*ExpSO3(ur);
    its++;
    if(its>=3)
    {
        NormalizeRotation(Rwb);
        its=0;
    }
    const Eigen::Matrix3d Rbw = Rwb.transpose();
    const Eigen::Vector3d tbw = -Rbw*twb;
    Rcw = Rcb*Rbw;
    tcw = Rcb*tbw+tcb;
}
void ImuCamPose::UpdateW(const double *pu)
{
    Eigen::Vector3d ur, ut;
    ur << pu[0], pu[1], pu[2];
    ut << pu[3], pu[4], pu[5];
    const Eigen::Matrix3d dR = ExpSO3(ur);
    DR = dR*DR;
    Rwb = DR*Rwb0;
    twb += ut;
    its++;
    if(its>=5)
    {
        DR(0,2)=0.0;
        DR(1,2)=0.0;
        DR(2,0)=0.0;
        DR(2,1)=0.0;
        NormalizeRotation(DR);
        its=0;
    }
    const Eigen::Matrix3d Rbw = Rwb.transpose();
    const Eigen::Vector3d tbw = -Rbw*twb;
    Rcw = Rcb*Rbw;
    tcw = Rcb*tbw+tcb;
}
bool VertexPose::read(std::istream& is)
{
    Eigen::Matrix<double,3,3> Rcw;
    Eigen::Matrix<double,3,1> tcw;
    Eigen::Matrix<double,3,3> Rbc;
    Eigen::Matrix<double,3,1> tbc;
    for (int i=0; i<3; i++){
        for (int j=0; j<3; j++)
            is >> Rcw(i,j);
    }
    for (int i=0; i<3; i++){
        is >> tcw(i);
    }
    for (int i=0; i<3; i++){
        for (int j=0; j<3; j++)
            is >> Rbc(i,j);
    }
    for (int i=0; i<3; i++){
        is >> tbc(i);
    }
    _estimate.SetParam(Rcw,tcw,Rbc,tbc);
    updateCache();
    return true;
}
bool VertexPose::write(std::ostream& os) const
{
    Eigen::Matrix<double,3,3> Rcw = _estimate.Rcw;
    Eigen::Matrix<double,3,1> tcw = _estimate.tcw;
    Eigen::Matrix<double,3,3> Rbc = _estimate.Rbc;
    Eigen::Matrix<double,3,1> tbc = _estimate.tbc;
    for (int i=0; i<3; i++){
        for (int j=0; j<3; j++)
            os << Rcw(i,j) << " ";
    }
    for (int i=0; i<3; i++){
        os << tcw(i) << " ";
    }
    for (int i=0; i<3; i++){
        for (int j=0; j<3; j++)
            os << Rbc(i,j) << " ";
    }
    for (int i=0; i<3; i++){
        os << tbc(i) << " ";
    }
    return os.good();
}
void EdgeMono::linearizeOplus()
{
    const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[1]);
    const g2o::VertexSBAPointXYZ* VPoint = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
    const Eigen::Matrix3d &Rcw = VPose->estimate().Rcw;
    const Eigen::Vector3d &tcw = VPose->estimate().tcw;
    const Eigen::Vector3d Xc = Rcw*VPoint->estimate() + tcw;
    const Eigen::Vector3d Xb = VPose->estimate().Rbc*Xc+VPose->estimate().tbc;
    const Eigen::Matrix3d &Rcb = VPose->estimate().Rcb;
    const Eigen::Matrix<double,2,3> proj_jac = VPose->estimate().mpF->projectJacUV(Xc);
    _jacobianOplusXi = -proj_jac * Rcw; 
    Eigen::Matrix<double,3,6> SE3deriv;
    double x = Xb(0);
    double y = Xb(1);
    double z = Xb(2);
    SE3deriv << 0.0, z, -y, 1.0, 0.0, 0.0,
                -z , 0.0, x, 0.0, 1.0, 0.0,
                y , -x , 0.0, 0.0, 0.0, 1.0;   
    _jacobianOplusXj = proj_jac * Rcb * SE3deriv; 
}
void EdgeMonoOnlyPose::linearizeOplus()
{
    const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[0]);
    const Eigen::Matrix3d &Rcw = VPose->estimate().Rcw;
    const Eigen::Vector3d &tcw = VPose->estimate().tcw;
    const Eigen::Vector3d Xc = Rcw*Xw + tcw;
    const Eigen::Vector3d Xb = VPose->estimate().Rbc*Xc+VPose->estimate().tbc;  
    const Eigen::Matrix3d &Rcb = VPose->estimate().Rcb;
    Eigen::Matrix<double,2,3> proj_jac = VPose->estimate().mpF->projectJacUV(Xc);
    Eigen::Matrix<double,3,6> SE3deriv;
    double x = Xb(0);
    double y = Xb(1);
    double z = Xb(2);
    SE3deriv << 0.0, z,   -y, 1.0, 0.0, 0.0,
            -z , 0.0, x, 0.0, 1.0, 0.0,
            y ,  -x , 0.0, 0.0, 0.0, 1.0;   
    _jacobianOplusXi = proj_jac * Rcb * SE3deriv; 
}
VertexVelocity::VertexVelocity(FramePtr pF)
{
    setEstimate(pF->mVw);
}
VertexVelocity::VertexVelocity(Frame* pF)
{
    setEstimate(pF->mVw);
}
VertexGyroBias::VertexGyroBias(FramePtr pF)
{
    Eigen::Vector3d bg;
    bg << pF->mImuBias.bwx, pF->mImuBias.bwy,pF->mImuBias.bwz;
    setEstimate(bg);
}
VertexGyroBias::VertexGyroBias(Frame* pF)
{
    Eigen::Vector3d bg;
    bg << pF->mImuBias.bwx, pF->mImuBias.bwy,pF->mImuBias.bwz;
    setEstimate(bg);
}
VertexAccBias::VertexAccBias(FramePtr pF)
{
    Eigen::Vector3d ba;
    ba << pF->mImuBias.bax, pF->mImuBias.bay,pF->mImuBias.baz;
    setEstimate(ba);
}
VertexAccBias::VertexAccBias(Frame* pF)
{
    Eigen::Vector3d ba;
    ba << pF->mImuBias.bax, pF->mImuBias.bay,pF->mImuBias.baz;
    setEstimate(ba);
}
EdgeInertial::EdgeInertial(IMU::Preintegrated *pInt):JRg(pInt->JRg),
                                                    JVg(pInt->JVg), JPg(pInt->JPg), 
                                                    JVa(pInt->JVa), JPa(pInt->JPa), 
                                                    mpInt(pInt), dt(pInt->dT)
{
    resize(6);
    g << 0, 0, -IMU::GRAVITY_VALUE;
    Matrix9d Info = pInt->C.block<9,9>(0,0).inverse();
    Info = (Info+Info.transpose())/2;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,9,9> > es(Info);
     Eigen::Matrix<double,9,1> eigs = es.eigenvalues();
     for(int i=0;i<9;i++)
         if(eigs[i]<1e-12)
             eigs[i]=0;
    Info = es.eigenvectors()*eigs.asDiagonal()*es.eigenvectors().transpose();
    setInformation(Info);
}
void EdgeInertial::computeError()
{
    const VertexPose*       VP1 = static_cast<const VertexPose*>(_vertices[0]);
    const VertexVelocity*   VV1 = static_cast<const VertexVelocity*>(_vertices[1]);
    const VertexGyroBias*   VG1 = static_cast<const VertexGyroBias*>(_vertices[2]);
    const VertexAccBias*    VA1 = static_cast<const VertexAccBias*>(_vertices[3]);
    const VertexPose*       VP2 = static_cast<const VertexPose*>(_vertices[4]);
    const VertexVelocity*   VV2 = static_cast<const VertexVelocity*>(_vertices[5]);
    const IMU::Bias b1(VA1->estimate()[0],VA1->estimate()[1],VA1->estimate()[2],VG1->estimate()[0],VG1->estimate()[1],VG1->estimate()[2]);
    const Eigen::Matrix3d dR = mpInt->GetDeltaRotation(b1);
    const Eigen::Vector3d dV = mpInt->GetDeltaVelocity(b1);
    const Eigen::Vector3d dP = mpInt->GetDeltaPosition(b1);
    const Eigen::Vector3d er = LogSO3(dR.transpose()*VP1->estimate().Rwb.transpose()*VP2->estimate().Rwb);
    const Eigen::Vector3d ev = VP1->estimate().Rwb.transpose()*(VV2->estimate() - VV1->estimate() - g*dt) - dV;
    const Eigen::Vector3d ep = VP1->estimate().Rwb.transpose()*(VP2->estimate().twb - VP1->estimate().twb - VV1->estimate()*dt - g*dt*dt/2) - dP;
    _error << er, ev, ep;
}
void EdgeInertial::linearizeOplus()
{
    const VertexPose*       VP1 = static_cast<const VertexPose*>(_vertices[0]);
    const VertexVelocity*   VV1 = static_cast<const VertexVelocity*>(_vertices[1]);
    const VertexGyroBias*   VG1 = static_cast<const VertexGyroBias*>(_vertices[2]);
    const VertexAccBias*    VA1 = static_cast<const VertexAccBias*>(_vertices[3]);
    const VertexPose*       VP2 = static_cast<const VertexPose*>(_vertices[4]);
    const VertexVelocity*   VV2 = static_cast<const VertexVelocity*>(_vertices[5]);
    const IMU::Bias b1(VA1->estimate()[0],VA1->estimate()[1],VA1->estimate()[2],VG1->estimate()[0],VG1->estimate()[1],VG1->estimate()[2]);
    const IMU::Bias db = mpInt->GetDeltaBias(b1);
    Eigen::Vector3d dbg;
    dbg << db.bwx, db.bwy, db.bwz;
    const Eigen::Matrix3d Rwb1 = VP1->estimate().Rwb;
    const Eigen::Matrix3d Rbw1 = Rwb1.transpose();
    const Eigen::Matrix3d Rwb2 = VP2->estimate().Rwb;
    const Eigen::Matrix3d dR = mpInt->GetDeltaRotation(b1);
    const Eigen::Matrix3d eR = dR.transpose()*Rbw1*Rwb2;
    const Eigen::Vector3d er = LogSO3(eR);
    const Eigen::Matrix3d invJr = InverseRightJacobianSO3(er);
    _jacobianOplus[0].setZero();
    _jacobianOplus[0].block<3,3>(0,0) = -invJr*Rwb2.transpose()*Rwb1; 
    _jacobianOplus[0].block<3,3>(3,0) = Skew(Rbw1*(VV2->estimate() - VV1->estimate() - g*dt)); 
    _jacobianOplus[0].block<3,3>(6,0) = Skew(Rbw1*(VP2->estimate().twb - VP1->estimate().twb
                                                   - VV1->estimate()*dt - 0.5*g*dt*dt)); 
    _jacobianOplus[0].block<3,3>(6,3) = -Eigen::Matrix3d::Identity(); 
    _jacobianOplus[1].setZero();
    _jacobianOplus[1].block<3,3>(3,0) = -Rbw1; 
    _jacobianOplus[1].block<3,3>(6,0) = -Rbw1*dt; 
    _jacobianOplus[2].setZero();
    _jacobianOplus[2].block<3,3>(0,0) = -invJr*eR.transpose()*RightJacobianSO3(JRg*dbg)*JRg; 
    _jacobianOplus[2].block<3,3>(3,0) = -JVg; 
    _jacobianOplus[2].block<3,3>(6,0) = -JPg; 
    _jacobianOplus[3].setZero();
    _jacobianOplus[3].block<3,3>(3,0) = -JVa; 
    _jacobianOplus[3].block<3,3>(6,0) = -JPa; 
    _jacobianOplus[4].setZero();
    _jacobianOplus[4].block<3,3>(0,0) = invJr; 
    _jacobianOplus[4].block<3,3>(6,3) = Rbw1*Rwb2; 
    _jacobianOplus[5].setZero();
    _jacobianOplus[5].block<3,3>(3,0) = Rbw1; 
}
EdgeInertialGS::EdgeInertialGS(IMU::Preintegrated *pInt):JRg(pInt->JRg),
                                                        JVg(pInt->JVg), JPg(pInt->JPg), 
                                                        JVa(pInt->JVa), JPa(pInt->JPa), 
                                                        mpInt(pInt), dt(pInt->dT)
{
    resize(8);
    gI << 0, 0, -IMU::GRAVITY_VALUE;    
    Matrix9d Info = pInt->C.block<9,9>(0,0).inverse();
    Info = (Info+Info.transpose())/2;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,9,9> > es(Info); 
    Eigen::Matrix<double,9,1> eigs = es.eigenvalues();  
    for(int i=0;i<9;i++)
        if(eigs[i]<1e-12)
            eigs[i]=0;
    Info = es.eigenvectors()*eigs.asDiagonal()*es.eigenvectors().transpose();
    setInformation(Info);
}
void EdgeInertialGS::computeError()
{
    const VertexPose*       VP1 = static_cast<const VertexPose*>(_vertices[0]);
    const VertexVelocity*   VV1 = static_cast<const VertexVelocity*>(_vertices[1]);
    const VertexGyroBias*   VG  = static_cast<const VertexGyroBias*>(_vertices[2]);
    const VertexAccBias*    VA  = static_cast<const VertexAccBias*>(_vertices[3]);
    const VertexPose*       VP2 = static_cast<const VertexPose*>(_vertices[4]);
    const VertexVelocity*   VV2 = static_cast<const VertexVelocity*>(_vertices[5]);
    const VertexGDir*       VGDir = static_cast<const VertexGDir*>(_vertices[6]);
    const VertexScale*      VS  = static_cast<const VertexScale*>(_vertices[7]);
    const IMU::Bias b(VA->estimate()[0],VA->estimate()[1],VA->estimate()[2],VG->estimate()[0],VG->estimate()[1],VG->estimate()[2]);
    g = VGDir->estimate().Rwg*gI;
    const double s = VS->estimate();
    const Eigen::Matrix3d dR = mpInt->GetDeltaRotation(b);
    const Eigen::Vector3d dV = mpInt->GetDeltaVelocity(b);
    const Eigen::Vector3d dP = mpInt->GetDeltaPosition(b);
    const Eigen::Vector3d er = LogSO3(dR.transpose() * VP1->estimate().Rwb.transpose() * VP2->estimate().Rwb);
    const Eigen::Vector3d ev = VP1->estimate().Rwb.transpose() * (s*(VV2->estimate() - VV1->estimate()) - g*dt) - dV;
    const Eigen::Vector3d ep = VP1->estimate().Rwb.transpose()*(s*(VP2->estimate().twb - VP1->estimate().twb - VV1->estimate()*dt) - g*dt*dt/2) - dP;
    _error << er, ev, ep;
}
void EdgeInertialGS::linearizeOplus()
{
    const VertexPose*       VP1 = static_cast<const VertexPose*>(_vertices[0]);
    const VertexVelocity*   VV1 = static_cast<const VertexVelocity*>(_vertices[1]);
    const VertexGyroBias*   VG  = static_cast<const VertexGyroBias*>(_vertices[2]);
    const VertexAccBias*    VA  = static_cast<const VertexAccBias*>(_vertices[3]);
    const VertexPose*       VP2 = static_cast<const VertexPose*>(_vertices[4]);
    const VertexVelocity*   VV2 = static_cast<const VertexVelocity*>(_vertices[5]);
    const VertexGDir*       VGDir = static_cast<const VertexGDir*>(_vertices[6]);
    const VertexScale*      VS  = static_cast<const VertexScale*>(_vertices[7]);
    const IMU::Bias b(VA->estimate()[0], VA->estimate()[1], VA->estimate()[2], VG->estimate()[0], VG->estimate()[1], VG->estimate()[2]);
    const IMU::Bias db = mpInt->GetDeltaBias(b);
    Eigen::Vector3d dbg;
    dbg << db.bwx, db.bwy, db.bwz;
    const double s = VS->estimate();
    const Eigen::Matrix3d Rwb1 = VP1->estimate().Rwb;
    const Eigen::Matrix3d Rbw1 = Rwb1.transpose();
    const Eigen::Matrix3d Rwb2 = VP2->estimate().Rwb;
    const Eigen::Matrix3d Rwg = VGDir->estimate().Rwg;
    Eigen::MatrixXd Gm = Eigen::MatrixXd::Zero(3,2);
    Gm(0,1) = -IMU::GRAVITY_VALUE;
    Gm(1,0) = IMU::GRAVITY_VALUE;
    const Eigen::MatrixXd dGdTheta = Rwg*Gm;    
    const Eigen::Matrix3d dR = mpInt->GetDeltaRotation(b);
    const Eigen::Matrix3d eR = dR.transpose()*Rbw1*Rwb2;
    const Eigen::Vector3d er = LogSO3(eR);
    const Eigen::Matrix3d invJr = InverseRightJacobianSO3(er);  
    _jacobianOplus[0].setZero();
    _jacobianOplus[0].block<3,3>(0,0) = -invJr*Rwb2.transpose()*Rwb1;                               
    _jacobianOplus[0].block<3,3>(3,0) = Skew(Rbw1*(s*(VV2->estimate() - VV1->estimate()) - g*dt));  
    _jacobianOplus[0].block<3,3>(6,0) = Skew(Rbw1*(s*(VP2->estimate().twb - VP1->estimate().twb - VV1->estimate()*dt) - 0.5*g*dt*dt));  
    _jacobianOplus[0].block<3,3>(6,3) = -s*Eigen::Matrix3d::Identity(); 
    _jacobianOplus[1].setZero();
    _jacobianOplus[1].block<3,3>(3,0) = -s*Rbw1;    
    _jacobianOplus[1].block<3,3>(6,0) = -s*Rbw1*dt; 
    _jacobianOplus[2].setZero();
    _jacobianOplus[2].block<3,3>(0,0) = -invJr*eR.transpose()*RightJacobianSO3(JRg*dbg)*JRg;    
    _jacobianOplus[2].block<3,3>(3,0) = -JVg;   
    _jacobianOplus[2].block<3,3>(6,0) = -JPg;   
    _jacobianOplus[3].setZero();
    _jacobianOplus[3].block<3,3>(3,0) = -JVa;   
    _jacobianOplus[3].block<3,3>(6,0) = -JPa;   
    _jacobianOplus[4].setZero();
    _jacobianOplus[4].block<3,3>(0,0) = invJr;          
    _jacobianOplus[4].block<3,3>(6,3) = s*Rbw1*Rwb2;    
    _jacobianOplus[5].setZero();
    _jacobianOplus[5].block<3,3>(3,0) = s*Rbw1; 
    _jacobianOplus[6].setZero();
    _jacobianOplus[6].block<3,2>(3,0) = -Rbw1*dGdTheta*dt;          
    _jacobianOplus[6].block<3,2>(6,0) = -0.5*Rbw1*dGdTheta*dt*dt;   
    _jacobianOplus[7].setZero();
    _jacobianOplus[7].block<3,1>(3,0) = Rbw1*(VV2->estimate()-VV1->estimate()); 
    _jacobianOplus[7].block<3,1>(6,0) = Rbw1*(VP2->estimate().twb-VP1->estimate().twb-VV1->estimate()*dt);  
}
EdgePriorPoseImu::EdgePriorPoseImu(ConstraintPoseImu *c)
{
    resize(4);
    Rwb = c->Rwb;
    twb = c->twb;
    vwb = c->vwb;
    bg = c->bg;
    ba = c->ba;
    setInformation(c->H);
}
void EdgePriorPoseImu::computeError()
{
    const VertexPose* VP = static_cast<const VertexPose*>(_vertices[0]);
    const VertexVelocity* VV = static_cast<const VertexVelocity*>(_vertices[1]);
    const VertexGyroBias* VG = static_cast<const VertexGyroBias*>(_vertices[2]);
    const VertexAccBias* VA = static_cast<const VertexAccBias*>(_vertices[3]);
    const Eigen::Vector3d er = LogSO3(Rwb.transpose()*VP->estimate().Rwb);  
    const Eigen::Vector3d et = Rwb.transpose()*(VP->estimate().twb-twb);    
    const Eigen::Vector3d ev = VV->estimate() - vwb;                        
    const Eigen::Vector3d ebg = VG->estimate() - bg;                        
    const Eigen::Vector3d eba = VA->estimate() - ba;                        
    _error << er, et, ev, ebg, eba; 
}
void EdgePriorPoseImu::linearizeOplus()
{
    const VertexPose* VP = static_cast<const VertexPose*>(_vertices[0]);
    const Eigen::Vector3d er = LogSO3(Rwb.transpose()*VP->estimate().Rwb);
    _jacobianOplus[0].setZero();
    _jacobianOplus[0].block<3,3>(0,0) = InverseRightJacobianSO3(er);        
    _jacobianOplus[0].block<3,3>(3,3) = Rwb.transpose()*VP->estimate().Rwb; 
    _jacobianOplus[1].setZero();
    _jacobianOplus[1].block<3,3>(6,0) = Eigen::Matrix3d::Identity();
    _jacobianOplus[2].setZero();
    _jacobianOplus[2].block<3,3>(9,0) = Eigen::Matrix3d::Identity();
    _jacobianOplus[3].setZero();
    _jacobianOplus[3].block<3,3>(12,0) = Eigen::Matrix3d::Identity();
}
void EdgePriorAcc::linearizeOplus()
{
    _jacobianOplusXi.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
}
void EdgePriorGyro::linearizeOplus()
{
    _jacobianOplusXi.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
}
void EdgeMonoOnlyPoseCornor::linearizeOplus()
{
    const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[0]);
    const Eigen::Matrix3d &Rcw = VPose->estimate().Rcw;
    const Eigen::Vector3d &tcw = VPose->estimate().tcw;
    const Eigen::Vector3d Xc = Rcw*Xw + tcw;
    const Eigen::Vector3d Xb = VPose->estimate().Rbc*Xc+VPose->estimate().tbc;  
    const Eigen::Matrix3d &Rcb = VPose->estimate().Rcb;
    Matrix2d Mfxfy = Matrix2d::Zero();
    Mfxfy << vfxfy.x(), 0, 0, vfxfy.y();
    Eigen::Matrix<double,2,3> proj_jac = VPose->estimate().mpF->projectJac(Xc);
    Eigen::Matrix<double,3,6> SE3deriv;
    double x = Xb(0);
    double y = Xb(1);
    double z = Xb(2);
    SE3deriv << 0.0, z, -y,  1.0, 0.0, 0.0,
                -z, 0.0, x,  0.0, 1.0, 0.0,
                y, -x , 0.0, 0.0, 0.0, 1.0;   
    _jacobianOplusXi = Mfxfy * proj_jac * Rcb * SE3deriv;   
}
void EdgeMonoOnlyPoseEdgeLet::linearizeOplus()
{
    const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[0]);
    const Eigen::Matrix3d &Rcw = VPose->estimate().Rcw;
    const Eigen::Vector3d &tcw = VPose->estimate().tcw;
    const Eigen::Vector3d Xc = Rcw*Xw + tcw;
    const Eigen::Vector3d Xb = VPose->estimate().Rbc*Xc+VPose->estimate().tbc;  
    const Eigen::Matrix3d &Rcb = VPose->estimate().Rcb;
    Matrix2d Mfxfy = Matrix2d::Zero();
    Mfxfy << vfxfy.x(), 0, 0, vfxfy.y();
    Eigen::Matrix<double,2,3> proj_jac = VPose->estimate().mpF->projectJac(Xc);
    Eigen::Matrix<double,3,6> SE3deriv;
    double x = Xb(0);
    double y = Xb(1);
    double z = Xb(2);
    SE3deriv << 0.0, z,   -y, 1.0, 0.0, 0.0,
            -z , 0.0, x, 0.0, 1.0, 0.0,
            y ,  -x , 0.0, 0.0, 0.0, 1.0;
    _jacobianOplusXi = grad.transpose() * Mfxfy * proj_jac * Rcb * SE3deriv;    
}
void EdgeIHTCorner::computeError()
{
    const VertexIdist* VD = static_cast<const VertexIdist*>(_vertices[0]);  
    const VertexPose*  VH = static_cast<const VertexPose*>(_vertices[1]);   
    const VertexPose*  VT = static_cast<const VertexPose*>(_vertices[2]);   
    Frame* pKF = VT->estimate().mpF;
    Sophus::SE3 Ttw = Sophus::SE3(VT->estimate().Rcw,VT->estimate().tcw);
    Sophus::SE3 Thw = Sophus::SE3(VH->estimate().Rcw,VH->estimate().tcw);
    Sophus::SE3 Tth = Ttw*Thw.inverse();
    Vector3d fHost = VD->f_Host;
    Vector2d obs(_measurement);
    _error = obs - pKF->cam_->world2cam(Tth * (fHost*(1.0/VD->estimate()))) ;  
}
void EdgeIHTCorner::linearizeOplus()
{
    const VertexIdist* VD = static_cast<const VertexIdist*>(_vertices[0]);  
    const VertexPose*  VH = static_cast<const VertexPose*>(_vertices[1]);   
    const VertexPose*  VT = static_cast<const VertexPose*>(_vertices[2]);   
    double idHost = VD->estimate(); 
    Vector3d fHost = VD->f_Host;    
    Sophus::SE3 Thw_c = Sophus::SE3(VH->estimate().Rcw,VH->estimate().tcw);
    Sophus::SE3 Ttw_c = Sophus::SE3(VT->estimate().Rcw,VT->estimate().tcw);
    Sophus::SE3 Twh_b = Sophus::SE3(VH->estimate().Rwb,VH->estimate().twb);
    Sophus::SE3 Twt_b = Sophus::SE3(VT->estimate().Rwb,VT->estimate().twb);
    Sophus::SE3 Tbc   = Sophus::SE3(VT->estimate().Rbc,VT->estimate().tbc);
    Matrix3d    Rcb   = VH->estimate().Rcb;
    Sophus::SE3 Tth_c = Ttw_c * Thw_c.inverse();
    Vector3d tth_c = Tth_c.translation();       
    Matrix3d Rth_c = Tth_c.rotation_matrix();   
    Sophus::SE3 Tth_b = Twt_b.inverse() * Twh_b;
    Vector3d tth_b = Tth_b.translation();       
    Matrix3d Rth_b = Tth_b.rotation_matrix();   
    Matrix2d Mfxfy = Matrix2d::Zero();
    Mfxfy << vfxfy.x(), 0, 0, vfxfy.y();
    Vector3d Pt_c = Tth_c * (fHost*(1.0/idHost));   
    Matrix<double,2,3> JdNdt;
    double x = Pt_c[0], y = Pt_c[1], z = Pt_c[2];
    double z2 = z*z;
    JdNdt(0,0) = 1.0/z;     JdNdt(0,1) = 0.0;       JdNdt(0,2) = -x/z2;
    JdNdt(1,0) = 0;         JdNdt(1,1) = 1.0/z;     JdNdt(1,2) = -y/z2;   
    _jacobianOplus[0] = Mfxfy * JdNdt * Rth_c*fHost*(1.0/(idHost*idHost));
    Vector3d Ph_b = Tbc * (fHost*(1.0/idHost));   
    Matrix<double,3,6> JdtdTh;
    JdtdTh <<   0.0, Ph_b[2], -Ph_b[1], 1.0, 0.0, 0.0,
                -Ph_b[2], 0.0, Ph_b[0], 0.0, 1.0, 0.0,
                Ph_b[1], -Ph_b[0], 0.0, 0.0, 0.0, 1.0;
    JdtdTh = Rth_b * JdtdTh;
    _jacobianOplus[1] = -Mfxfy * JdNdt * Rcb * JdtdTh;
    Vector3d Pt_b = Tbc * Tth_c * (fHost*(1.0/idHost));   
    Matrix<double,3,6> JdtdTt;
    JdtdTt <<   0.0, Pt_b[2], -Pt_b[1], 1.0, 0.0, 0.0,  
                -Pt_b[2], 0.0, Pt_b[0], 0.0, 1.0, 0.0,
                Pt_b[1], -Pt_b[0], 0.0, 0.0, 0.0, 1.0;
    _jacobianOplus[2] = Mfxfy * JdNdt * Rcb * JdtdTt;
}
void EdgeIHTEdgeLet::computeError()
{
    const VertexIdist* VD = static_cast<const VertexIdist*>(_vertices[0]);  
    const VertexPose* VH = static_cast<const VertexPose*>(_vertices[1]);    
    const VertexPose* VT = static_cast<const VertexPose*>(_vertices[2]);    
    Frame* pKF = VT->estimate().mpF;
    Sophus::SE3 Ttw = Sophus::SE3(VT->estimate().Rcw,VT->estimate().tcw);
    Sophus::SE3 Thw = Sophus::SE3(VH->estimate().Rcw,VH->estimate().tcw);
    Sophus::SE3 Tth = Ttw*Thw.inverse();
    Vector3d fHost = VD->f_Host;
    double obs(_measurement);
    _error(0,0) = obs - _normal.transpose()*pKF->cam_->world2cam(Tth * (fHost*(1.0/VD->estimate()))) ;  
}
void EdgeIHTEdgeLet::linearizeOplus()
{
    const VertexIdist* VD = static_cast<const VertexIdist*>(_vertices[0]);  
    const VertexPose* VH = static_cast<const VertexPose*>(_vertices[1]);    
    const VertexPose* VT = static_cast<const VertexPose*>(_vertices[2]);    
    double idHost = VD->estimate(); 
    Vector3d fHost = VD->f_Host;    
    Sophus::SE3 Thw_c = Sophus::SE3(VH->estimate().Rcw,VH->estimate().tcw);
    Sophus::SE3 Ttw_c = Sophus::SE3(VT->estimate().Rcw,VT->estimate().tcw);
    Sophus::SE3 Twh_b = Sophus::SE3(VH->estimate().Rwb,VH->estimate().twb);
    Sophus::SE3 Twt_b = Sophus::SE3(VT->estimate().Rwb,VT->estimate().twb);
    Sophus::SE3 Tbc   = Sophus::SE3(VT->estimate().Rbc,VT->estimate().tbc);
    Matrix3d    Rcb   = VH->estimate().Rcb;
    Sophus::SE3 Tth_c = Ttw_c * Thw_c.inverse();
    Vector3d tth_c = Tth_c.translation();       
    Matrix3d Rth_c = Tth_c.rotation_matrix();   
    Sophus::SE3 Tth_b = Twt_b.inverse() * Twh_b;
    Vector3d tth_b = Tth_b.translation();       
    Matrix3d Rth_b = Tth_b.rotation_matrix();   
    Matrix2d Mfxfy = Matrix2d::Zero();
    Mfxfy << vfxfy.x(), 0, 0, vfxfy.y();
    Vector3d Pt_c = Tth_c * (fHost*(1.0/idHost));   
    Matrix<double,2,3> JdNdt;
    double x = Pt_c[0], y = Pt_c[1], z = Pt_c[2];
    double z2 = z*z;
    JdNdt(0,0) = 1.0/z;     JdNdt(0,1) = 0.0;       JdNdt(0,2) = -x/z2;
    JdNdt(1,0) = 0;         JdNdt(1,1) = 1.0/z;     JdNdt(1,2) = -y/z2;   
    _jacobianOplus[0] = _normal.transpose() * Mfxfy * JdNdt * Rth_c*fHost*(1.0/(idHost*idHost));
    Vector3d Ph_b = Tbc * (fHost*(1.0/idHost));   
    Matrix<double,3,6> JdtdTh;
    JdtdTh <<   0.0, Ph_b[2], -Ph_b[1], 1.0, 0.0, 0.0,
                -Ph_b[2], 0.0, Ph_b[0], 0.0, 1.0, 0.0,
                Ph_b[1], -Ph_b[0], 0.0, 0.0, 0.0, 1.0;
    JdtdTh = Rth_b * JdtdTh;
    _jacobianOplus[1] = -_normal.transpose() * Mfxfy * JdNdt * Rcb * JdtdTh; 
    Vector3d Pt_b = Tbc * Tth_c * (fHost*(1.0/idHost));   
    Matrix<double,3,6> JdtdTt;
    JdtdTt <<   0.0, Pt_b[2], -Pt_b[1], 1.0, 0.0, 0.0,  
                -Pt_b[2], 0.0, Pt_b[0], 0.0, 1.0, 0.0,
                Pt_b[1], -Pt_b[0], 0.0, 0.0, 0.0, 1.0;
    _jacobianOplus[2] = _normal.transpose() * Mfxfy * JdNdt * Rcb * JdtdTt;
}
void EdgePhotometric::computeError()
{
	const VertexPhotometricPose* VT21 = static_cast<const VertexPhotometricPose*>(_vertices[0]);
	const VertexExposureRatio* VExpRatio = static_cast<const VertexExposureRatio*>(_vertices[1]);
	double exposure_rat = VExpRatio->estimate();
	Xcur = VT21->estimate() * Xref;
	if(!computeConfig())
	{
		_error(0,0) = 0;
	}
	else
	{
		float residual = cur_color - (exposure_rat*_measurement + b); 
		float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
		if(level == 4)
		{
			Eigen::Matrix<double,1,1> Info(hw);
			setInformation(Info);
		}
		else
		{
			Eigen::Matrix<double,1,1> Info(hw*(2-hw));
			setInformation(Info);
		}
		_error(0,0) = residual;
	}
}
void EdgePhotometric::linearizeOplus()
{
	if(!binverse)	
	{
		uint8_t* cur_img_ptr = (uint8_t*)curImg.data + (v_cur_i + patternRow)*stride + u_cur_i + patternCol;
		float dx = 0.5f * ((w_cur_tl*cur_img_ptr[1]       + w_cur_tr*cur_img_ptr[2]        + w_cur_bl*cur_img_ptr[stride+1] + w_cur_br*cur_img_ptr[stride+2])
							-(w_cur_tl*cur_img_ptr[-1]      + w_cur_tr*cur_img_ptr[0]        + w_cur_bl*cur_img_ptr[stride-1] + w_cur_br*cur_img_ptr[stride]));
		float dy = 0.5f * ((w_cur_tl*cur_img_ptr[stride]  + w_cur_tr*cur_img_ptr[1+stride] + w_cur_bl*cur_img_ptr[stride*2] + w_cur_br*cur_img_ptr[stride*2+1])
							-(w_cur_tl*cur_img_ptr[-stride] + w_cur_tr*cur_img_ptr[1-stride] + w_cur_bl*cur_img_ptr[0]        + w_cur_br*cur_img_ptr[1]));
		Vector6d J_T = dx*frame_jac.row(0)*fxl + dy*frame_jac.row(1)*fyl;   
		Eigen::Matrix<double,1,1> J_e(-_measurement);
		_jacobianOplusXi = J_T;
    	_jacobianOplusXj = J_e;
	}
	else	
	{
		Eigen::Matrix<double,1,1> J_e(-_measurement);
		_jacobianOplusXi = J_inv;
		_jacobianOplusXj = J_e;
	}
}
Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &w)
{
    return ExpSO3(w[0],w[1],w[2]);
}
Eigen::Matrix3d ExpSO3(const double x, const double y, const double z)
{
    const double d2 = x*x+y*y+z*z;
    const double d = sqrt(d2);
    Eigen::Matrix3d W;
    W << 0.0, -z, y,z, 0.0, -x,-y,  x, 0.0;
    if(d<1e-5)
    {
        Eigen::Matrix3d res = Eigen::Matrix3d::Identity() + W +0.5*W*W;
        return IMU::NormalizeRotation(res);
    }
    else
    {
        Eigen::Matrix3d res =Eigen::Matrix3d::Identity() + W*sin(d)/d + W*W*(1.0-cos(d))/d2;
        return IMU::NormalizeRotation(res);
    }
}
Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R)
{
    const double tr = R(0,0)+R(1,1)+R(2,2);
    Eigen::Vector3d w;
    w << (R(2,1)-R(1,2))/2, (R(0,2)-R(2,0))/2, (R(1,0)-R(0,1))/2;
    const double costheta = (tr-1.0)*0.5f;
    if(costheta>1 || costheta<-1)
        return w;
    const double theta = acos(costheta);
    const double s = sin(theta);
    if(fabs(s)<1e-5)
        return w;
    else
        return theta*w/s;
}
Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d &v)
{
    return InverseRightJacobianSO3(v[0],v[1],v[2]);
}
Eigen::Matrix3d InverseRightJacobianSO3(const double x, const double y, const double z)
{
    const double d2 = x*x+y*y+z*z;
    const double d = sqrt(d2);
    Eigen::Matrix3d W;
    W << 0.0, -z, y,z, 0.0, -x,-y,  x, 0.0;
    if(d<1e-5)
        return Eigen::Matrix3d::Identity();
    else
        return Eigen::Matrix3d::Identity() + W/2 + W*W*(1.0/d2 - (1.0+cos(d))/(2.0*d*sin(d)));
}
Eigen::Matrix3d RightJacobianSO3(const Eigen::Vector3d &v)
{
    return RightJacobianSO3(v[0],v[1],v[2]);
}
Eigen::Matrix3d RightJacobianSO3(const double x, const double y, const double z)
{
    const double d2 = x*x+y*y+z*z;
    const double d = sqrt(d2);
    Eigen::Matrix3d W;
    W << 0.0, -z, y,z, 0.0, -x,-y,  x, 0.0;
    if(d<1e-5)
    {
        return Eigen::Matrix3d::Identity();
    }
    else
    {
        return Eigen::Matrix3d::Identity() - W*(1.0-cos(d))/d2 + W*W*(d-sin(d))/(d2*d);
    }
}
Eigen::Matrix3d Skew(const Eigen::Vector3d &w)
{
    Eigen::Matrix3d W;
    W << 0.0, -w[2], w[1],w[2], 0.0, -w[0],-w[1],  w[0], 0.0;
    return W;
}
Eigen::Matrix3d NormalizeRotation(const Eigen::Matrix3d &R)
{
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R,Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.matrixU()*svd.matrixV();
}
}
