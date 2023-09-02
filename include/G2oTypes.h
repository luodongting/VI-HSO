#ifndef G2OTYPES_H
#define G2OTYPES_H
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/types/types_sba.h>
#include <g2o/core/base_multi_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/types/types_six_dof_expmap.h>
#include <g2o/types/se3quat.h>
#include <g2o/core/robust_kernel_impl.h>
#include<opencv2/core/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include "vihso/frame.h"
#include "vihso/point.h"
#include "vihso/global.h"
#include <math.h>
namespace vihso
{
class Frame;
class Point;
struct Feature;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 9, 1> Vector9d;
typedef Eigen::Matrix<double, 12, 1> Vector12d;
typedef Eigen::Matrix<double, 15, 1> Vector15d;
typedef Eigen::Matrix<double, 12, 12> Matrix12d;
typedef Eigen::Matrix<double, 15, 15> Matrix15d;
typedef Eigen::Matrix<double, 9, 9> Matrix9d;
Eigen::Matrix3d ExpSO3(const double x, const double y, const double z);
Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &w);
Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R);
Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d &v);
Eigen::Matrix3d RightJacobianSO3(const Eigen::Vector3d &v);
Eigen::Matrix3d RightJacobianSO3(const double x, const double y, const double z);
Eigen::Matrix3d Skew(const Eigen::Vector3d &w);
Eigen::Matrix3d InverseRightJacobianSO3(const double x, const double y, const double z);
Eigen::Matrix3d NormalizeRotation(const Eigen::Matrix3d &R);
class ImuCamPose
{
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ImuCamPose(){}
    ImuCamPose(FramePtr pF);
    ImuCamPose(Frame* pF);
    ImuCamPose(Eigen::Matrix3d &_Rwc, Eigen::Vector3d &_twc, FramePtr pF);
    void SetParam(const Eigen::Matrix3d &_Rcw, const Eigen::Vector3d &_tcw, 
                  const Eigen::Matrix3d &_Rbc, const Eigen::Vector3d &_tbc);
    void Update(const double *pu); 
    void UpdateW(const double *pu); 
    Eigen::Vector2d Project(const Eigen::Vector3d &Xw) const; 
    Eigen::Vector2d Project2d(const Eigen::Vector3d &Xw) const;
    bool isDepthPositive(const Eigen::Vector3d &Xw) const;
 public:
    Eigen::Matrix3d Rwb;
    Eigen::Vector3d twb;
    Eigen::Matrix3d Rcw;
    Eigen::Vector3d tcw;
    Eigen::Matrix3d Rcb, Rbc;
    Eigen::Vector3d tcb, tbc;
    Eigen::Matrix3d Rwb0;
    Eigen::Matrix3d DR;
    int its;
    vihso::Frame* mpF;
};
class VertexIdist: public g2o::BaseVertex<1, double>
{
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexIdist(): BaseVertex<1, double>()
    {}
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}
    virtual void setToOriginImpl() 
    {
        _estimate = 0;
    }
    virtual void oplusImpl(const double* update) 
    {
        _estimate += (*update);
    }
 public:
    Vector3d f_Host;   
    void setfHost(Vector3d f) 
    {
        f_Host = f;
    }
};
class VertexPose : public g2o::BaseVertex<6, ImuCamPose>
{
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexPose(){}
    VertexPose(FramePtr pF)
    {
        setEstimate(ImuCamPose(pF));
    }
    VertexPose(Frame* pF)
    {
        setEstimate(ImuCamPose(pF));
    }
    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;
    virtual void setToOriginImpl() {}
    virtual void oplusImpl(const double* update_)
    {
        _estimate.Update(update_);
        updateCache();
    }
};
class VertexPose4DoF : public g2o::BaseVertex<4,ImuCamPose>
{
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexPose4DoF(){}
    VertexPose4DoF(FramePtr pF)
    {
        setEstimate(ImuCamPose(pF));
    }
    VertexPose4DoF(Eigen::Matrix3d &_Rwc, Eigen::Vector3d &_twc, FramePtr pF)
    {
        setEstimate(ImuCamPose(_Rwc, _twc, pF));
    }
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}
    virtual void setToOriginImpl() {}
    virtual void oplusImpl(const double* update_)
    {
        double update6DoF[6];
        update6DoF[0] = 0;
        update6DoF[1] = 0;
        update6DoF[2] = update_[0];
        update6DoF[3] = update_[1];
        update6DoF[4] = update_[2];
        update6DoF[5] = update_[3];
        _estimate.UpdateW(update6DoF);
        updateCache();
    }
};
class VertexVelocity : public g2o::BaseVertex<3,Eigen::Vector3d>
{
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexVelocity(){}
    VertexVelocity(FramePtr pF);
    VertexVelocity(Frame* pF);
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}
    virtual void setToOriginImpl() {}
    virtual void oplusImpl(const double* update_)
    {
        Eigen::Vector3d uv;
        uv << update_[0], update_[1], update_[2];
        setEstimate(estimate()+uv);
    }
};
class VertexGyroBias : public g2o::BaseVertex<3,Eigen::Vector3d>
{
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexGyroBias(){}
    VertexGyroBias(FramePtr pF);
    VertexGyroBias(Frame* pF);
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}
    virtual void setToOriginImpl() {}
    virtual void oplusImpl(const double* update_)
    {
        Eigen::Vector3d ubg;
        ubg << update_[0], update_[1], update_[2];
        setEstimate(estimate()+ubg);
    }
};
class VertexAccBias : public g2o::BaseVertex<3,Eigen::Vector3d>
{
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexAccBias(){}
    VertexAccBias(FramePtr pF);
    VertexAccBias(Frame* pF);
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}
    virtual void setToOriginImpl() {}
    virtual void oplusImpl(const double* update_)
    {
        Eigen::Vector3d uba;
        uba << update_[0], update_[1], update_[2];
        setEstimate(estimate()+uba);
    }
};
class GDirection
{
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    GDirection(){}
    GDirection(Eigen::Matrix3d pRwg): Rwg(pRwg){}
    void Update(const double *pu)
    {
        Rwg=Rwg*ExpSO3(pu[0],pu[1],0.0);    
    }
    Eigen::Matrix3d Rwg, Rgw;
    int its;
};
class VertexGDir : public g2o::BaseVertex<2,GDirection>
{
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexGDir(){}
    VertexGDir(Eigen::Matrix3d pRwg)
    {
        setEstimate(GDirection(pRwg));
    }
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}
    virtual void setToOriginImpl() {}
    virtual void oplusImpl(const double* update_)
    {
        _estimate.Update(update_);  
        updateCache();
    }
};
class VertexScale : public g2o::BaseVertex<1,double>
{
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexScale()    { setEstimate(1.0); }
    VertexScale(double ps) { setEstimate(ps); }
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}
    virtual void setToOriginImpl(){ setEstimate(1.0); }
    virtual void oplusImpl(const double *update_)
    {
        setEstimate(estimate()*exp(*update_));  
    }
};
class EdgeMono : public g2o::BaseBinaryEdge<2,Eigen::Vector2d,g2o::VertexSBAPointXYZ,VertexPose>
{
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeMono(){}
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}
    void computeError()
    {
        const g2o::VertexSBAPointXYZ* VPoint = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[1]);
        const Eigen::Vector2d obs(_measurement);
        _error = obs - VPose->estimate().Project(VPoint->estimate());
    }
    virtual void linearizeOplus();
    bool isDepthPositive()
    {
        const g2o::VertexSBAPointXYZ* VPoint = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[1]);
        return VPose->estimate().isDepthPositive(VPoint->estimate());
    }
    Eigen::Matrix<double,2,9> GetJacobian()
    {
        linearizeOplus();
        Eigen::Matrix<double,2,9> J;
        J.block<2,3>(0,0) = _jacobianOplusXi;
        J.block<2,6>(0,3) = _jacobianOplusXj;
        return J;
    }
    Eigen::Matrix<double,9,9> GetHessian(){
        linearizeOplus();
        Eigen::Matrix<double,2,9> J;
        J.block<2,3>(0,0) = _jacobianOplusXi;
        J.block<2,6>(0,3) = _jacobianOplusXj;
        return J.transpose()*information()*J;
    }
};
class EdgeMonoOnlyPose : public g2o::BaseUnaryEdge<2,Eigen::Vector2d,VertexPose>
{
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeMonoOnlyPose(const Eigen::Vector3d &Xw_):Xw(Xw_) {}
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}
    void computeError()
    {
        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[0]); 
        const Eigen::Vector2d obs(_measurement);
        _error = obs - VPose->estimate().Project(Xw);
    }
    virtual void linearizeOplus();
    bool isDepthPositive()
    {
        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[0]);
        return VPose->estimate().isDepthPositive(Xw);
    }
    Eigen::Matrix<double,6,6> GetHessian()
    {
        linearizeOplus();
        return _jacobianOplusXi.transpose()*information()*_jacobianOplusXi;
    }
 public:
    const Eigen::Vector3d Xw;
};
class EdgeInertial : public g2o::BaseMultiEdge<9,Vector9d>
{
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeInertial(IMU::Preintegrated* pInt);
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}
    void computeError();
    virtual void linearizeOplus();
    Eigen::Matrix<double,24,24> GetHessian()
    {
        linearizeOplus();
        Eigen::Matrix<double,9,24> J;
        J.block<9,6>(0,0) = _jacobianOplus[0];
        J.block<9,3>(0,6) = _jacobianOplus[1];
        J.block<9,3>(0,9) = _jacobianOplus[2];
        J.block<9,3>(0,12) = _jacobianOplus[3];
        J.block<9,6>(0,15) = _jacobianOplus[4];
        J.block<9,3>(0,21) = _jacobianOplus[5];
        return J.transpose()*information()*J;
    }
    Eigen::Matrix<double,18,18> GetHessianNoPose1()
    {
        linearizeOplus();
        Eigen::Matrix<double,9,18> J;
        J.block<9,3>(0,0) = _jacobianOplus[1];
        J.block<9,3>(0,3) = _jacobianOplus[2];
        J.block<9,3>(0,6) = _jacobianOplus[3];
        J.block<9,6>(0,9) = _jacobianOplus[4];
        J.block<9,3>(0,15) = _jacobianOplus[5];
        return J.transpose()*information()*J;
    }
    Eigen::Matrix<double,9,9> GetHessian2()
    {
        linearizeOplus();
        Eigen::Matrix<double,9,9> J;
        J.block<9,6>(0,0) = _jacobianOplus[4];
        J.block<9,3>(0,6) = _jacobianOplus[5];
        return J.transpose()*information()*J;
    }
    const Eigen::Matrix3d JRg, JVg, JPg;
    const Eigen::Matrix3d JVa, JPa;
    IMU::Preintegrated* mpInt;
    const double dt;
    Eigen::Vector3d g;
};
class EdgeInertialGS : public g2o::BaseMultiEdge<9,Vector9d>
{
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeInertialGS(IMU::Preintegrated* pInt);
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}
    void computeError();
    virtual void linearizeOplus();
    const Eigen::Matrix3d JRg, JVg, JPg;    
    const Eigen::Matrix3d JVa, JPa;         
    IMU::Preintegrated* mpInt;  
    const double dt;
    Eigen::Vector3d gI; 
    Eigen::Vector3d g;  
    Eigen::Matrix<double,27,27> GetHessian()
    {
        linearizeOplus();
        Eigen::Matrix<double,9,27> J;
        J.block<9,6>(0,0) = _jacobianOplus[0];
        J.block<9,3>(0,6) = _jacobianOplus[1];
        J.block<9,3>(0,9) = _jacobianOplus[2];
        J.block<9,3>(0,12) = _jacobianOplus[3];
        J.block<9,6>(0,15) = _jacobianOplus[4];
        J.block<9,3>(0,21) = _jacobianOplus[5];
        J.block<9,2>(0,24) = _jacobianOplus[6];
        J.block<9,1>(0,26) = _jacobianOplus[7];
        return J.transpose()*information()*J;
    }
    Eigen::Matrix<double,27,27> GetHessian2()
    {
        linearizeOplus();
        Eigen::Matrix<double,9,27> J;
        J.block<9,3>(0,0) = _jacobianOplus[2];
        J.block<9,3>(0,3) = _jacobianOplus[3];
        J.block<9,2>(0,6) = _jacobianOplus[6];
        J.block<9,1>(0,8) = _jacobianOplus[7];
        J.block<9,3>(0,9) = _jacobianOplus[1];
        J.block<9,3>(0,12) = _jacobianOplus[5];
        J.block<9,6>(0,15) = _jacobianOplus[0];
        J.block<9,6>(0,21) = _jacobianOplus[4];
        return J.transpose()*information()*J;
    }
    Eigen::Matrix<double,9,9> GetHessian3()
    {
        linearizeOplus();
        Eigen::Matrix<double,9,9> J;
        J.block<9,3>(0,0) = _jacobianOplus[2];
        J.block<9,3>(0,3) = _jacobianOplus[3];
        J.block<9,2>(0,6) = _jacobianOplus[6];
        J.block<9,1>(0,8) = _jacobianOplus[7];
        return J.transpose()*information()*J;
    }
    Eigen::Matrix<double,1,1> GetHessianScale()
    {
        linearizeOplus();
        Eigen::Matrix<double,9,1> J = _jacobianOplus[7];
        return J.transpose()*information()*J;
    }
    Eigen::Matrix<double,3,3> GetHessianBiasGyro()
    {
        linearizeOplus();
        Eigen::Matrix<double,9,3> J = _jacobianOplus[2];
        return J.transpose()*information()*J;
    }
    Eigen::Matrix<double,3,3> GetHessianBiasAcc()
    {
        linearizeOplus();
        Eigen::Matrix<double,9,3> J = _jacobianOplus[3];
        return J.transpose()*information()*J;
    }
    Eigen::Matrix<double,2,2> GetHessianGDir()
    {
        linearizeOplus();
        Eigen::Matrix<double,9,2> J = _jacobianOplus[6];
        return J.transpose()*information()*J;
    }
};
class EdgeGyroRW : public g2o::BaseBinaryEdge<3,Eigen::Vector3d,VertexGyroBias,VertexGyroBias>
{
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeGyroRW(){}
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}
    void computeError()
    {
        const VertexGyroBias* VG1= static_cast<const VertexGyroBias*>(_vertices[0]);
        const VertexGyroBias* VG2= static_cast<const VertexGyroBias*>(_vertices[1]);
        _error = VG2->estimate()-VG1->estimate();
    }
    virtual void linearizeOplus()
    {
        _jacobianOplusXi = -Eigen::Matrix3d::Identity();
        _jacobianOplusXj.setIdentity();
    }
    Eigen::Matrix<double,6,6> GetHessian()
    {
        linearizeOplus();
        Eigen::Matrix<double,3,6> J;
        J.block<3,3>(0,0) = _jacobianOplusXi;
        J.block<3,3>(0,3) = _jacobianOplusXj;
        return J.transpose()*information()*J;
    }
    Eigen::Matrix3d GetHessian2()
    {
        linearizeOplus();
        return _jacobianOplusXj.transpose()*information()*_jacobianOplusXj;
    }
};
class EdgeAccRW : public g2o::BaseBinaryEdge<3,Eigen::Vector3d,VertexAccBias,VertexAccBias>
{
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeAccRW(){}
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}
    void computeError()
    {
        const VertexAccBias* VA1= static_cast<const VertexAccBias*>(_vertices[0]);
        const VertexAccBias* VA2= static_cast<const VertexAccBias*>(_vertices[1]);
        _error = VA2->estimate()-VA1->estimate();
    }
    virtual void linearizeOplus()
    {
        _jacobianOplusXi = -Eigen::Matrix3d::Identity();
        _jacobianOplusXj.setIdentity();
    }
    Eigen::Matrix<double,6,6> GetHessian()
    {
        linearizeOplus();
        Eigen::Matrix<double,3,6> J;
        J.block<3,3>(0,0) = _jacobianOplusXi;
        J.block<3,3>(0,3) = _jacobianOplusXj;
        return J.transpose()*information()*J;
    }
    Eigen::Matrix3d GetHessian2()
    {
        linearizeOplus();
        return _jacobianOplusXj.transpose()*information()*_jacobianOplusXj;
    }
};
class ConstraintPoseImu
{
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ConstraintPoseImu(const Eigen::Matrix3d &Rwb_, const Eigen::Vector3d &twb_, const Eigen::Vector3d &vwb_,
                       const Eigen::Vector3d &bg_, const Eigen::Vector3d &ba_, const Matrix15d &H_):
                       Rwb(Rwb_), twb(twb_), vwb(vwb_), bg(bg_), ba(ba_), H(H_)
    {
        H = (H+H)/2;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,15,15> > es(H);
        Eigen::Matrix<double,15,1> eigs = es.eigenvalues();
        for(int i=0;i<15;i++)
            if(eigs[i]<1e-12)
                eigs[i]=0;
        H = es.eigenvectors()*eigs.asDiagonal()*es.eigenvectors().transpose();
    }
    ConstraintPoseImu(const Eigen::Matrix3d &Rwb_, const Eigen::Vector3d &twb_, const Eigen::Vector3d &vwb_,
                       const IMU::Bias &b, const Matrix15d &H_):
                       Rwb(Rwb_), twb(twb_), vwb(vwb_),
                       bg(Eigen::Vector3d(b.bwx, b.bwy, b.bwz)), 
                       ba(Eigen::Vector3d(b.bax, b.bay, b.baz)), H(H_)
    {
        H = (H+H)/2;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,15,15> > es(H);
        Eigen::Matrix<double,15,1> eigs = es.eigenvalues();
        for(int i=0;i<15;i++)
            if(eigs[i]<1e-12)
                eigs[i]=0;
        H = es.eigenvectors()*eigs.asDiagonal()*es.eigenvectors().transpose();
    }
    Eigen::Matrix3d Rwb;
    Eigen::Vector3d twb;
    Eigen::Vector3d vwb;
    Eigen::Vector3d bg;
    Eigen::Vector3d ba;
    Matrix15d H;        
};
class EdgePriorPoseImu : public g2o::BaseMultiEdge<15,Vector15d>
{
 public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        EdgePriorPoseImu(ConstraintPoseImu* c);
        virtual bool read(std::istream& is){return false;}
        virtual bool write(std::ostream& os) const{return false;}
        void computeError();
        virtual void linearizeOplus();
        Eigen::Matrix<double,15,15> GetHessian(){
            linearizeOplus();
            Eigen::Matrix<double,15,15> J;
            J.block<15,6>(0,0) = _jacobianOplus[0];
            J.block<15,3>(0,6) = _jacobianOplus[1];
            J.block<15,3>(0,9) = _jacobianOplus[2];
            J.block<15,3>(0,12) = _jacobianOplus[3];
            return J.transpose()*information()*J;
        }
        Eigen::Matrix<double,9,9> GetHessianNoPose(){
            linearizeOplus();
            Eigen::Matrix<double,15,9> J;
            J.block<15,3>(0,0) = _jacobianOplus[1];
            J.block<15,3>(0,3) = _jacobianOplus[2];
            J.block<15,3>(0,6) = _jacobianOplus[3];
            return J.transpose()*information()*J;
        }
        Eigen::Matrix3d Rwb;
        Eigen::Vector3d twb, vwb;
        Eigen::Vector3d bg, ba;
};
class EdgePriorAcc : public g2o::BaseUnaryEdge<3,Eigen::Vector3d,VertexAccBias>
{
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgePriorAcc(const Eigen::Vector3d &bprior_):bprior(bprior_) {}
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}
    void computeError()
    {
        const VertexAccBias* VA = static_cast<const VertexAccBias*>(_vertices[0]);
        _error = bprior - VA->estimate();
    }
    virtual void linearizeOplus();
    Eigen::Matrix<double,3,3> GetHessian()
    {
        linearizeOplus();   
        return _jacobianOplusXi.transpose()*information()*_jacobianOplusXi;
    }
    const Eigen::Vector3d bprior;   
};
class EdgePriorGyro : public g2o::BaseUnaryEdge<3,Eigen::Vector3d,VertexGyroBias>
{
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgePriorGyro(const Eigen::Vector3d &bprior_):bprior(bprior_){}
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}
    void computeError()
    {
        const VertexGyroBias* VG = static_cast<const VertexGyroBias*>(_vertices[0]);
        _error = bprior - VG->estimate();
    }
    virtual void linearizeOplus();
    Eigen::Matrix<double,3,3> GetHessian()
    {
        linearizeOplus();
        return _jacobianOplusXi.transpose()*information()*_jacobianOplusXi;
    }
    const Eigen::Vector3d bprior;   
};
class Edge4DoF : public g2o::BaseBinaryEdge<6,Vector6d,VertexPose4DoF,VertexPose4DoF>
{
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Edge4DoF(const Eigen::Matrix4d &deltaT){
        dTij = deltaT;
        dRij = deltaT.block<3,3>(0,0);
        dtij = deltaT.block<3,1>(0,3);
    }
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}
    void computeError()
    {
        const VertexPose4DoF* VPi = static_cast<const VertexPose4DoF*>(_vertices[0]);
        const VertexPose4DoF* VPj = static_cast<const VertexPose4DoF*>(_vertices[1]);
        _error << LogSO3(VPi->estimate().Rcw*VPj->estimate().Rcw.transpose()*dRij.transpose()),
                 VPi->estimate().Rcw*(-VPj->estimate().Rcw.transpose()*VPj->estimate().tcw)+VPi->estimate().tcw - dtij;
    }
    Eigen::Matrix4d dTij;
    Eigen::Matrix3d dRij;
    Eigen::Vector3d dtij;
};
class EdgeMonoOnlyPoseCornor : public g2o::BaseUnaryEdge<2,Eigen::Vector2d,VertexPose>
{
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeMonoOnlyPoseCornor(const Eigen::Vector3d &Xw_):Xw(Xw_),bOutlier(false)
    {}
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}
    void computeError()
    {
        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[0]); 
        const Eigen::Vector2d obs(_measurement);
        _error = obs - VPose->estimate().Project(Xw);   
    }
    virtual void linearizeOplus();
    bool isDepthPositive()
    {
        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[0]);
        return VPose->estimate().isDepthPositive(Xw);
    }
    Eigen::Matrix<double,6,6> GetHessian()
    {
        linearizeOplus();
        return _jacobianOplusXi.transpose()*information()*_jacobianOplusXi;
    }
    void Setfxfy(const float & fxfy_, const Vector2d &vfxfy_)
    {
        fxfy = fxfy_;
        vfxfy = vfxfy_;
    }
	void SetFeature(Feature* ft)
	{
		pFeature = ft;
	}
 public:
    const Eigen::Vector3d Xw;
    float fxfy;
    Vector2d vfxfy;
    bool bOutlier;  
	Feature* pFeature;
};
class EdgeMonoOnlyPoseEdgeLet : public g2o::BaseUnaryEdge<1,double,VertexPose>
{
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeMonoOnlyPoseEdgeLet(const Eigen::Vector3d &Xw_):Xw(Xw_),bOutlier(false) 
    {}
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}
    void computeError()
    {
        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[0]); 
        const double obs(_measurement);
        _error(0,0) = obs - grad.transpose() * VPose->estimate().Project(Xw); 
    }
    virtual void linearizeOplus();
    bool isDepthPositive()
    {
        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[0]);
        return VPose->estimate().isDepthPositive(Xw);
    }
    Eigen::Matrix<double,6,6> GetHessian()
    {
        linearizeOplus();
        return _jacobianOplusXi.transpose()*information()*_jacobianOplusXi;
    }
    void Setfxfy(const float & fxfy_, const Vector2d &vfxfy_)
    {
        fxfy = fxfy_;
        vfxfy = vfxfy_;
    }
    void setGrad(Eigen::Vector2d& n)
    {
        grad = n;
    }
	void SetFeature(Feature* ft)
	{
		pFeature = ft;
	}
 public:
    const Eigen::Vector3d Xw;
    float fxfy;
    Vector2d vfxfy;
    Eigen::Vector2d grad;
    bool bOutlier;  
	Feature* pFeature;
};
class EdgeIHTCorner : public g2o::BaseMultiEdge<2, Vector2d>
{
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeIHTCorner() {}
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}
    void computeError();
    virtual void linearizeOplus();
 public:
    float fxfy;
    Vector2d vfxfy;
    Vector2d cam_project(const Vector3d & trans_xyz)
    {
        return hso::project2d(trans_xyz);
    }
    void Setfxfy(const float & fxfy_, const Vector2d &vfxfy_)
    {
        fxfy = fxfy_;
        vfxfy = vfxfy_;
    }
};
class EdgeIHTEdgeLet : public g2o::BaseMultiEdge<1, double>
{
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeIHTEdgeLet() {}
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}
    void computeError();
    virtual void linearizeOplus();
 public:
    Vector2d _normal;   
    float fxfy;
    Vector2d vfxfy;
    Vector2d cam_project(const Vector3d & trans_xyz)
    {
        return hso::project2d(trans_xyz);
    }
    void setTargetNormal(Vector2d n) 
    {
        _normal = n;
    }
    void Setfxfy(const float & fxfy_, const Vector2d &vfxfy_)
    {
        fxfy = fxfy_;
        vfxfy = vfxfy_;
    }
};
class VertexPhotometricPose : public g2o::BaseVertex<6, g2o::SE3Quat> 	
{
 public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	VertexPhotometricPose()
	{}
	virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}
	virtual void setToOriginImpl()
	{
  		_estimate = g2o::SE3Quat();
	}
	virtual void oplusImpl(const double* update_)
	{
		Eigen::Map<const Vector6d> update(update_);
		setEstimate(g2o::SE3Quat::exp(update)*estimate());
	}
};
class VertexExposureRatio : public g2o::BaseVertex<1, double> 
{
 public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	VertexExposureRatio(){ setEstimate(1.0); }
	VertexExposureRatio(double ps) { setEstimate(ps); }
	virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}
	virtual void setToOriginImpl() { setEstimate(1.0); }
	virtual void oplusImpl(const double* update_)
	{
        setEstimate(estimate()*exp(*update_));  
    }
};
class EdgePhotometric : public g2o::BaseBinaryEdge<1, double, VertexPhotometricPose, VertexExposureRatio> 
{
 public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	EdgePhotometric(){}
	virtual void computeError();
	virtual void linearizeOplus();
	virtual bool read(std::istream& in) { return false;}
	virtual bool write(std::ostream& out) const {return false; }
 public:
	void setConfig(	const Vector3d& Xref_, const cv::Mat& curImg_, int stride_, int border_,
    				int level_, float scale_, double fxl_, double fyl_, float setting_huberTH_, float cutoff_error_, float max_energy_,
					FramePtr pCurFrame_, bool binverse_, int patternRow_, int patternCol_)
	{
		Xref = Xref_;
		curImg = curImg_;
		stride = stride_;
		border = border_;
		level = level_;
		scale = scale_;
		fxl = fxl_;
		fyl = fyl_;
		setting_huberTH = setting_huberTH_;
		cutoff_error = cutoff_error_;
		max_energy = max_energy_;
		pCurFrame = pCurFrame_;
		binverse = binverse_;
		patternRow = patternRow_;
		patternCol = patternCol_;
	}
	bool isDepthPositive()
    {
        const VertexPhotometricPose* VT21 = static_cast<const VertexPhotometricPose*>(_vertices[0]);
		Xcur = VT21->estimate() * Xref;
        return (Xcur[2]<0.0);
    }
	bool computeConfig()
	{
		Vector2f uv_cur_0(pCurFrame->cam_->world2cam(Xcur).cast<float>());
		Vector2f uv_cur_pyr(uv_cur_0 * scale);
		float u_cur = uv_cur_pyr[0];
		float v_cur = uv_cur_pyr[1];
		u_cur_i = floorf(u_cur);
		v_cur_i = floorf(v_cur);
		if(u_cur_i-border < 0 || v_cur_i-border < 0 || u_cur_i+border >= curImg.cols || v_cur_i+border >= curImg.rows)	
			return false;
		float subpix_u_cur = u_cur-u_cur_i;
		float subpix_v_cur = v_cur-v_cur_i;
		w_cur_tl = (1.0-subpix_u_cur) * (1.0-subpix_v_cur);
		w_cur_tr = subpix_u_cur * (1.0-subpix_v_cur);
		w_cur_bl = (1.0-subpix_u_cur) * subpix_v_cur;
		w_cur_br = subpix_u_cur * subpix_v_cur;
		if(!binverse)
			Frame::jacobian_xyz2uv(Xcur, frame_jac);	
		uint8_t* cur_img_ptr = (uint8_t*)curImg.data + (v_cur_i + patternRow)*stride + u_cur_i + patternCol;
		cur_color = w_cur_tl*cur_img_ptr[0] + w_cur_tr*cur_img_ptr[1] + w_cur_bl*cur_img_ptr[stride] + w_cur_br*cur_img_ptr[stride+1];
		if(!std::isfinite(cur_color))
			return false;
		return true;
	}
	void setJacobian_inv(const Vector6d& J_inv_)
	{
		J_inv = J_inv_;
	}
	Vector3d Xref;		
	Vector3d Xcur;		
	cv::Mat curImg;		
	int stride, border;	
	int level;			
    float scale;		
    double fxl,fyl;		
    float setting_huberTH;	
	float cutoff_error;
    float max_energy;		
	FramePtr pCurFrame;
	bool binverse;
	Eigen::Matrix<double,2,6> frame_jac;			
	Vector6d J_inv;									
	int u_cur_i, v_cur_i;							
	float w_cur_tl, w_cur_tr, w_cur_bl, w_cur_br;	
	int patternRow, patternCol;						
	float cur_color;								
	float b=0.0;
};
} 
#endif 
