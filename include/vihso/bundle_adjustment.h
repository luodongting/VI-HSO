#ifndef VIHSO_BUNDLE_ADJUSTMENT_H_
#define VIHSO_BUNDLE_ADJUSTMENT_H_

#include <vihso/global.h>

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_multi_edge.h>
#include <g2o/types/types_six_dof_expmap.h>
#include <g2o/core/robust_kernel.h>

#include "vihso/vikit/math_utils.h"
#include "G2oTypes.h"

using namespace g2o;

namespace g2o {
class EdgeSE3ProjectXYZ;
class SparseOptimizer;
class VertexSE3Expmap;
class VertexSBAPointXYZ;
}

namespace vihso {

typedef g2o::EdgeSE3ProjectXYZ g2oEdgeSE3;  
typedef g2o::VertexSE3Expmap g2oFrameSE3;   
typedef g2o::VertexSBAPointXYZ g2oPoint;    

class Frame;
class Point;
class Feature;
class Map;
class rdvoEdgeProjectXYZ2UV;
class EdgeProjectID2UV;
class EdgeProjectID2UVEdgeLet;

namespace ba 
{

struct EdgeContainerSE3
{
  g2oEdgeSE3*     edge;
  Frame*          frame;
  Feature*        feature;
  bool            is_deleted;
  EdgeContainerSE3(g2oEdgeSE3* e, Frame* frame, Feature* feature) :
    edge(e), frame(frame), feature(feature), is_deleted(false)
  {}
};

struct EdgeContainerEdgelet
{
  rdvoEdgeProjectXYZ2UV*  edge;
  Frame*                  frame;
  Feature*                feature;
  bool                    is_deleted;
  EdgeContainerEdgelet(rdvoEdgeProjectXYZ2UV* e, Frame* frame, Feature* feature) :
    edge(e), frame(frame), feature(feature), is_deleted(false)
  {}
};

struct EdgeContainerID
{
    EdgeProjectID2UV* edge;
    Frame* frame;
    Feature* feature;
    bool is_deleted;

    EdgeContainerID(EdgeProjectID2UV* e, Frame* frame, Feature* feature) :
    edge(e), frame(frame), feature(feature), is_deleted(false) {}
};
struct EdgeContainerIDEdgeLet
{
    EdgeProjectID2UVEdgeLet* edge;
    Frame* frame;
    Feature* feature;
    bool is_deleted;

    EdgeContainerIDEdgeLet(EdgeProjectID2UVEdgeLet* e, Frame* frame, Feature* feature):
    edge(e), frame(frame), feature(feature), is_deleted(false) {}
};

struct EdgeFrameFeature
{
    EdgeIHTCorner* edge;
    Frame* frame;
    Feature* feature;
    bool is_deleted;

    EdgeFrameFeature(EdgeIHTCorner* e, Frame* frame, Feature* feature) :
    edge(e), frame(frame), feature(feature), is_deleted(false) {}
};
struct EdgeLetFrameFeature
{
    EdgeIHTEdgeLet* edge;
    Frame* frame;
    Feature* feature;
    bool is_deleted;

    EdgeLetFrameFeature(EdgeIHTEdgeLet* e, Frame* frame, Feature* feature):
    edge(e), frame(frame), feature(feature), is_deleted(false) {}
};

void setupG2o(g2o::SparseOptimizer * optimizer);

void runSparseBAOptimizer(
    g2o::SparseOptimizer* optimizer,
    unsigned int num_iter,
    double& init_error,
    double& final_error);

g2oFrameSE3* createG2oFrameSE3(
    Frame* kf,
    size_t id,
    bool fixed);

g2oPoint* createG2oPoint(
    Vector3d pos,
    size_t id,
    bool fixed);

g2oEdgeSE3* createG2oEdgeSE3(
    g2oFrameSE3* v_kf,
    g2oPoint* v_mp,
    const Vector2d& f_up,
    bool robust_kernel,
    double huber_width,
    double weight = 1);

rdvoEdgeProjectXYZ2UV* createG2oEdgeletSE3(
    g2oFrameSE3* v_kf,
    g2oPoint* v_mp,
    const Vector2d& f_up,
    bool robust_kernel,
    double huber_width,
    double weight = 1,
    const Vector2d& grad = Vector2d(0,0));


void VisiualOnlyLocalBA(Frame* center_kf, set<Frame*>* core_kfs, Map* map, size_t& n_incorrect_edges_1, size_t& n_incorrect_edges_2,
                        double& init_error, double& final_error, bool bVERBOSE=false);

void InertialOptimization(Map *pMap, Eigen::Matrix3d &Rwg, double &scale, Eigen::Vector3d &bg,Eigen::Vector3d &ba, 
                        bool bFixedVel=false, bool bGauss=false,
                        float priorG = 1e2, float priorA = 1e6);
void InertialOptimization(Map *pMap, Eigen::Matrix3d &Rwg, double &scale);
void ScaleDelayOptimization(const std::vector<FramePtr> vpKFs, Eigen::Matrix3d &Rwg, double &scale);

void FullInertialBA(Map *pMap, int its, const bool bFixLocal, bool *pbStopFlag, bool bInit, double priorG = 1e2, double priorA=1e6);

int PoseInertialOptimizationLastKeyFrame(FramePtr pFrame, bool bVERBOSE = false);
int PoseInertialOptimizationLastFrame(FramePtr pFrame, bool bVERBOSE = false);
void visualImuLocalBundleAdjustment(Frame* center_kf, set<Frame*>* core_kfs, Map* map, bool bVERBOSE=false);


void ComputeHuberthreshold(FramePtr pFrame ,float &huber_corner, float &huber_edge);
void ComputeHuberthreshold(std::vector<FramePtr> vpFrames ,float &huber_corner, float &huber_edge);
void ComputeHuberthreshold(std::list<Point*> lpMPs, FramePtr pFrame, float &huber_corner, float &huber_edge);
void ComputeHuberthreshold(std::set<Point*> spMPs, Frame* pFrame, float &huber_corner, float &huber_edge);
Eigen::MatrixXd Marginalize(const Eigen::MatrixXd &H, const int &start, const int &end);

const double kf_cor = 1.0;
const double kf_edge = 1.0;
const double f_cor = 1.0;
const double f_edge = 1.0;
const double local_cor = 1.0;
const double local_edge = 1.0;


} // namespace ba

class VertexSBAPointID : public BaseVertex<1, double>
{
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexSBAPointID() : BaseVertex<1, double>()
    {}

    virtual bool read(std::istream& is) { return true; }
    virtual bool write(std::ostream& os) const { return true; } 

    virtual void setToOriginImpl() 
    {
        _estimate = 0;
    }

    virtual void oplusImpl(const double* update) 
    {
        _estimate += (*update);
    }
};

class EdgeProjectID2UV : public BaseMultiEdge<2, Vector2d>
{
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeProjectID2UV()
    {}

    virtual bool read(std::istream& is) {return true;}

    virtual bool write(std::ostream& os) const {return true;}

    void computeError()
    {
        const VertexSBAPointID* point = static_cast<const VertexSBAPointID*>(_vertices[0]);
        const VertexSE3Expmap* host   = static_cast<const VertexSE3Expmap*>(_vertices[1]); 
        const VertexSE3Expmap* target = static_cast<const VertexSE3Expmap*>(_vertices[2]); 

        SE3 Ttw = SE3(target->estimate().rotation(), target->estimate().translation());
        SE3 Thw = SE3(host->estimate().rotation(), host->estimate().translation());
        SE3 Tth = Ttw*Thw.inverse();    

        Vector2d obs(_measurement);
        _error = obs - cam_project( Tth * (_fH*(1.0/point->estimate())) );  
        
    }

    virtual void linearizeOplus()
    {
        VertexSBAPointID* vp = static_cast<VertexSBAPointID*>(_vertices[0]);
        double idHost = vp->estimate();

        VertexSE3Expmap * vh = static_cast<VertexSE3Expmap *>(_vertices[1]);
        SE3 Thw(vh->estimate().rotation(), vh->estimate().translation());
        VertexSE3Expmap * vt = static_cast<VertexSE3Expmap *>(_vertices[2]);
        SE3 Ttw(vt->estimate().rotation(), vt->estimate().translation());

        SE3 Tth = Ttw*Thw.inverse();        
        Vector3d t_th = Tth.translation();  
        Matrix3d R_th = Tth.rotation_matrix();  
        Vector3d Rf = R_th*_fH;
        Vector3d pTarget = Tth * (_fH*(1.0/idHost));
        Vector2d proj = hso::project2d(pTarget);

        Vector2d Juvdd;
        Juvdd[0] = -(t_th[0] - proj[0]*t_th[2]) / (Rf[2] + idHost*t_th[2]);
        Juvdd[1] = -(t_th[1] - proj[1]*t_th[2]) / (Rf[2] + idHost*t_th[2]);
        _jacobianOplus[0] = Juvdd;

        Matrix<double,2,6> Jpdxi;
        double x = pTarget[0];
        double y = pTarget[1];
        double z = pTarget[2];
        double z_2 = z*z;

        Jpdxi(0,0) = x*y/z_2;           
        Jpdxi(0,1) = -(1+(x*x/z_2));
        Jpdxi(0,2) = y/z;
        Jpdxi(0,3) = -1./z;
        Jpdxi(0,4) = 0;
        Jpdxi(0,5) = x/z_2;

        Jpdxi(1,0) = (1+y*y/z_2);
        Jpdxi(1,1) = -x*y/z_2;
        Jpdxi(1,2) = -x/z;
        Jpdxi(1,3) = 0;
        Jpdxi(1,4) = -1./z;
        Jpdxi(1,5) = y/z_2;

        Matrix<double,6,6> adHost;
        SE3Quat TthG20(Tth.unit_quaternion(),Tth.translation());
        adHost = -TthG20.adj();
        Matrix<double,6,6> adTarget;
        adTarget = Matrix<double,6,6>::Identity();
        _jacobianOplus[1] = Jpdxi*adHost;
        _jacobianOplus[2] = Jpdxi*adTarget;    

    }

    Vector3d _fH;   
    void setHostBearing(Vector3d f) 
    {
        _fH = f;
    }

    Vector2d cam_project(const Vector3d & trans_xyz)
    {
        return hso::project2d(trans_xyz);
    }
};

class EdgeProjectID2UVEdgeLet : public BaseMultiEdge<1, double>
{
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeProjectID2UVEdgeLet()
    {}

    virtual bool read(std::istream& is) {return true;}

    virtual bool write(std::ostream& os) const {return true;}

    void computeError()
    {
        const VertexSBAPointID* point = static_cast<const VertexSBAPointID*>(_vertices[0]);
        const VertexSE3Expmap* host = static_cast<const VertexSE3Expmap*>(_vertices[1]); 
        const VertexSE3Expmap* target = static_cast<const VertexSE3Expmap*>(_vertices[2]); 

        SE3 Ttw = SE3(target->estimate().rotation(), target->estimate().translation());
        SE3 Thw = SE3(host->estimate().rotation(), host->estimate().translation());
        SE3 Tth = Ttw*Thw.inverse();

        double obs = _measurement;
        _error(0,0) = obs - _normal.transpose()*cam_project( Tth * (_fH*(1.0/point->estimate())) );     
    }

    virtual void linearizeOplus()
    {
        VertexSBAPointID* vp = static_cast<VertexSBAPointID*>(_vertices[0]);
        double idHost = vp->estimate();

        VertexSE3Expmap * vh = static_cast<VertexSE3Expmap *>(_vertices[1]);
        SE3 Thw(vh->estimate().rotation(), vh->estimate().translation());
        VertexSE3Expmap * vt = static_cast<VertexSE3Expmap *>(_vertices[2]);
        SE3 Ttw(vt->estimate().rotation(), vt->estimate().translation());

        SE3 Tth = Ttw*Thw.inverse();
        Vector3d t_th = Tth.translation();
        Matrix3d R_th = Tth.rotation_matrix();
        Vector3d Rf = R_th*_fH;
        Vector3d pTarget = Tth * (_fH*(1.0/idHost));
        Vector2d proj = hso::project2d(pTarget);

        Vector2d Juvdd;
        Juvdd[0] = -(t_th[0] - proj[0]*t_th[2]) / (Rf[2] + idHost*t_th[2]);
        Juvdd[1] = -(t_th[1] - proj[1]*t_th[2]) / (Rf[2] + idHost*t_th[2]);
        _jacobianOplus[0] = _normal.transpose()*Juvdd;

        Matrix<double,2,6> Jpdxi;
        double x = pTarget[0];
        double y = pTarget[1];
        double z = pTarget[2];
        double z_2 = z*z;

        Jpdxi(0,0) = x*y/z_2;
        Jpdxi(0,1) = -(1+(x*x/z_2));
        Jpdxi(0,2) = y/z;
        Jpdxi(0,3) = -1./z;
        Jpdxi(0,4) = 0;
        Jpdxi(0,5) = x/z_2;

        Jpdxi(1,0) = (1+y*y/z_2);
        Jpdxi(1,1) = -x*y/z_2;
        Jpdxi(1,2) = -x/z;
        Jpdxi(1,3) = 0;
        Jpdxi(1,4) = -1./z;
        Jpdxi(1,5) = y/z_2;

        Matrix<double,6,6> adHost;
        SE3Quat TthG20(Tth.unit_quaternion(),Tth.translation());
        adHost = -TthG20.adj();
        Matrix<double,6,6> adTarget;
        adTarget = Matrix<double,6,6>::Identity();
        _jacobianOplus[1] = _normal.transpose()*Jpdxi*adHost;
        _jacobianOplus[2] = _normal.transpose()*Jpdxi*adTarget;
        
    }

    Vector3d _fH;
    void setHostBearing(Vector3d f) 
    {
        _fH = f;
    }

    Vector2d _normal;
    void setTargetNormal(Vector2d n) 
    {
        _normal = n;
    }
    Vector2d cam_project(const Vector3d & trans_xyz)
    {
        return hso::project2d(trans_xyz);
    }
};


class rdvoEdgeProjectXYZ2UV : public BaseBinaryEdge<1, double, VertexSBAPointXYZ, VertexSE3Expmap>
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  rdvoEdgeProjectXYZ2UV() : BaseBinaryEdge<1, double, VertexSBAPointXYZ, VertexSE3Expmap>() {}

  virtual bool read(std::istream& is) {return true;}

  virtual bool write(std::ostream& os) const {return true;}

  void computeError()  {}

  virtual void linearizeOplus() {}

  void setGrad(const Vector2d& g) { _grad = g; }

  Vector2d _grad;

};

} // namespace vihso

#endif // VIHSO_BUNDLE_ADJUSTMENT_H_
