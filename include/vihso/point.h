#ifndef VIHSO_POINT_H_
#define VIHSO_POINT_H_
#include <boost/noncopyable.hpp>
#include <boost/thread.hpp>
#include <vihso/global.h>
#include "vihso/vikit/math_utils.h"

namespace g2o {
class VertexSBAPointXYZ; }
typedef g2o::VertexSBAPointXYZ g2oPoint;

namespace vihso 
{
class Feature;
class VertexSBAPointID;
typedef Matrix<double, 2, 3> Matrix23d;
class Point : boost::noncopyable
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    enum PointType {TYPE_DELETED, TYPE_TEMPORARY, TYPE_CANDIDATE, TYPE_UNKNOWN, TYPE_GOOD};
    enum FeatureType {FEATURE_GRADIENT, FEATURE_EDGELET, FEATURE_CORNER};
    static int point_counter_;  
    int id_;                    
    Vector3d pos_;          
    double idist_;          
    Feature* hostFeature_;  
    list<Feature*> obs_;    
    PointType type_;            
    FeatureType ftr_type_;      
    int last_published_ts_;     
    int last_structure_optim_;  
    int last_projected_kf_id_;  
    int n_failed_reproj_;       
    int n_succeeded_reproj_;    
    int seedStates_;    
    bool isBad_;
    g2oPoint* v_pt_;            
    VertexSBAPointID* vPoint_;  
    size_t nBA_;
    float color_ = 128; 
    Feature* m_last_feature = NULL;
    int m_last_feature_kf_id = -1;
    vector<double> m_rad_est; 
    Vector3d normal_;           
    Matrix3d normal_information_;    
    bool normal_set_;           
public:
    Point(const Vector3d& pos);
    Point(const Vector3d& pos, Feature* ftr);
    ~Point();
    void addFrameRef(Feature* ftr);
    bool deleteFrameRef(Frame* frame);
    Feature* findFrameRef(Frame* frame);
    void initNormal();
    bool getCloseViewObs(const Vector3d& pos, Feature*& obs);
    void optimize(const size_t n_iter); 
    void optimizeLM(const size_t n_iter); 
    void optimizeID(const size_t n_iter);
    inline static void jacobian_xyz2uv(const Vector3d& p_in_f,const Matrix3d& R_f_w, Matrix23d& point_jac) 
    {
        const double z_inv = 1.0/p_in_f[2];
        const double z_inv_sq = z_inv*z_inv;
        point_jac(0, 0) = z_inv;
        point_jac(0, 1) = 0.0;
        point_jac(0, 2) = -p_in_f[0] * z_inv_sq;
        point_jac(1, 0) = 0.0;
        point_jac(1, 1) = z_inv;
        point_jac(1, 2) = -p_in_f[1] * z_inv_sq;
        point_jac = - point_jac * R_f_w;    
    }
    inline static void jacobian_id2uv(const Vector3d& p_in_f, const SE3& Tth, const double idH, const Vector3d& fH, Vector2d& point_jac)
    {
        Vector2d proj = hso::project2d(p_in_f); 
        Vector3d t_th = Tth.translation();      
        Matrix3d R_th = Tth.rotation_matrix();  
        Vector3d Rf = R_th*fH;                  
        point_jac[0] = -(t_th[0] - proj[0]*t_th[2]) / (Rf[2] + t_th[2]*idH);    
        point_jac[1] = -(t_th[1] - proj[1]*t_th[2]) / (Rf[2] + t_th[2]*idH);    
    }
    int nApplyScaled;       
    int nGetAllMapPoints;   
    int nFullBAmaxKFID;     
    int mnBALocalForKF; 
    void SetWorldPos(const Vector3d &pos);
    void SetIdist(const double &idist);
    Vector3d GetWorldPos();
    double GetIdist();
    void UpdatePose();
    void UpdatePoseScale(const double &scale);
    void UpdatePoseidist(const double &idist);
    list<vihso::Feature*> GetObservations();
    size_t GetNumofObs();
    void ClearObservations();
    std::mutex mMutexPos;   
    std::mutex mMutexFeatures;
	static std::mutex mGlobalMutex;	
};
} 
#endif 
