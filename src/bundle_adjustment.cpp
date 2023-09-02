#include <boost/thread.hpp>

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/linear_solver_eigen.h>
#include <g2o/solvers/linear_solver_dense.h>
#include <g2o/types/types_six_dof_expmap.h>

#include <vihso/bundle_adjustment.h>
#include <vihso/frame.h>
#include <vihso/feature.h>
#include <vihso/point.h>
#include <vihso/config.h>
#include <vihso/map.h>
#include <vihso/matcher.h>

#include "vihso/vikit/math_utils.h"
#include "G2oTypes.h"

#define SCHUR_TRICK 1

namespace vihso {
namespace ba {



void setupG2o(g2o::SparseOptimizer * optimizer)
{
  optimizer->setVerbose(false);

    #if SCHUR_TRICK
    g2o::BlockSolver_6_3::LinearSolverType* linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    #else
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverCholmod<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    #endif

  solver->setMaxTrialsAfterFailure(5);
  optimizer->setAlgorithm(solver);

}


void
runSparseBAOptimizer(g2o::SparseOptimizer* optimizer,
                     unsigned int num_iter,
                     double& init_error, double& final_error)
{
    optimizer->initializeOptimization();
    optimizer->computeActiveErrors();
    init_error = optimizer->activeChi2();
    optimizer->optimize(num_iter);
    final_error = optimizer->activeChi2();
}
g2oFrameSE3*
createG2oFrameSE3(Frame* frame, size_t id, bool fixed)
{
    g2oFrameSE3* v = new g2oFrameSE3();
    v->setId(id);
    v->setFixed(fixed);
	Sophus::SE3 Tcw = frame->GetPoseSE3();
    v->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion(), Tcw.translation()));
    return v;
}


g2oPoint*
createG2oPoint(Vector3d pos,size_t id,bool fixed)
{
  g2oPoint* v = new g2oPoint();
  v->setId(id);

    #if SCHUR_TRICK
    v->setMarginalized(true);
    #endif

  v->setFixed(fixed);
  v->setEstimate(pos);
  return v;
}

g2oEdgeSE3*
createG2oEdgeSE3( g2oFrameSE3* v_frame,
                  g2oPoint* v_point,
                  const Vector2d& f_up,
                  bool robust_kernel,
                  double huber_width,
                  double weight)
{
  g2oEdgeSE3* e = new g2oEdgeSE3();
  e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_point));
  e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_frame));
  e->setMeasurement(f_up);
  e->setInformation(Eigen::Matrix2d::Identity()*weight);
  g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();   
  rk->setDelta(huber_width);
  e->setRobustKernel(rk);
  e->setParameterId(0, 0);
  return e;
}



rdvoEdgeProjectXYZ2UV* 
createG2oEdgeletSE3(g2oFrameSE3* v_frame,
                    g2oPoint* v_point,
                    const Vector2d& f_up,
                    bool robust_kernel,
                    double huber_width,
                    double weight,
                    const Vector2d& grad)
{
  rdvoEdgeProjectXYZ2UV* e = new rdvoEdgeProjectXYZ2UV();
  e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_point));
  e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_frame));
  e->setMeasurement(grad.transpose() * f_up);
  e->information() = weight * Eigen::Matrix<double,1,1>::Identity(1,1);
  g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();      
  rk->setDelta(huber_width);
  e->setRobustKernel(rk);
  e->setParameterId(0, 0); 
  e->setGrad(grad);
  return e;
}

void VisiualOnlyLocalBA(Frame* center_kf, set<Frame*>* core_kfs, Map* map,
                        size_t& n_incorrect_edges_1, size_t& n_incorrect_edges_2,   
                        double& init_error, double& final_error, 					
						bool bVERBOSE)                    
{
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);

    g2o::BlockSolverX::LinearSolverType* linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();                  
    g2o::BlockSolverX* solver_ptr = new g2o::BlockSolverX(linearSolver);                                
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);  
    solver->setMaxTrialsAfterFailure(5);
	optimizer.setVerbose(bVERBOSE);
    optimizer.setAlgorithm(solver);

    list<EdgeContainerID> edges;
    list<EdgeContainerIDEdgeLet> edgeLets;

    set<Point*> mps;            
    list<Frame*> neib_kfs;      
    list<Frame*> hostKeyFrame;  

    size_t v_id = 0;                    
    size_t n_mps = 0;
    size_t n_fix_kfs = 0;       
    size_t n_var_kfs = 1;       
    size_t n_edges = 0;         
    n_incorrect_edges_1 = 0;
    n_incorrect_edges_2 = 0;

    for(set<Frame*>::iterator it_kf = core_kfs->begin(); it_kf != core_kfs->end(); ++it_kf)
    {
        g2oFrameSE3* v_kf;
        if((*it_kf)->keyFrameId_ == 0 || (*it_kf)->keyFrameId_+20 < center_kf->keyFrameId_) 
            v_kf = createG2oFrameSE3((*it_kf), v_id++, true);
        else
            v_kf = createG2oFrameSE3((*it_kf), v_id++, false); 

        (*it_kf)->v_kf_ = v_kf; 
        ++n_var_kfs;
        assert(optimizer.addVertex(v_kf));

        for(Features::iterator it_pt=(*it_kf)->fts_.begin(); it_pt!=(*it_kf)->fts_.end(); ++it_pt)
        {
            if((*it_pt)->point == NULL) continue;
            assert((*it_pt)->point->type_ != Point::TYPE_CANDIDATE);
            mps.insert((*it_pt)->point);
        }
    }
    
    vector<float> errors_pt, errors_ls, errors_tt;
    int n_pt=0, n_ls=0, n_tt=0;

    for(set<Point*>::iterator it_pt = mps.begin(); it_pt!=mps.end(); ++it_pt)
    {
        Frame* host_frame = (*it_pt)->hostFeature_->frame;
        Vector3d pHost = (*it_pt)->hostFeature_->f * (1.0/(*it_pt)->GetIdist());

        list<Feature*> observations = (*it_pt)->GetObservations();
        for(auto it_ft=observations.begin(); it_ft!=observations.end(); ++it_ft)
        {
            if((*it_ft)->frame->id_ == host_frame->id_) continue;
            assert((*it_ft)->point == *it_pt);

			const Sophus::SE3 Ttw = (*it_ft)->frame->GetPoseSE3();
			const Sophus::SE3 Twh = host_frame->GetPoseInverseSE3();
            SE3 Tth = Ttw * Twh;  
            Vector3d pTarget = Tth * pHost;
            Vector2d e = hso::project2d((*it_ft)->f) - hso::project2d(pTarget);
            e *= 1.0 / (1<<(*it_ft)->level);

            if((*it_ft)->type == Feature::EDGELET)
            {
                errors_ls.push_back(fabs((*it_ft)->grad.transpose()*e));
                n_ls++;
            }
            else
            {
                errors_pt.push_back(e.norm());
                n_pt++;
            }
        }
    }

    float huber_corner,huber_edge;
    if(!errors_pt.empty() && !errors_ls.empty())
    {

        huber_corner = 1.4826*hso::getMedian(errors_pt);
        huber_edge = 1.4826*hso::getMedian(errors_ls);
    }
    else if(errors_pt.empty() && !errors_ls.empty())
    {
        huber_corner = 1.0 / center_kf->cam_->errorMultiplier2();
        huber_edge = 1.4826*hso::getMedian(errors_ls);
    }
    else if(!errors_pt.empty() && errors_ls.empty())
    {
        huber_corner = 1.4826*hso::getMedian(errors_pt);
        huber_edge   = 0.5 / center_kf->cam_->errorMultiplier2();
    }
    else
    {
        assert(false);
    }
    
    for(set<Point*>::iterator it_pt = mps.begin(); it_pt!=mps.end(); ++it_pt)
    {
        VertexSBAPointID* vPoint = new VertexSBAPointID();
        vPoint->setId(v_id++);
        vPoint->setFixed(false);
        vPoint->setEstimate((*it_pt)->GetIdist());
        (*it_pt)->vPoint_ = vPoint;
        (*it_pt)->nBA_++;

        g2oFrameSE3* vHost = NULL;  
        if((*it_pt)->hostFeature_->frame->v_kf_ == NULL)
        {
            g2oFrameSE3* v_kf = createG2oFrameSE3((*it_pt)->hostFeature_->frame, v_id++, true);
            (*it_pt)->hostFeature_->frame->v_kf_ = v_kf;
            ++n_fix_kfs;
            assert(optimizer.addVertex(v_kf));
            hostKeyFrame.push_back((*it_pt)->hostFeature_->frame);
            vHost = v_kf;
        }
        else
            vHost = (*it_pt)->hostFeature_->frame->v_kf_;

        assert(optimizer.addVertex(vPoint));
        ++n_mps;
        list<Feature*> observations = (*it_pt)->GetObservations();
        list<Feature*>::iterator it_obs=observations.begin();
        while(it_obs!=observations.end())
        {
            if((*it_obs)->frame->id_ == (*it_pt)->hostFeature_->frame->id_)
            {
                ++it_obs;
                continue;
            }

            g2oFrameSE3* vTarget = NULL;   
            if((*it_obs)->frame->v_kf_ == NULL) 
            {
                g2oFrameSE3* v_kf = createG2oFrameSE3((*it_obs)->frame, v_id++, true);
                (*it_obs)->frame->v_kf_ = v_kf;
                ++n_fix_kfs;
                assert(optimizer.addVertex(v_kf));
                neib_kfs.push_back((*it_obs)->frame);

                vTarget = v_kf;
            }
            else
                vTarget = (*it_obs)->frame->v_kf_;

            if((*it_obs)->type != Feature::EDGELET) 
            {
                EdgeProjectID2UV* edge = new EdgeProjectID2UV();
                edge->resize(3);
                edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(vPoint));
                edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(vHost));
                edge->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(vTarget));

                edge->setHostBearing((*it_pt)->hostFeature_->f);                
                edge->setMeasurement(hso::project2d((*it_obs)->f));             

                float inv_sigma2 = 1.0/((1<<(*it_obs)->level)*(1<<(*it_obs)->level));
                edge->setInformation(Eigen::Matrix2d::Identity() * inv_sigma2); 

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();    
                rk->setDelta(huber_corner);
                edge->setRobustKernel(rk);
            
                edge->setParameterId(0, 0); 
     
                edges.push_back(EdgeContainerID(edge, (*it_obs)->frame, *it_obs));  
                assert(optimizer.addEdge(edge));
            }
            else 
            {
                EdgeProjectID2UVEdgeLet* edgeLet = new EdgeProjectID2UVEdgeLet();
                edgeLet->resize(3);
                edgeLet->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(vPoint));
                edgeLet->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(vHost));
                edgeLet->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(vTarget));

                edgeLet->setHostBearing((*it_pt)->hostFeature_->f); 
                edgeLet->setTargetNormal((*it_obs)->grad);          
                edgeLet->setMeasurement((*it_obs)->grad.transpose()*hso::project2d((*it_obs)->f));

                float inv_sigma2 = 1.0/((1<<(*it_obs)->level)*(1<<(*it_obs)->level));
                edgeLet->setInformation(Eigen::Matrix<double,1,1>::Identity()*inv_sigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
                rk->setDelta(huber_edge);
                edgeLet->setRobustKernel(rk);
                edgeLet->setParameterId(0, 0); 

                edgeLets.push_back(EdgeContainerIDEdgeLet(edgeLet, (*it_obs)->frame, *it_obs));  
                assert(optimizer.addEdge(edgeLet));
            }

            ++n_edges;
            ++it_obs;
        }
    }

    if(map->size() > 5) 
    {
        if(center_kf->fts_.size() < 100)
            runSparseBAOptimizer(&optimizer, Config::lobaNumIter()+10, init_error, final_error);    
        else
            runSparseBAOptimizer(&optimizer, Config::lobaNumIter(), init_error, final_error);      
    }
    else    
        runSparseBAOptimizer(&optimizer, 50, init_error, final_error);    


    for(set<Frame*>::iterator it = core_kfs->begin(); it != core_kfs->end(); ++it)
    {
		Sophus::SE3 Tcw( (*it)->v_kf_->estimate().rotation(), (*it)->v_kf_->estimate().translation());
        (*it)->SetPose(Tcw);
        (*it)->v_kf_ = NULL;
        map->mCandidatesManager.changeCandidatePosition(*it);
    }

    for(list<Frame*>::iterator it = neib_kfs.begin(); it != neib_kfs.end(); ++it)
        (*it)->v_kf_ = NULL;

    for(list<Frame*>::iterator it = hostKeyFrame.begin(); it != hostKeyFrame.end(); ++it)
        (*it)->v_kf_ = NULL;

    for(set<Point*>::iterator it = mps.begin(); it != mps.end(); ++it)
    {
        (*it)->UpdatePoseidist((*it)->vPoint_->estimate());
        (*it)->vPoint_ = NULL;
    }

    const double reproj_thresh_2 = 2.0 / center_kf->cam_->errorMultiplier2();  
    const double reproj_thresh_1 = 1.2 / center_kf->cam_->errorMultiplier2(); 
    const double reproj_thresh_2_squared = reproj_thresh_2*reproj_thresh_2;
    for(list<EdgeContainerID>::iterator it = edges.begin(); it != edges.end(); ++it)
    {
        if(it->feature->point == NULL) continue;

        if(it->edge->chi2() > reproj_thresh_2_squared)
        {
            if(it->feature->point->type_ == Point::TYPE_TEMPORARY) 
            {
                it->feature->point->isBad_ = true;
                continue;
            }
            map->removePtFrameRef(it->frame, it->feature);
            ++n_incorrect_edges_1;
        }
    }

    const double reproj_thresh_1_squared = reproj_thresh_1*reproj_thresh_1;
    for(list<EdgeContainerIDEdgeLet>::iterator it = edgeLets.begin(); it != edgeLets.end(); ++it)
    {
        if(it->feature->point == NULL) continue;


        if(it->edge->chi2() > reproj_thresh_1_squared)
        {
            if(it->feature->point->type_ == Point::TYPE_TEMPORARY) 
            {
                it->feature->point->isBad_ = true;
                continue;
            }
            map->removePtFrameRef(it->frame, it->feature);
            ++n_incorrect_edges_2;

            continue;
        }
    }

    init_error = sqrt(init_error)*center_kf->cam_->errorMultiplier2();
    final_error = sqrt(final_error)*center_kf->cam_->errorMultiplier2();
    map->IncreaseChangeIndex();
}

void InertialOptimization(  Map *pMap, Eigen::Matrix3d &Rwg, double &scale, Eigen::Vector3d &bg, Eigen::Vector3d &ba,
                            bool bFixedVel, 
                            bool bGauss, float priorG, float priorA)   
{
    int its = 200; 
    int maxKFid = pMap->GetMaxKFid();
    const std::vector<FramePtr> vpAllKFs = pMap->GetAllKeyFrames();
    std::vector<FramePtr> vpKFs;
    for(size_t i=0; i<vpAllKFs.size(); i++)
    {
        vpKFs.push_back(vpAllKFs[i]);
    }
   
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    if (priorG!=0.0)
        solver->setUserLambdaInit(1e3);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);
 
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        FramePtr pKFi = vpKFs[i];
        if(pKFi->keyFrameId_ > maxKFid)
            continue;

        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->keyFrameId_);
        VP->setFixed(true); 
        optimizer.addVertex(VP);
        VertexVelocity* VV = new VertexVelocity(pKFi);
        VV->setId(maxKFid+(pKFi->keyFrameId_)+1);
        if (bFixedVel)
            VV->setFixed(true);
        else
            VV->setFixed(false);
        optimizer.addVertex(VV);
    }

    VertexGyroBias* VG = new VertexGyroBias(vpKFs.front());
    VG->setId(maxKFid*2+2);
    if (bFixedVel)
        VG->setFixed(true);
    else
        VG->setFixed(false);
    optimizer.addVertex(VG);

    VertexAccBias* VA = new VertexAccBias(vpKFs.front());
    VA->setId(maxKFid*2+3);
    if (bFixedVel)
        VA->setFixed(true);
    else
        VA->setFixed(false);
    optimizer.addVertex(VA);

    EdgePriorAcc* epa = new EdgePriorAcc(Eigen::Vector3d::Zero());
    epa->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
    double infoPriorA = priorA;
    epa->setInformation(infoPriorA*Eigen::Matrix3d::Identity());
    optimizer.addEdge(epa);
    EdgePriorGyro* epg = new EdgePriorGyro(Eigen::Vector3d::Zero());
    epg->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
    double infoPriorG = priorG;
    epg->setInformation(infoPriorG*Eigen::Matrix3d::Identity());
    optimizer.addEdge(epg);

    VertexGDir* VGDir = new VertexGDir(Rwg);
    VGDir->setId(maxKFid*2+4);
    VGDir->setFixed(false);
    optimizer.addVertex(VGDir);

    VertexScale* VS = new VertexScale(scale);
    VS->setId(maxKFid*2+5);
    VS->setFixed(false);
    optimizer.addVertex(VS);

    vector<EdgeInertialGS*> vpei;
    vpei.reserve(vpKFs.size());
    vector<pair<FramePtr, FramePtr> > vppUsedKF;
    vppUsedKF.reserve(vpKFs.size());
    

    for(size_t i=0;i<vpKFs.size();i++)
    {
        FramePtr pKFi = vpKFs[i];

        if(pKFi->mpLastKeyFrame && pKFi->keyFrameId_<=maxKFid)
        {
            if(!pKFi->isKeyframe() || pKFi->mpLastKeyFrame->keyFrameId_>maxKFid)
                continue;
            if(!pKFi->mpImuPreintegrated)
                std::cout << "Not preintegrated measurement" << std::endl;

            pKFi->mpImuPreintegrated->SetNewBias(pKFi->mpLastKeyFrame->GetImuBias());
            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mpLastKeyFrame->keyFrameId_);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+(pKFi->mpLastKeyFrame->keyFrameId_)+1);
            g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->keyFrameId_);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+(pKFi->keyFrameId_)+1);
            g2o::HyperGraph::Vertex* VG = optimizer.vertex(maxKFid*2+2);
            g2o::HyperGraph::Vertex* VA = optimizer.vertex(maxKFid*2+3);
            g2o::HyperGraph::Vertex* VGDir = optimizer.vertex(maxKFid*2+4);
            g2o::HyperGraph::Vertex* VS = optimizer.vertex(maxKFid*2+5);
            if(!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS)
            {
                cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG << ", "<< VA << ", " << VP2 << ", " << VV2 <<  ", "<< VGDir << ", "<< VS <<endl;

                continue;
            }

            EdgeInertialGS* ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
            ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
            ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
            ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
            ei->setVertex(6,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGDir));
            ei->setVertex(7,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VS));

            vpei.push_back(ei);

            vppUsedKF.push_back(make_pair(pKFi->mpLastKeyFrame,pKFi));
            optimizer.addEdge(ei);

        }
    }

    optimizer.initializeOptimization();
    optimizer.computeActiveErrors(); 
    double init_error = optimizer.activeChi2();
    optimizer.optimize(its);
    optimizer.computeActiveErrors();
    double end_error = optimizer.activeChi2();

    VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid*2+2));
    VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid*2+3));
    bg << VG->estimate();
    ba << VA->estimate();
    Rwg = VGDir->estimate().Rwg;
    scale = VS->estimate();

    Vector6d vb;
    vb << VG->estimate(), VA->estimate();
    IMU::Bias b (vb[3],vb[4],vb[5],vb[0],vb[1],vb[2]);
    
    const size_t N = vpKFs.size();
    for(size_t i=0; i<N; i++)
    {
        FramePtr pKFi = vpKFs[i];
        if(pKFi->keyFrameId_>maxKFid)
            continue;

        VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+(pKFi->keyFrameId_)+1));
        Eigen::Vector3d Vw = VV->estimate();
        pKFi->SetVelocity(Vw);

        if ((pKFi->GetGyroBias()-bg).norm() > 0.01) 
        {
            pKFi->SetNewBias(b);
            if (pKFi->mpImuPreintegrated)
                pKFi->mpImuPreintegrated->Reintegrate();
        }
        else
            pKFi->SetNewBias(b);
    }
}

void InertialOptimization(Map *pMap, Eigen::Matrix3d &Rwg, double &scale)
{
    int its = 10;
    int maxKFid = pMap->GetMaxKFid();
    const std::vector<FramePtr> vpKFs = pMap->GetAllKeyFrames();

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);    

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        assert(vpKFs[i]);

        FramePtr pKFi = vpKFs[i];
        if(pKFi->keyFrameId_ > maxKFid)
            continue;
        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->keyFrameId_);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        VertexVelocity* VV = new VertexVelocity(pKFi);
        VV->setId(maxKFid+1+(pKFi->keyFrameId_));
        VV->setFixed(true);
        optimizer.addVertex(VV);

        VertexGyroBias* VG = new VertexGyroBias(pKFi);
        VG->setId(2*(maxKFid+1)+(pKFi->keyFrameId_));
        VG->setFixed(true);
        optimizer.addVertex(VG);

        VertexAccBias* VA = new VertexAccBias(pKFi);
        VA->setId(3*(maxKFid+1)+(pKFi->keyFrameId_));
        VA->setFixed(true);
        optimizer.addVertex(VA);
    }

    VertexGDir* VGDir = new VertexGDir(Rwg);
    VGDir->setId(4*(maxKFid+1));
    VGDir->setFixed(false);
    optimizer.addVertex(VGDir);
    VertexScale* VS = new VertexScale(scale);
    VS->setId(4*(maxKFid+1)+1);
    VS->setFixed(false);
    optimizer.addVertex(VS);

    for(size_t i=0;i<vpKFs.size();i++)
    {
        FramePtr pKFi = vpKFs[i];

        if(pKFi->mpLastKeyFrame && pKFi->keyFrameId_<=maxKFid)
        {
            assert(vpKFs[i]);
            assert(vpKFs[i]->mpLastKeyFrame);

            if(!pKFi->isKeyframe() || pKFi->mpLastKeyFrame->keyFrameId_>maxKFid)
                continue;

            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mpLastKeyFrame->keyFrameId_);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex((maxKFid+1)+pKFi->mpLastKeyFrame->keyFrameId_);
            g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->keyFrameId_);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex((maxKFid+1)+pKFi->keyFrameId_);
            g2o::HyperGraph::Vertex* VG = optimizer.vertex(2*(maxKFid+1)+pKFi->mpLastKeyFrame->keyFrameId_);
            g2o::HyperGraph::Vertex* VA = optimizer.vertex(3*(maxKFid+1)+pKFi->mpLastKeyFrame->keyFrameId_);
            g2o::HyperGraph::Vertex* VGDir = optimizer.vertex(4*(maxKFid+1));
            g2o::HyperGraph::Vertex* VS = optimizer.vertex(4*(maxKFid+1)+1);
            if(!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS)
            {
                cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG << ", "<< VA << ", " << VP2 << ", " << VV2 <<  ", "<< VGDir << ", "<< VS <<endl;
                continue;
            }

            EdgeInertialGS* ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
            ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
            ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
            ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
            ei->setVertex(6,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGDir));
            ei->setVertex(7,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VS));

            optimizer.addEdge(ei);
        }
    }

    optimizer.initializeOptimization();
    optimizer.computeActiveErrors(); 
    double init_error = optimizer.activeChi2();
    optimizer.optimize(its);
    optimizer.computeActiveErrors();
    double end_error = optimizer.activeChi2();
    
    scale = VS->estimate();
    Rwg = VGDir->estimate().Rwg;
}

void ScaleDelayOptimization(const std::vector<FramePtr> vpKFs, Eigen::Matrix3d &Rwg, double &scale)
{

    int its = 10;
    int maxKFid = vpKFs.back()->GetKeyFrameID();

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    solver->setUserLambdaInit(1e3);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);
	
	std::vector<VertexVelocity*> vVel;	vVel.reserve(vpKFs.size());
	std::vector<double> vpreVel;		vpreVel.reserve(vpKFs.size());

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        assert(vpKFs[i]);

        FramePtr pKFi = vpKFs[i];
        if(pKFi->keyFrameId_ > maxKFid)
            continue;
        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->keyFrameId_);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        VertexVelocity* VV = new VertexVelocity(pKFi);
        VV->setId(maxKFid+1+(pKFi->keyFrameId_));
        VV->setFixed(false);
        optimizer.addVertex(VV);

		vVel.push_back(VV);
		vpreVel.push_back(pKFi->mVw.norm());
		
        VertexGyroBias* VG = new VertexGyroBias(pKFi);
        VG->setId(2*(maxKFid+1)+(pKFi->keyFrameId_));
        VG->setFixed(true);
        optimizer.addVertex(VG);
        VertexAccBias* VA = new VertexAccBias(pKFi);
        VA->setId(3*(maxKFid+1)+(pKFi->keyFrameId_));
        VA->setFixed(true);
        optimizer.addVertex(VA);
    }

    VertexGDir* VGDir = new VertexGDir(Rwg);
    VGDir->setId(4*(maxKFid+1));
    VGDir->setFixed(false);
    optimizer.addVertex(VGDir);
    VertexScale* VS = new VertexScale(scale);
    VS->setId(4*(maxKFid+1)+1);
    VS->setFixed(false);
    optimizer.addVertex(VS);

    for(size_t i=1;i<vpKFs.size();i++)
    {
        FramePtr pKFi = vpKFs[i];

        if(pKFi->mpLastKeyFrame && pKFi->keyFrameId_<=maxKFid)
        {
            assert(vpKFs[i]);
            assert(vpKFs[i]->mpLastKeyFrame);

            if(!pKFi->isKeyframe() || pKFi->mpLastKeyFrame->keyFrameId_>maxKFid)
                continue;

            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mpLastKeyFrame->keyFrameId_);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex((maxKFid+1)+pKFi->mpLastKeyFrame->keyFrameId_);
            g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->keyFrameId_);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex((maxKFid+1)+pKFi->keyFrameId_);
            g2o::HyperGraph::Vertex* VG = optimizer.vertex(2*(maxKFid+1)+pKFi->mpLastKeyFrame->keyFrameId_);
            g2o::HyperGraph::Vertex* VA = optimizer.vertex(3*(maxKFid+1)+pKFi->mpLastKeyFrame->keyFrameId_);
            g2o::HyperGraph::Vertex* VGDir = optimizer.vertex(4*(maxKFid+1));
            g2o::HyperGraph::Vertex* VS = optimizer.vertex(4*(maxKFid+1)+1);
            if(!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS)
            {
                cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG << ", "<< VA << ", " << VP2 << ", " << VV2 <<  ", "<< VGDir << ", "<< VS <<endl;
                continue;
            }

            EdgeInertialGS* ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
            ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
            ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
            ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
            ei->setVertex(6,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGDir));
            ei->setVertex(7,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VS));

            optimizer.addEdge(ei);
        }
    }

    optimizer.initializeOptimization();
    optimizer.computeActiveErrors(); 
    double init_error = optimizer.activeChi2();
    optimizer.optimize(its);
    optimizer.computeActiveErrors();
    double end_error = optimizer.activeChi2();

    scale = VS->estimate();
    Rwg = VGDir->estimate().Rwg;
}

void FullInertialBA(Map *pMap, int its,    
                    const bool bFixLocal,   
                    bool *pbStopFlag,      
                    bool bInit,            
                    double priorG, double priorA)  
{
    int maxKFid = pMap->GetMaxKFid();
    const vector<FramePtr> vpKFs = pMap->GetAllKeyFrames();
    const vector<vihso::Point*> vpMPs = pMap->GetAllMapPoints();
    const float fxfy = vpKFs[0]->cam_->errorMultiplier2();  
    const Vector2d vfxfy = vpKFs[0]->cam_->focal_length(); 

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    solver->setUserLambdaInit(1e-5);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);
    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    FramePtr pIncKF;
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        FramePtr pKFi = vpKFs[i];
        if(pKFi->keyFrameId_>maxKFid)
            continue;
        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->keyFrameId_);
        pIncKF=pKFi;
        bool bFixed = false;
        if(bFixLocal)
            bFixed = true;
        VP->setFixed(bFixed);
        optimizer.addVertex(VP);

        if(pKFi->bImu)
        {
            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid+3*(pKFi->keyFrameId_)+1);
            VV->setFixed(bFixed);
            optimizer.addVertex(VV);
            if (!bInit)
            {
                VertexGyroBias* VG = new VertexGyroBias(pKFi);
                VG->setId(maxKFid+3*(pKFi->keyFrameId_)+2);
                VG->setFixed(bFixed);
                optimizer.addVertex(VG);
                VertexAccBias* VA = new VertexAccBias(pKFi);
                VA->setId(maxKFid+3*(pKFi->keyFrameId_)+3);
                VA->setFixed(bFixed);
                optimizer.addVertex(VA);
            }
        }
    }

    if (bInit)
    {
        VertexGyroBias* VG = new VertexGyroBias(pIncKF);
        VG->setId(4*maxKFid+2);
        VG->setFixed(false);
        optimizer.addVertex(VG);
        VertexAccBias* VA = new VertexAccBias(pIncKF);
        VA->setId(4*maxKFid+3);
        VA->setFixed(false);
        optimizer.addVertex(VA);
    }


    for(size_t i=0;i<vpKFs.size();i++)
    {
        FramePtr pKFi = vpKFs[i];

        if(!pKFi->mpLastKeyFrame)
        {
            continue;
        }

        if(pKFi->mpLastKeyFrame && pKFi->keyFrameId_<=maxKFid)
        {
            if(pKFi->mpLastKeyFrame->keyFrameId_>maxKFid)
                continue;
            if(pKFi->bImu && pKFi->mpLastKeyFrame->bImu)
            {
                pKFi->mpImuPreintegrated->SetNewBias(pKFi->mpLastKeyFrame->GetImuBias());
                g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mpLastKeyFrame->keyFrameId_);
                g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+3*(pKFi->mpLastKeyFrame->keyFrameId_)+1);

                g2o::HyperGraph::Vertex* VG1;
                g2o::HyperGraph::Vertex* VA1;
                g2o::HyperGraph::Vertex* VG2;
                g2o::HyperGraph::Vertex* VA2;
                if (!bInit)
                {
                    VG1 = optimizer.vertex(maxKFid+3*(pKFi->mpLastKeyFrame->keyFrameId_)+2);
                    VA1 = optimizer.vertex(maxKFid+3*(pKFi->mpLastKeyFrame->keyFrameId_)+3);
                    VG2 = optimizer.vertex(maxKFid+3*(pKFi->keyFrameId_)+2);
                    VA2 = optimizer.vertex(maxKFid+3*(pKFi->keyFrameId_)+3);
                }
                else 
                {
                    VG1 = optimizer.vertex(4*maxKFid+2);
                    VA1 = optimizer.vertex(4*maxKFid+3);
                }

                g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->keyFrameId_);
                g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+3*(pKFi->keyFrameId_)+1);

                if (!bInit)
                {
                    if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
                    {
                        continue;
                    }
                }
                else
                {
                    if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2)
                    {
                        continue;
                    }
                }

                EdgeInertial* ei = new EdgeInertial(pKFi->mpImuPreintegrated);
                ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
                ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
                ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
                ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
                ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
                ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));

                g2o::RobustKernelHuber* rki = new g2o::RobustKernelHuber;
                ei->setRobustKernel(rki);
                rki->setDelta(sqrt(16.92));

                optimizer.addEdge(ei);

                if (!bInit)
                {
                    EdgeGyroRW* egr= new EdgeGyroRW();
                    egr->setVertex(0,VG1);
                    egr->setVertex(1,VG2);

                    Eigen::Matrix3d InfoG = pKFi->mpImuPreintegrated->C.block<3,3>(9,9).inverse();
                    egr->setInformation(InfoG);
                    egr->computeError();
                    optimizer.addEdge(egr);

                    EdgeAccRW* ear = new EdgeAccRW();
                    ear->setVertex(0,VA1);
                    ear->setVertex(1,VA2);

                    Eigen::Matrix3d InfoA = pKFi->mpImuPreintegrated->C.block<3,3>(12,12).inverse();
                    ear->setInformation(InfoA);
                    ear->computeError();
                    optimizer.addEdge(ear);
                }
            }
            else
            {
                cout << pKFi->keyFrameId_ << " or " << pKFi->mpLastKeyFrame->keyFrameId_ << " no imu" << endl;
            }
        }
    }

    if (bInit)
    {
        g2o::HyperGraph::Vertex* VG = optimizer.vertex(4*maxKFid+2);
        g2o::HyperGraph::Vertex* VA = optimizer.vertex(4*maxKFid+3);

        EdgePriorAcc* epa = new EdgePriorAcc(Eigen::Vector3d::Zero());
        epa->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
        double infoPriorA = priorA;
        epa->setInformation(infoPriorA*Eigen::Matrix3d::Identity());
        optimizer.addEdge(epa);

        EdgePriorGyro* epg = new EdgePriorGyro(Eigen::Vector3d::Zero());
        epg->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
        double infoPriorG = priorG;
        epg->setInformation(infoPriorG*Eigen::Matrix3d::Identity());
        optimizer.addEdge(epg);
    }
   
    float huber_corner=sqrt(5.991), huber_edge=sqrt(5.991);
    ComputeHuberthreshold(vpKFs ,huber_corner, huber_edge);
    huber_corner *= fxfy;   huber_edge *= fxfy;
    
    const unsigned long iniMPid = maxKFid*5;    
    for(size_t i=0; i<vpMPs.size(); i++)
    {
        vihso::Point* pMP = vpMPs[i];
        VertexIdist* VIdist = new VertexIdist();
        unsigned long VPointID = pMP->id_+iniMPid+1;
        VIdist->setId(VPointID);
        VIdist->setFixed(false);
        VIdist->setfHost(pMP->hostFeature_->f);
        VIdist->setEstimate(pMP->GetIdist());
        optimizer.addVertex(VIdist);
        g2o::HyperGraph::Vertex* VD =  optimizer.vertex(VPointID);

        int HostID = pMP->hostFeature_->frame->keyFrameId_;
        g2o::HyperGraph::Vertex* VH =  optimizer.vertex(HostID);
        list<Feature*> observations = pMP->GetObservations();
        list<Feature*>::iterator it_obs = observations.begin();
        while(it_obs != observations.end())
        {
            int TargetID = (*it_obs)->frame->keyFrameId_;
            if(TargetID == HostID)
            {
                ++it_obs;
                continue;
            }

            g2o::HyperGraph::Vertex* VT =  optimizer.vertex(TargetID);
            if(!VD || !VH || !VT)
            {
                cout << "Error" << VD << ", "<< VH << ", "<< VT <<endl;
                continue;
            }

            if((*it_obs)->type != Feature::EDGELET)
            {
                EdgeIHTCorner* eCor = new EdgeIHTCorner();
                eCor->resize(3);
                eCor->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VD));
                eCor->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VH));
                eCor->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VT));

                eCor->Setfxfy(fxfy, vfxfy);
                eCor->setMeasurement((*it_obs)->px);
                double inv_sigma2 = 1.0/((1<<(*it_obs)->level)*(1<<(*it_obs)->level));
                eCor->setInformation(Eigen::Matrix2d::Identity() * inv_sigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
                rk->setDelta(huber_corner);
                eCor->setRobustKernel(rk);
            
                eCor->setParameterId(0, 0);
                optimizer.addEdge(eCor);
            }
            else
            {
                EdgeIHTEdgeLet* eEdge = new EdgeIHTEdgeLet();
                eEdge->resize(3);
                eEdge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VD));
                eEdge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VH));
                eEdge->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VT));

                eEdge->setTargetNormal((*it_obs)->grad); 
                eEdge->Setfxfy(fxfy, vfxfy);
                eEdge->setMeasurement((*it_obs)->grad.transpose()*(*it_obs)->px);
                double inv_sigma2 = 1.0/((1<<(*it_obs)->level)*(1<<(*it_obs)->level));
                eEdge->setInformation(Eigen::Matrix<double,1,1>::Identity()*inv_sigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
                rk->setDelta(huber_edge);
                eEdge->setRobustKernel(rk);
                
                eEdge->setParameterId(0, 0); 
                optimizer.addEdge(eEdge);
            }
            ++it_obs;
        }
    }
    if(pbStopFlag)
        if(*pbStopFlag)
            return;

    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    double init_error = optimizer.activeChi2();
    optimizer.optimize(its);
    optimizer.computeActiveErrors();
    double end_error = optimizer.activeChi2();
   
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        FramePtr pKFi = vpKFs[i];
        if(pKFi->keyFrameId_>maxKFid)
            continue;

        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->keyFrameId_));
        Sophus::SE3 Tcw(VP->estimate().Rcw, VP->estimate().tcw);
        pKFi->SetPose(Tcw);

        if(pKFi->bImu)
        {
            VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+3*(pKFi->keyFrameId_)+1));
            pKFi->SetVelocity(VV->estimate());
           
            VertexGyroBias* VG;
            VertexAccBias* VA;
            if (!bInit)
            {
                VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid+3*(pKFi->keyFrameId_)+2));
                VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid+3*(pKFi->keyFrameId_)+3));
            }
            else
            {
                VG = static_cast<VertexGyroBias*>(optimizer.vertex(4*maxKFid+2));
                VA = static_cast<VertexAccBias*>(optimizer.vertex(4*maxKFid+3));
            }

            Vector6d vb;
            vb << VG->estimate(), VA->estimate();
            IMU::Bias b (vb[3],vb[4],vb[5],vb[0],vb[1],vb[2]);
            pKFi->SetNewBias(b);
            
        }
    }

    for(size_t i=0; i<vpMPs.size(); i++)
    {
        Point* pMP = vpMPs[i];
        VertexIdist* VD = static_cast<VertexIdist*>(optimizer.vertex(pMP->id_+iniMPid+1));
        pMP->UpdatePoseidist(VD->estimate());
    }
    pMap->IncreaseChangeIndex();
}

int PoseInertialOptimizationLastKeyFrame(FramePtr pFrame, bool bVERBOSE)
{
    const float fxfy = pFrame->cam_->errorMultiplier2();  
    const Vector2d vfxfy = pFrame->cam_->focal_length(); 
    
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(bVERBOSE);

    VertexPose* VP = new VertexPose(pFrame);
    VP->setId(0);
    VP->setFixed(false);
    optimizer.addVertex(VP);
    VertexVelocity* VV = new VertexVelocity(pFrame);
    VV->setId(1);
    VV->setFixed(false);
    optimizer.addVertex(VV);
    VertexGyroBias* VG = new VertexGyroBias(pFrame);
    VG->setId(2);
    VG->setFixed(false);
    optimizer.addVertex(VG);
    VertexAccBias* VA = new VertexAccBias(pFrame);
    VA->setId(3);
    VA->setFixed(false);
    optimizer.addVertex(VA);

    FramePtr pFp = pFrame->mpLastKeyFrame;

    VertexPose* VPk = new VertexPose(pFp);
    VPk->setId(4);
    VPk->setFixed(true);
    optimizer.addVertex(VPk);
    VertexVelocity* VVk = new VertexVelocity(pFp);
    VVk->setId(5);
    VVk->setFixed(true);
    optimizer.addVertex(VVk);
    VertexGyroBias* VGk = new VertexGyroBias(pFp);
    VGk->setId(6);
    VGk->setFixed(true);
    optimizer.addVertex(VGk);
    VertexAccBias* VAk = new VertexAccBias(pFp);
    VAk->setId(7);
    VAk->setFixed(true);
    optimizer.addVertex(VAk);
    double curv=pFrame->mVw.norm();  

    EdgeInertial* ei = new EdgeInertial(pFrame->mpImuPreintegrated);
    ei->setVertex(0, VPk);
    ei->setVertex(1, VVk);
    ei->setVertex(2, VGk);
    ei->setVertex(3, VAk);
    ei->setVertex(4, VP);
    ei->setVertex(5, VV);
    optimizer.addEdge(ei);

    EdgeGyroRW* egr = new EdgeGyroRW();
    egr->setVertex(0,VGk);
    egr->setVertex(1,VG);
    Eigen::Matrix3d InfoG = pFrame->mpImuPreintegrated->C.block<3,3>(9,9).inverse();
    egr->setInformation(InfoG);
    optimizer.addEdge(egr);

    EdgeAccRW* ear = new EdgeAccRW();
    ear->setVertex(0,VAk);
    ear->setVertex(1,VA);
    Eigen::Matrix3d InfoA = pFrame->mpImuPreintegrated->C.block<3,3>(12,12).inverse();
    ear->setInformation(InfoA);
    optimizer.addEdge(ear);

    const size_t N = pFrame->nObs(); 
    vector<EdgeMonoOnlyPoseCornor*> edgeCors;   edgeCors.reserve(N);
    vector<EdgeMonoOnlyPoseEdgeLet*> edgeLets;  edgeLets.reserve(N);
    int nCorners=0, nEdgelets=0, nEdgeNum=0;

    float huber_corner=sqrt(5.991), huber_edge=sqrt(5.991);
     double vision_chi2 = 0;
    {
		unique_lock<mutex> lock(Point::mGlobalMutex);

		list<Feature*> lfts = pFrame->GetFeatures();
        for(std::list<Feature*>::iterator itf=lfts.begin(), itend=lfts.end(); itf!=itend; itf++)
        {
            Feature* pFeat = *itf;
            if((*itf)->point == NULL) continue;

            Point* pMP = (*itf)->point;

            if(pFeat->type != Feature::EDGELET) 
            {
                nCorners++;
                pMP->UpdatePose();

                EdgeMonoOnlyPoseCornor* e = new EdgeMonoOnlyPoseCornor(pMP->GetWorldPos());
                e->setVertex(0,VP);
                e->Setfxfy(fxfy, vfxfy);
                e->setMeasurement(pFeat->px);
                e->setInformation(Eigen::Matrix2d::Identity());

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();       
                rk->setDelta(huber_corner);
                e->setRobustKernel(rk);

                optimizer.addEdge(e);
                edgeCors.push_back(e);
                nEdgeNum++;
                e->computeError();
                vision_chi2 += e->chi2();
            }
            else   
            {
                nEdgelets++;
                pMP->UpdatePose();

                EdgeMonoOnlyPoseEdgeLet* e = new EdgeMonoOnlyPoseEdgeLet(pMP->GetWorldPos());
                e->setVertex(0,VP);

                e->Setfxfy(fxfy, vfxfy);
                Eigen::Vector2d Grad = pFeat->grad;
                e->setGrad(Grad);
                e->setMeasurement(Grad.transpose() * pFeat->px);
                e->setInformation(Eigen::Matrix<double,1,1>::Identity());

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();         
                rk->setDelta(huber_edge);
                e->setRobustKernel(rk);

                optimizer.addEdge(e);
                edgeLets.push_back(e);
                nEdgeNum++;

                e->computeError();
                vision_chi2 += e->chi2();
            }   
        }
    }

    const double chiCornor = 2.0;   
    const double chiEdgelet = 1.2;  
    const double chi2Cornor = chiCornor*chiCornor;
    const double chi2Edgelet = chiEdgelet*chiEdgelet;

    int nBad=0;
    int nInliers=0; 
    for(size_t it=0; it<1; it++)
    {
        optimizer.initializeOptimization(0);
        optimizer.computeActiveErrors(); 
        double init_error = optimizer.activeChi2();
        optimizer.optimize(10);
        optimizer.computeActiveErrors();
        double end_error = optimizer.activeChi2();
        nBad=0;
        nInliers=0;
        
        for(vector<EdgeMonoOnlyPoseCornor*>::iterator itCor = edgeCors.begin(); itCor != edgeCors.end(); ++itCor)
        {
            EdgeMonoOnlyPoseCornor* eCor = *itCor;
            if(eCor->bOutlier)
            {
                eCor->computeError();
            }
            if( (eCor->chi2() > chi2Cornor) || (!eCor->isDepthPositive()) )
            {
                eCor->bOutlier = true;
                eCor->setLevel(1);
                nBad++;
            }
            else
            {
                eCor->bOutlier = false;
                eCor->setLevel(0);
                nInliers++;
            }
            if (it==2)
                eCor->setRobustKernel(0);
        }

        for(vector<EdgeMonoOnlyPoseEdgeLet*>::iterator itLet = edgeLets.begin(); itLet != edgeLets.end(); ++itLet)
        {
            EdgeMonoOnlyPoseEdgeLet* eLet = *itLet;
            if(eLet->bOutlier)
            {
                eLet->computeError();
            }
            if( (eLet->chi2() > chi2Edgelet) || (!eLet->isDepthPositive()) )
            {
                eLet->bOutlier = true;
                eLet->setLevel(1);
                nBad++;
            }
            else
            {
                eLet->bOutlier = false;
                eLet->setLevel(0);
                nInliers++;
            }

            if (it==2)
                eLet->setRobustKernel(0);
        }
    
        if(optimizer.edges().size()<10)
        {
            cout << "PIOLF: NOT ENOUGH EDGES" << endl;
            break;
        }

    }

    pFrame->SetImuPoseVelocity(VP->estimate().Rwb, VP->estimate().twb, VV->estimate());
    Vector6d b;
    b << VG->estimate(), VA->estimate();
    pFrame->mImuBias = IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]);
    Eigen::Matrix<double,15,15> H;
    H.setZero();
    H.block<9,9>(0,0)+= ei->GetHessian2(); 
    H.block<3,3>(9,9) += egr->GetHessian2();
    H.block<3,3>(12,12) += ear->GetHessian2();

    int tot_in = 0, tot_out = 0;
    for(vector<EdgeMonoOnlyPoseCornor*>::iterator itCor = edgeCors.begin(); itCor != edgeCors.end(); ++itCor)
    {
        EdgeMonoOnlyPoseCornor* eCor = *itCor;

        if(!eCor->bOutlier)
        {
            H.block<6,6>(0,0) += eCor->GetHessian();
            tot_in++;
        }
        else
        {
            tot_out++;
        }
            
    }
    for(vector<EdgeMonoOnlyPoseEdgeLet*>::iterator itLet = edgeLets.begin(); itLet != edgeLets.end(); ++itLet)
    {
        EdgeMonoOnlyPoseEdgeLet* eLet = *itLet;

        if(!eLet->bOutlier)
        {
            H.block<6,6>(0,0) += eLet->GetHessian();
            tot_in++;
        }
        else
        {
            tot_out++;
        }
    } 

    pFrame->mpcpi = new ConstraintPoseImu(VP->estimate().Rwb,VP->estimate().twb,VV->estimate(),VG->estimate(),VA->estimate(),H);
    delete pFp->mpcpi;
    pFp->mpcpi = NULL;

	int n_incorrect_edges_1 = 0, n_incorrect_edges_2 = 0;
    const double reproj_thresh_2_squared = 1.0*1.0;
    const double reproj_thresh_1_squared = 0.5*0.5;
    for(Features::iterator it=pFrame->fts_.begin(); it!=pFrame->fts_.end(); ++it)
    {
        if((*it)->point == NULL) continue;
        Feature* ft = *it;

        Frame* host = ft->point->hostFeature_->frame;
        Vector3d pHost = ft->point->hostFeature_->f * (1.0/ft->point->GetIdist());
		const Sophus::SE3 Ttw = pFrame->GetPoseSE3();
		const Sophus::SE3 Twh = host->GetPoseInverseSE3();
        SE3 Tth = Ttw * Twh;
        Vector3d pTarget = Tth * pHost;
        if(ft->type != Feature::EDGELET)
        {
            Eigen::Vector2d e = hso::project2d(ft->f) -  hso::project2d(pTarget);
            double chi2 = e[0]*e[0] + e[1]*e[1];
            if(chi2 >= reproj_thresh_2_squared)
            {
				++n_incorrect_edges_1;
				ft->imuBAOutlier = true;
			}    
        }
        else
        {
            double e = ft->grad.transpose() * ( hso::project2d(ft->f) -  hso::project2d(pTarget) );
            double chi2 = e*e;
            if(chi2 >= reproj_thresh_1_squared)
			{
				++n_incorrect_edges_2;
				ft->imuBAOutlier = true;
			}   
        }

    }
    size_t nsuccessful = size_t( nEdgeNum - n_incorrect_edges_1 - n_incorrect_edges_2);

    return nEdgeNum-nBad;
}


int PoseInertialOptimizationLastFrame(  FramePtr pFrame, bool bVERBOSE)
{
    const float fxfy = pFrame->cam_->errorMultiplier2(); 
    const Vector2d vfxfy = pFrame->cam_->focal_length(); 

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    solver->setUserLambdaInit(1e-1);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(bVERBOSE);

    VertexPose* VP = new VertexPose(pFrame);
    VP->setId(0);
    VP->setFixed(false);
    optimizer.addVertex(VP);
    VertexVelocity* VV = new VertexVelocity(pFrame);
    VV->setId(1);
    VV->setFixed(false);
    optimizer.addVertex(VV);
    VertexGyroBias* VG = new VertexGyroBias(pFrame);
    VG->setId(2);
    VG->setFixed(false);
    optimizer.addVertex(VG);
    VertexAccBias* VA = new VertexAccBias(pFrame);
    VA->setId(3);
    VA->setFixed(false);
    optimizer.addVertex(VA);

    FramePtr pFlast = pFrame->m_last_frame;
    VertexPose* VPk = new VertexPose(pFlast);
    VPk->setId(4);
    VPk->setFixed(false);
    optimizer.addVertex(VPk);
    VertexVelocity* VVk = new VertexVelocity(pFlast);
    VVk->setId(5);
    VVk->setFixed(false);
    optimizer.addVertex(VVk);
    VertexGyroBias* VGk = new VertexGyroBias(pFlast);
    VGk->setId(6);
    VGk->setFixed(false);
    optimizer.addVertex(VGk);
    VertexAccBias* VAk = new VertexAccBias(pFlast);
    VAk->setId(7);
    VAk->setFixed(false);
    optimizer.addVertex(VAk);
    double curv=pFrame->mVw.norm();  

    EdgeInertial* ei = new EdgeInertial(pFrame->mpImuPreintegratedFrame);
    ei->setVertex(0, VPk);
    ei->setVertex(1, VVk);
    ei->setVertex(2, VGk);
    ei->setVertex(3, VAk);
    ei->setVertex(4, VP);
    ei->setVertex(5, VV);
    optimizer.addEdge(ei);

    EdgeGyroRW* egr = new EdgeGyroRW();
    egr->setVertex(0,VGk);
    egr->setVertex(1,VG);
    Eigen::Matrix3d InfoG = pFrame->mpImuPreintegratedFrame->C.block<3,3>(9,9).inverse();
    egr->setInformation(InfoG);
    optimizer.addEdge(egr);

    EdgeAccRW* ear = new EdgeAccRW();
    ear->setVertex(0,VAk);
    ear->setVertex(1,VA);
    Eigen::Matrix3d InfoA = pFrame->mpImuPreintegratedFrame->C.block<3,3>(12,12).inverse();
    ear->setInformation(InfoA);
    optimizer.addEdge(ear);

    if (!pFlast->mpcpi)
        cout << "pFp->mpcpi does not exist!!!\nPrevious FrameID = " << pFlast->id_ << endl;

    EdgePriorPoseImu* ep = new EdgePriorPoseImu(pFlast->mpcpi);
    ep->setVertex(0,VPk);
    ep->setVertex(1,VVk);
    ep->setVertex(2,VGk);
    ep->setVertex(3,VAk);
    g2o::RobustKernelHuber* rkp = new g2o::RobustKernelHuber;
    ep->setRobustKernel(rkp);
    rkp->setDelta(5);
    optimizer.addEdge(ep);

    const size_t N = pFrame->nObs(); 
    vector<EdgeMonoOnlyPoseCornor*> edgeCors;   edgeCors.reserve(N);
    vector<EdgeMonoOnlyPoseEdgeLet*> edgeLets;  edgeLets.reserve(N);
    int nCorners=0, nEdgelets=0, nEdgeNum=0;

    float huber_corner=sqrt(5.991), huber_edge=sqrt(5.991);
    double vision_chi2 = 0;
    {
		list<Feature*> lfts = pFrame->GetFeatures();
        for(std::list<Feature*>::iterator itf=lfts.begin(), itend=lfts.end(); itf!=itend; itf++)
        {
            Feature* pFeat = *itf;
            if((*itf)->point == NULL) continue;

            Point* pMP = (*itf)->point;

            if(pFeat->type != Feature::EDGELET) 
            {
                nCorners++;
                pMP->UpdatePose();

                EdgeMonoOnlyPoseCornor* e = new EdgeMonoOnlyPoseCornor(pMP->GetWorldPos());
                e->setVertex(0,VP);

                e->Setfxfy(fxfy, vfxfy);
                e->setMeasurement(pFeat->px);
				e->SetFeature(pFeat);
                e->setInformation(Eigen::Matrix2d::Identity());

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();      
                rk->setDelta(huber_corner);
                e->setRobustKernel(rk);

                optimizer.addEdge(e);
                edgeCors.push_back(e);
                nEdgeNum++;

                e->computeError();
                vision_chi2 += e->chi2();
            }
            else    
            {
                nEdgelets++;
                pMP->UpdatePose();

                EdgeMonoOnlyPoseEdgeLet* e = new EdgeMonoOnlyPoseEdgeLet(pMP->GetWorldPos());
                e->setVertex(0,VP);

                e->Setfxfy(fxfy, vfxfy);
                Eigen::Vector2d Grad = pFeat->grad;
                e->setGrad(Grad);
                e->setMeasurement(Grad.transpose() * pFeat->px);
				e->SetFeature(pFeat);
                e->setInformation(Eigen::Matrix<double,1,1>::Identity());

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();          
                rk->setDelta(huber_edge);
                e->setRobustKernel(rk);

                optimizer.addEdge(e);
                edgeLets.push_back(e);
                nEdgeNum++;

                e->computeError();
                vision_chi2 += e->chi2();
            }   
        }
    }
    const double chiCornor = 2.0;    
    const double chiEdgelet = 1.2;   
    double chi2Cornor = chiCornor*chiCornor;
    double chi2Edgelet = chiEdgelet*chiEdgelet;

    int nBad=0;
    int nInliers=0;        
    for(size_t it=0; it<1; it++)
    {
        optimizer.initializeOptimization(0);
        optimizer.computeActiveErrors();
        double init_error = optimizer.activeChi2();
        optimizer.optimize(10);
        optimizer.computeActiveErrors();
        double end_error = optimizer.activeChi2();
        nBad=0;
        nInliers=0;
        
        for(vector<EdgeMonoOnlyPoseCornor*>::iterator itCor = edgeCors.begin(); itCor != edgeCors.end(); ++itCor)
        {
            EdgeMonoOnlyPoseCornor* eCor = *itCor;

            if(eCor->bOutlier)
            {
                eCor->computeError();
            }

            if( (eCor->chi2() > chi2Cornor) || (!eCor->isDepthPositive()) )
            {
                eCor->bOutlier = true;
				eCor->pFeature->imuBAOutlier = true;
                eCor->setLevel(1);
                nBad++;
            }
            else
            {
                eCor->bOutlier = false;
				eCor->pFeature->imuBAOutlier = false;
                eCor->setLevel(0);
                nInliers++;
            }

            if (it==2)
                eCor->setRobustKernel(0);
        }

        for(vector<EdgeMonoOnlyPoseEdgeLet*>::iterator itLet = edgeLets.begin(); itLet != edgeLets.end(); ++itLet)
        {
            EdgeMonoOnlyPoseEdgeLet* eLet = *itLet;

            if(eLet->bOutlier)
            {
                eLet->computeError();
            }

            if( (eLet->chi2() > chi2Edgelet) || (!eLet->isDepthPositive()) )
            {
                eLet->bOutlier = true;
				eLet->pFeature->imuBAOutlier = true;
                eLet->setLevel(1);
                nBad++;
            }
            else
            {
                eLet->bOutlier = false;
				eLet->pFeature->imuBAOutlier = false;
                eLet->setLevel(0);
                nInliers++;
            }

            if (it==2)
                eLet->setRobustKernel(0);
        }
    
        if(optimizer.edges().size()<10)
        {
            cout << "PIOLF: NOT ENOUGH EDGES" << endl;
            break;
        }
    }

    pFrame->SetImuPoseVelocity(VP->estimate().Rwb, VP->estimate().twb, VV->estimate());
    Vector6d b;
    b << VG->estimate(), VA->estimate();
    pFrame->mImuBias = IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]);

    Eigen::Matrix<double,30,30> H;
    H.setZero();
    H.block<24,24>(0,0)+= ei->GetHessian(); 
    Eigen::Matrix<double,6,6> Hgr = egr->GetHessian(); 
    H.block<3,3>(9,9) += Hgr.block<3,3>(0,0);
    H.block<3,3>(9,24) += Hgr.block<3,3>(0,3);
    H.block<3,3>(24,9) += Hgr.block<3,3>(3,0);
    H.block<3,3>(24,24) += Hgr.block<3,3>(3,3);
    Eigen::Matrix<double,6,6> Har = ear->GetHessian();  
    H.block<3,3>(12,12) += Har.block<3,3>(0,0);
    H.block<3,3>(12,27) += Har.block<3,3>(0,3);
    H.block<3,3>(27,12) += Har.block<3,3>(3,0);
    H.block<3,3>(27,27) += Har.block<3,3>(3,3);
    H.block<15,15>(0,0) += ep->GetHessian();    

    int tot_in = 0, tot_out = 0;
    for(vector<EdgeMonoOnlyPoseCornor*>::iterator itCor = edgeCors.begin(); itCor != edgeCors.end(); ++itCor)
    {
        EdgeMonoOnlyPoseCornor* eCor = *itCor;

        if(!eCor->bOutlier)
        {
            H.block<6,6>(15,15) += eCor->GetHessian();
            tot_in++;
        }
        else
        {
            tot_out++;
        }
            
    }
    for(vector<EdgeMonoOnlyPoseEdgeLet*>::iterator itLet = edgeLets.begin(); itLet != edgeLets.end(); ++itLet)
    {
        EdgeMonoOnlyPoseEdgeLet* eLet = *itLet;

        if(!eLet->bOutlier)
        {
            H.block<6,6>(15,15) += eLet->GetHessian();
            tot_in++;
        }
        else
        {
            tot_out++;
        }
    } 
    H = Marginalize(H,0,14);
    pFrame->mpcpi = new ConstraintPoseImu(VP->estimate().Rwb,VP->estimate().twb,VV->estimate(),VG->estimate(),VA->estimate(),H.block<15,15>(15,15));
    delete pFlast->mpcpi;
    pFlast->mpcpi = NULL;

	int n_incorrect_edges_1 = 0, n_incorrect_edges_2 = 0;
    const double reproj_thresh_2_squared = 1.0*1.0;
    const double reproj_thresh_1_squared = 0.5*0.5;
    for(Features::iterator it=pFrame->fts_.begin(); it!=pFrame->fts_.end(); ++it)
    {
        if((*it)->point == NULL) continue;
        Feature* ft = *it;

        Frame* host = ft->point->hostFeature_->frame;
        Vector3d pHost = ft->point->hostFeature_->f * (1.0/ft->point->GetIdist());
		const Sophus::SE3 Ttw = pFrame->GetPoseSE3();
		const Sophus::SE3 Twh = host->GetPoseInverseSE3();
        SE3 Tth = Ttw * Twh;
        Vector3d pTarget = Tth * pHost;
        if(ft->type != Feature::EDGELET)
        {
            Eigen::Vector2d e = hso::project2d(ft->f) -  hso::project2d(pTarget);
            double chi2 = e[0]*e[0] + e[1]*e[1];
            if(chi2 >= reproj_thresh_2_squared)
            {
				++n_incorrect_edges_1;
				ft->imuBAOutlier = true;
			}    
        }
        else
        {
            double e = ft->grad.transpose() * ( hso::project2d(ft->f) -  hso::project2d(pTarget) );
            double chi2 = e*e;
            if(chi2 >= reproj_thresh_1_squared)
			{
				++n_incorrect_edges_2;
				ft->imuBAOutlier = true;
			}   
        }

    }
    size_t nsuccessful = size_t( nEdgeNum - n_incorrect_edges_1 - n_incorrect_edges_2);

    return nEdgeNum-nBad;
}

void visualImuLocalBundleAdjustment(Frame* center_kf, set<Frame*>* core_kfs, Map* map, bool bVERBOSE)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-2);
    optimizer.setAlgorithm(solver);

    const int n_opt_kfs = 25;
    int n_features = 0;
    int n_features_threshold = 1500;
    std::vector<Frame*> act_kf_vec;
    Frame* fixed_kf = NULL;
    bool is_in_act_vec = true;
    int fix_id = -1;

    act_kf_vec.push_back(center_kf);
    act_kf_vec.back()->mnBALocalForKF = center_kf->id_;
    while(act_kf_vec.back()->mpLastKeyFrame)
    {
        act_kf_vec.push_back(act_kf_vec.back()->mpLastKeyFrame.get());
        act_kf_vec.back()->mnBALocalForKF = center_kf->id_;
        
		// limit optimization points
        n_features += act_kf_vec.back()->fts_.size();
        if(n_features >= n_features_threshold)
        {
            fix_id = act_kf_vec.back()->id_;
            break;
        }

        if(act_kf_vec.size() >= n_opt_kfs)
        {
            fix_id = act_kf_vec.back()->id_;
            break;
        }

    }
    std::vector<Frame*> covisual_kf_vec;
    for(auto it=core_kfs->begin();it!=core_kfs->end();++it)
    {
        if((*it)->mnBALocalForKF == center_kf->id_)
            continue;

        if((*it)->id_ < fix_id)
        {
            fix_id = (*it)->id_;
            is_in_act_vec = false;
            fixed_kf = *it;
        }
    }
    if(is_in_act_vec)
    {        
        if(act_kf_vec.back()->mpLastKeyFrame)
        {
            fixed_kf = act_kf_vec.back()->mpLastKeyFrame.get();
        }
        else
        {
            fixed_kf = act_kf_vec.back();
            act_kf_vec.pop_back();
        }
    }
    else
    {  
        for(auto it=core_kfs->begin();it!=core_kfs->end();++it)
        {
            if((*it)->mnBALocalForKF == center_kf->id_)
                continue;

            if((*it)->id_ == fix_id)
                continue;

            (*it)->mnBALocalForKF = center_kf->id_;
            covisual_kf_vec.push_back(*it);
        }
        
    }
    fixed_kf->mnBALocalForKF = center_kf->id_;
    
    
    const int N = act_kf_vec.size() + covisual_kf_vec.size(); 
    const int max_id = center_kf->id_;

    set<Point*> mps;
    auto iter = act_kf_vec.begin();
    while(iter != act_kf_vec.end())
    {
        Frame* kf = *iter;
        if(kf->id_ > max_id)
        {
            ++iter;
            continue;
        }
        bool is_fixed = false;

        VertexPose * VP = new VertexPose(kf);
        VP->setId(kf->id_);
        optimizer.addVertex(VP);
        VP->setFixed(is_fixed);
    
        if(kf->mpImuPreintegrated)
        {
            VertexVelocity* VV = new VertexVelocity(kf);
            VV->setId( max_id + 3*kf->id_ + 1 );
            VV->setFixed(is_fixed);
            optimizer.addVertex(VV);

            VertexGyroBias* VG = new VertexGyroBias(kf);
            VG->setId(max_id + 3*kf->id_ + 2);
            VG->setFixed(is_fixed);
            optimizer.addVertex(VG);

            VertexAccBias* VA = new VertexAccBias(kf);
            VA->setId(max_id + 3*kf->id_ + 3);
            VA->setFixed(is_fixed);
            optimizer.addVertex(VA);
        }

        for(Features::iterator it_pt=kf->fts_.begin(); it_pt!=kf->fts_.end(); ++it_pt)
        {
            if((*it_pt)->point == NULL) continue;
            assert((*it_pt)->point->type_ != Point::TYPE_CANDIDATE);
            mps.insert((*it_pt)->point);
        }
        ++iter;
    }

    VertexPose * VP = new VertexPose(fixed_kf);
    VP->setId(fixed_kf->id_);
    VP->setFixed(true);
    optimizer.addVertex(VP);
    if(fixed_kf->mpImuPreintegrated)
    {
        VertexVelocity* VV = new VertexVelocity(fixed_kf);
        VV->setId( max_id + 3*fixed_kf->id_ + 1 );
        VV->setFixed(true);
        optimizer.addVertex(VV);

        VertexGyroBias* VG = new VertexGyroBias(fixed_kf);
        VG->setId(max_id + 3*fixed_kf->id_ + 2);
        VG->setFixed(true);
        optimizer.addVertex(VG);

        VertexAccBias* VA = new VertexAccBias(fixed_kf);
        VA->setId(max_id + 3*fixed_kf->id_ + 3);
        VA->setFixed(true);
        optimizer.addVertex(VA);
    }
    for(Features::iterator it_pt=fixed_kf->fts_.begin(); it_pt!=fixed_kf->fts_.end(); ++it_pt)
    {
        if((*it_pt)->point == NULL) continue;
        assert((*it_pt)->point->type_ != Point::TYPE_CANDIDATE);
        mps.insert((*it_pt)->point);
    }

    iter = covisual_kf_vec.begin();
    while(iter != covisual_kf_vec.end())
    {
        Frame* kf = *iter;
        if(kf->id_ > max_id)
        {
            ++iter;
            continue;
        }

        VertexPose * VP = new VertexPose(kf);
        VP->setId(kf->id_);
        VP->setFixed(false);
        optimizer.addVertex(VP);

        for(Features::iterator it_pt=kf->fts_.begin(); it_pt!=kf->fts_.end(); ++it_pt)
        {
            if((*it_pt)->point == NULL) continue;

            assert((*it_pt)->point->type_ != Point::TYPE_CANDIDATE);
            mps.insert((*it_pt)->point);
        }

        ++iter;
    }


    float huber_corner = 0, huber_edge = 0;
    double focal_length = center_kf->cam_->errorMultiplier2();
    Eigen::Vector2d fxy = center_kf->cam_->focal_length();
    Eigen::Matrix2d Fxy; Fxy<<fxy[0], 0.0, 0.0, fxy[1];
    ComputeHuberthreshold(mps, center_kf, huber_corner, huber_edge);
    huber_corner *= focal_length;   huber_edge *= focal_length;

    vector<EdgeInertial*> vi_vec(N,(EdgeInertial*)NULL);
    vector<EdgeGyroRW*> eg_vec(N,(EdgeGyroRW*)NULL); 
    vector<EdgeAccRW*> ea_vec(N,(EdgeAccRW*)NULL); 

    for(size_t i=0, iend=act_kf_vec.size()-1;i<iend;i++)
    {
        Frame* kf = act_kf_vec[i];
        if(kf->id_ > max_id)
            continue;

        if(kf->mpLastKeyFrame && kf->mpLastKeyFrame->id_ < max_id)
        {
            if(!kf->mpImuPreintegrated || !kf->mpLastKeyFrame->mpImuPreintegrated)
            {
                std::cout << "Not preintegrated measurement" << std::endl;
                continue;
            }

            kf->mpImuPreintegrated->SetNewBias(kf->mpLastKeyFrame->GetImuBias());  
            
            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(kf->mpLastKeyFrame->id_);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(max_id + 3*kf->mpLastKeyFrame->id_ + 1);
            g2o::HyperGraph::Vertex* VG1 = optimizer.vertex(max_id + 3*kf->mpLastKeyFrame->id_ + 2);
            g2o::HyperGraph::Vertex* VA1 = optimizer.vertex(max_id + 3*kf->mpLastKeyFrame->id_ + 3);
            g2o::HyperGraph::Vertex* VP2 = optimizer.vertex(kf->id_);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(max_id + 3*kf->id_ + 1);
            g2o::HyperGraph::Vertex* VG2 = optimizer.vertex(max_id + 3*kf->id_ + 2);
            g2o::HyperGraph::Vertex* VA2 = optimizer.vertex(max_id + 3*kf->id_ + 3);

            if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
            {
                cerr << "Error " << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<  ", "<< VG2 << ", "<< VA2 <<endl;
                continue;
            }

            EdgeInertial* vi= NULL;
            vi = new EdgeInertial(kf->mpImuPreintegrated);

            vi->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            vi->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            vi->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
            vi->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
            vi->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            vi->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
            optimizer.addEdge(vi);
            vi_vec.push_back(vi);

            EdgeGyroRW* eg = new EdgeGyroRW();
            eg->setVertex(0,VG1);
            eg->setVertex(1,VG2);
            Eigen::Matrix3d InfoG = kf->mpImuPreintegrated->C.block<3,3>(9,9).inverse();
            eg->setInformation(InfoG);
            optimizer.addEdge(eg);
            eg_vec.push_back(eg);

            EdgeAccRW* ea = new EdgeAccRW();
            ea->setVertex(0,VA1);
            ea->setVertex(1,VA2);
            Eigen::Matrix3d InfoA = kf->mpImuPreintegrated->C.block<3,3>(12,12).inverse();
            ea->setInformation(InfoA);
            optimizer.addEdge(ea);
            ea_vec.push_back(ea);

            vi->computeError();
            double ee_vi = vi->chi2();
            eg->computeError();
            double ee_eg = eg->chi2();
            ea->computeError();
            double ee_ea = ea->chi2();
        }

    }

    list<EdgeFrameFeature> edges;
    list<EdgeLetFrameFeature> edgeLets;
    int n_edges = 0;
    int v_id = 4 * (max_id+1);
    for(set<Point*>::iterator it_pt = mps.begin(); it_pt!=mps.end(); ++it_pt)
    {
        VertexIdist* vPoint = new VertexIdist();
        vPoint->setId(v_id++);
        vPoint->setFixed(false);
        vPoint->setfHost((*it_pt)->hostFeature_->f);
        vPoint->setEstimate((*it_pt)->GetIdist());
        assert(optimizer.addVertex(vPoint));
        (*it_pt)->mnBALocalForKF = center_kf->id_;

        if((*it_pt)->hostFeature_->frame->mnBALocalForKF != center_kf->id_)
        {
            (*it_pt)->hostFeature_->frame->mnBALocalForKF = center_kf->id_;
            VertexPose* vHost = new VertexPose((*it_pt)->hostFeature_->frame);
            vHost->setId((*it_pt)->hostFeature_->frame->id_);
            vHost->setFixed(true);
            assert(optimizer.addVertex(vHost));
        }

		double depth_in_hostframe = ((*it_pt)->hostFeature_->frame->GetCameraCenter() - (*it_pt)->GetWorldPos()).norm();
		double point_weight=1.0;
        list<Feature*> observations = (*it_pt)->GetObservations();
        list<Feature*>::iterator it_obs=observations.begin();
        while(it_obs!=observations.end())
        {
            if((*it_obs)->frame->id_ == (*it_pt)->hostFeature_->frame->id_)
            {
                ++it_obs;
                continue;
            }

            if((*it_obs)->frame->mnBALocalForKF != center_kf->id_)
            {
                (*it_obs)->frame->mnBALocalForKF = center_kf->id_;
                VertexPose* vTarget = new VertexPose((*it_obs)->frame);
                vTarget->setId((*it_obs)->frame->id_);
                vTarget->setFixed(true);
                assert(optimizer.addVertex(vTarget));
            }

			const Sophus::SE3 Ttw = (*it_obs)->frame->GetPoseSE3();
			const Sophus::SE3 Twh = (*it_pt)->hostFeature_->frame->GetPoseInverseSE3();
			SE3 Tth = Ttw * Twh;

			double depth_in_targetframe = ((*it_obs)->frame->GetCameraCenter() - (*it_pt)->GetWorldPos()).norm();
			if(depth_in_targetframe<20.0) 			point_weight=1.0;
			else if(depth_in_targetframe<30.0) 		point_weight=0.9;
			else if(depth_in_targetframe<40.0) 		point_weight=0.7;
			else if(depth_in_targetframe<50.0) 		point_weight=0.5;
			else									point_weight=0.1;

            if((*it_obs)->type != Feature::EDGELET)
            {
                EdgeIHTCorner* edge = new EdgeIHTCorner();
                edge->resize(3);


                g2o::HyperGraph::Vertex* VP = optimizer.vertex(v_id-1);
                g2o::HyperGraph::Vertex* VH = optimizer.vertex((*it_pt)->hostFeature_->frame->id_);
                g2o::HyperGraph::Vertex* VT = optimizer.vertex((*it_obs)->frame->id_);

                if(!VP || !VH || !VT)
                {
                    cout<<"frame: VH "<<(*it_pt)->hostFeature_->frame->id_<<" ,VT:"<<(*it_obs)->frame->id_<<endl;
                    cerr << "Error " << VP << ", "<< VH << ", "<< VT<<endl;
                    sleep(10000);
                    continue;
                }

                edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP));
                edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VH));
                edge->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VT));
                edge->Setfxfy(focal_length, fxy);
                edge->setMeasurement((*it_obs)->px);
                float inv_sigma2 = 1.0/((1<<(*it_obs)->level)*(1<<(*it_obs)->level));
                edge->setInformation(Eigen::Matrix2d::Identity()*point_weight*local_cor);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
                rk->setDelta(huber_corner);
                edge->setRobustKernel(rk);
                edge->setParameterId(0, 0);

                edges.push_back(EdgeFrameFeature(edge, (*it_obs)->frame, *it_obs));
                assert(optimizer.addEdge(edge));
            }
            else
            {
                EdgeIHTEdgeLet* edgeLet = new EdgeIHTEdgeLet();
                edgeLet->resize(3);

                g2o::HyperGraph::Vertex* VP = optimizer.vertex(v_id-1);
                g2o::HyperGraph::Vertex* VH = optimizer.vertex((*it_pt)->hostFeature_->frame->id_);
                g2o::HyperGraph::Vertex* VT = optimizer.vertex((*it_obs)->frame->id_);

                if(!VP || !VH || !VT)
                {
                    cout<<"frame: VH "<<(*it_pt)->hostFeature_->frame->id_<<" ,VT:"<<(*it_obs)->frame->id_<<endl;
                    cerr << "Error " << VP << ", "<< VH << ", "<< VT<<endl;
                    sleep(10000);
                    continue;
                }

                edgeLet->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP));
                edgeLet->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VH));
                edgeLet->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VT));
                edgeLet->Setfxfy(focal_length, fxy);
                edgeLet->setTargetNormal((*it_obs)->grad);
                edgeLet->setMeasurement((*it_obs)->grad.transpose()*(*it_obs)->px);
                float inv_sigma2 = 1.0/((1<<(*it_obs)->level)*(1<<(*it_obs)->level));
                edgeLet->setInformation(Eigen::Matrix<double,1,1>::Identity()*point_weight*local_edge);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
                rk->setDelta(huber_edge);
                edgeLet->setRobustKernel(rk);
                edgeLet->setParameterId(0, 0);

                edgeLets.push_back(EdgeLetFrameFeature(edgeLet, (*it_obs)->frame, *it_obs));
                assert(optimizer.addEdge(edgeLet));
            }

            ++n_edges;
            ++it_obs;
        }


    }

    double init_error, final_error=0;
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    init_error = optimizer.activeChi2();
    optimizer.setVerbose(bVERBOSE);
    int n_its = (n_edges>=4000 ? 10 : 15);
    optimizer.optimize(4); 
    final_error = optimizer.activeChi2();

    iter = act_kf_vec.begin();
    while(iter!=act_kf_vec.end())
    {
        Frame* frame = *iter;
        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex( frame->id_ ) );
        VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(  max_id + 3*frame->id_ + 1 ) );
        VertexGyroBias* VG = static_cast<VertexGyroBias*>(optimizer.vertex(  max_id + 3*frame->id_ + 2 ) );
        VertexAccBias* VA = static_cast<VertexAccBias*>(optimizer.vertex(  max_id + 3*frame->id_ + 3 ) );

        Sophus::SE3 Tcw(VP->estimate().Rcw, VP->estimate().tcw);
        frame->SetPose(Tcw);
        frame->SetVelocity(VV->estimate());
        Vector6d b;
        b << VG->estimate(), VA->estimate();
        frame->SetNewBias(IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]));

        map->mCandidatesManager.changeCandidatePosition(*iter);
        ++iter;
    }

    set<Frame*>::iterator sit = core_kfs->begin();
    while(sit!=core_kfs->end())
    {
        VertexPose* VPose = static_cast<VertexPose*>(optimizer.vertex( (*sit)->id_ ) );
        Sophus::SE3 Tcw(VPose->estimate().Rcw, VPose->estimate().tcw);
        (*sit)->SetPose(Tcw);
        map->mCandidatesManager.changeCandidatePosition(*sit);
        ++sit;
       
    }

    v_id = 4 * (max_id+1);
    for(set<Point*>::iterator it = mps.begin(); it != mps.end(); ++it,++v_id)
    {
        VertexIdist* VPoint = static_cast<VertexIdist*>(optimizer.vertex(v_id) );
        (*it)->UpdatePoseidist(VPoint->estimate());
    }

    int n_incorrect_edges_1 = 0, n_incorrect_edges_2 = 0;   
    const double reproj_thresh_2 = 2.0; 
    const double reproj_thresh_1 = 1.2; 

    const double reproj_thresh_2_squared = reproj_thresh_2*reproj_thresh_2;
    for(list<EdgeFrameFeature>::iterator it = edges.begin(); it != edges.end(); ++it)
    {
        if( it->feature->point == NULL) continue;

        if(it->edge->chi2() > reproj_thresh_2_squared)
        {
            if(it->feature->point->type_ == Point::TYPE_TEMPORARY) 
            {
                it->feature->point->isBad_ = true;
                continue;
            }
            map->removePtFrameRef(it->frame, it->feature);
            ++n_incorrect_edges_1;
        }
    }

    const double reproj_thresh_1_squared = reproj_thresh_1*reproj_thresh_1;
    for(list<EdgeLetFrameFeature>::iterator it = edgeLets.begin(); it != edgeLets.end(); ++it)
    {
        if(it->feature->point == NULL) continue;


        if(it->edge->chi2() > reproj_thresh_1_squared)
        {
            if(it->feature->point->type_ == Point::TYPE_TEMPORARY) 
            {
                it->feature->point->isBad_ = true;
                continue;
            }
            map->removePtFrameRef(it->frame, it->feature);
            ++n_incorrect_edges_2;

            continue;
        }
    }
    map->IncreaseChangeIndex(); 
}

void ComputeHuberthreshold(FramePtr pFrame, float &huber_corner, float &huber_edge)
{
    vector<float> errors_pt, errors_ls; 
    for(std::list<Feature*>::iterator itf=pFrame->fts_.begin(), itend=pFrame->fts_.end(); itf!=itend; itf++)
    {
        Point* pMP = (*itf)->point;
        if(pMP == NULL) continue;
        
        Frame* host_frame = pMP->hostFeature_->frame;
        Vector3d pHost = pMP->hostFeature_->f * (1.0/pMP->GetIdist());

        list<Feature*> observations = pMP->GetObservations();
        for(auto it_ft = observations.begin(); it_ft != observations.end(); ++it_ft)
        {
            if((*it_ft)->frame->id_ == host_frame->id_) continue;
            assert((*it_ft)->point == pMP);

			const Sophus::SE3 Ttw = (*it_ft)->frame->GetPoseSE3();
			const Sophus::SE3 Twh = host_frame->GetPoseInverseSE3();
			SE3 Tth = Ttw * Twh;
            Vector3d pTarget = Tth * pHost;
            Vector2d e = hso::project2d((*it_ft)->f) - hso::project2d(pTarget);
            if((*it_ft)->type == Feature::EDGELET)
            {
                errors_ls.push_back(fabs((*it_ft)->grad.transpose()*e));
            }
            else
            {
                errors_pt.push_back(e.norm());
            }
        }
    }

    if(!errors_pt.empty() && !errors_ls.empty())
    {
        huber_corner = 1.4826*hso::getMedian(errors_pt);
        huber_edge = 1.4826*hso::getMedian(errors_ls);
    }
    else if(errors_pt.empty() && !errors_ls.empty())
    {
        huber_corner = 1.0 / pFrame->cam_->errorMultiplier2();
        huber_edge = 1.4826*hso::getMedian(errors_ls);
    }
    else if(!errors_pt.empty() && errors_ls.empty())
    {
        huber_corner = 1.4826*hso::getMedian(errors_pt);
        huber_edge   = 0.5 / pFrame->cam_->errorMultiplier2();
    }
    else
    {
        assert(false);
    }
}

void ComputeHuberthreshold(std::vector<FramePtr> vpFrames, float &huber_corner, float &huber_edge)
{
    set<Point*> spMPs;                  
    for(std::vector<FramePtr>::iterator sit=vpFrames.begin(), sitend=vpFrames.end(); sit!=sitend; sit++)
    {
        FramePtr pFrame = *sit;

        for(std::list<Feature*>::iterator itf=pFrame->fts_.begin(), itend=pFrame->fts_.end(); itf!=itend; itf++)
        {
            if((*itf)->point == NULL) continue;
            assert((*itf)->point->type_ != Point::TYPE_CANDIDATE);
            spMPs.insert((*itf)->point);
        }
    }


    vector<float> errors_pt, errors_ls; 
    for(std::set<Point*>::iterator itp=spMPs.begin(), itpend=spMPs.end(); itp!=itpend; itp++)
    {
        Point* pMP = *itp;
        if(pMP == NULL) continue;
        
        Frame* host_frame = pMP->hostFeature_->frame;
        Vector3d pHost = pMP->hostFeature_->f * (1.0/pMP->GetIdist());

        list<Feature*> observations = pMP->GetObservations();
        for(auto it_ft = observations.begin(); it_ft != observations.end(); ++it_ft)
        {
            if((*it_ft)->frame->id_ == host_frame->id_) continue;
            assert((*it_ft)->point == pMP);

			const Sophus::SE3 Ttw = (*it_ft)->frame->GetPoseSE3();
			const Sophus::SE3 Twh = host_frame->GetPoseInverseSE3();
			SE3 Tth = Ttw * Twh;
            Vector3d pTarget = Tth * pHost;
            Vector2d e = hso::project2d((*it_ft)->f) - hso::project2d(pTarget);
            e *= 1.0 / (1<<(*it_ft)->level);

            if((*it_ft)->type == Feature::EDGELET)
            {
                errors_ls.push_back(fabs((*it_ft)->grad.transpose()*e));
            }
            else
            {
                errors_pt.push_back(e.norm());
            }
        }
    }
    
    if(!errors_pt.empty() && !errors_ls.empty())
    {
        huber_corner = 1.4826*hso::getMedian(errors_pt);
        huber_edge = 1.4826*hso::getMedian(errors_ls);
    }
    else if(errors_pt.empty() && !errors_ls.empty())
    {
        huber_corner = 1.0 / vpFrames[0]->cam_->errorMultiplier2();
        huber_edge = 1.4826*hso::getMedian(errors_ls);
    }
    else if(!errors_pt.empty() && errors_ls.empty())
    {
        huber_corner = 1.4826*hso::getMedian(errors_pt);
        huber_edge   = 0.5 / vpFrames[0]->cam_->errorMultiplier2();
    }
    else
    {
        assert(false);
    }
}

void ComputeHuberthreshold(std::list<Point*> lpMPs, FramePtr pFrame, float &huber_corner, float &huber_edge)
{
    vector<float> errors_pt, errors_ls; 
    for(std::list<Point*>::iterator itp=lpMPs.begin(), itpend=lpMPs.end(); itp!=itpend; itp++)
    {
        Point* pMP = *itp;
        if(pMP == NULL) continue;
        
        Frame* host_frame = pMP->hostFeature_->frame;
        Vector3d pHost = pMP->hostFeature_->f * (1.0/pMP->GetIdist());

        list<Feature*> observations = pMP->GetObservations();
        for(auto it_ft = observations.begin(); it_ft != observations.end(); ++it_ft)
        {
            if((*it_ft)->frame->id_ == host_frame->id_) continue;
            assert((*it_ft)->point == pMP);

			const Sophus::SE3 Ttw = (*it_ft)->frame->GetPoseSE3();
			const Sophus::SE3 Twh = host_frame->GetPoseInverseSE3();
			SE3 Tth = Ttw * Twh;
            Vector3d pTarget = Tth * pHost;
            Vector2d e = hso::project2d((*it_ft)->f) - hso::project2d(pTarget);

            if((*it_ft)->type == Feature::EDGELET)
            {
                errors_ls.push_back(fabs((*it_ft)->grad.transpose()*e));
            }
            else
            {
                errors_pt.push_back(e.norm());
            }
        }
    }
    
    if(!errors_pt.empty() && !errors_ls.empty())
    {
        huber_corner = 1.4826*hso::getMedian(errors_pt);
        huber_edge = 1.4826*hso::getMedian(errors_ls);
    }
    else if(errors_pt.empty() && !errors_ls.empty())
    {
        huber_corner = 1.0 / pFrame->cam_->errorMultiplier2();
        huber_edge = 1.4826*hso::getMedian(errors_ls);
    }
    else if(!errors_pt.empty() && errors_ls.empty())
    {
        huber_corner = 1.4826*hso::getMedian(errors_pt);
        huber_edge   = 0.5 / pFrame->cam_->errorMultiplier2();
    }
    else
    {
        assert(false);
    }
}

void ComputeHuberthreshold(std::set<Point*> spMPs, Frame* pFrame, float &huber_corner, float &huber_edge)
{
    vector<float> errors_pt, errors_ls; 
    for(std::set<Point*>::iterator itp=spMPs.begin(), itpend=spMPs.end(); itp!=itpend; itp++)
    {
        Point* pMP = *itp;
        if(pMP == NULL) continue;
        
        Frame* host_frame = pMP->hostFeature_->frame;
        Vector3d pHost = pMP->hostFeature_->f * (1.0/pMP->GetIdist());

        list<Feature*> observations = pMP->GetObservations();
        for(auto it_ft = observations.begin(); it_ft != observations.end(); ++it_ft)
        {
            if((*it_ft)->frame->id_ == host_frame->id_) continue;
            assert((*it_ft)->point == pMP);

			const Sophus::SE3 Ttw = (*it_ft)->frame->GetPoseSE3();
			const Sophus::SE3 Twh = host_frame->GetPoseInverseSE3();
			SE3 Tth = Ttw * Twh;
            Vector3d pTarget = Tth * pHost;
            Vector2d e = hso::project2d((*it_ft)->f) - hso::project2d(pTarget);

            if((*it_ft)->type == Feature::EDGELET)
            {
                errors_ls.push_back(fabs((*it_ft)->grad.transpose()*e));
            }
            else
            {
                errors_pt.push_back(e.norm());
            }
        }
    }
    
    if(!errors_pt.empty() && !errors_ls.empty())
    {
        huber_corner = 1.4826*hso::getMedian(errors_pt);
        huber_edge = 1.4826*hso::getMedian(errors_ls);
    }
    else if(errors_pt.empty() && !errors_ls.empty())
    {
        huber_corner = 1.0 / pFrame->cam_->errorMultiplier2();
        huber_edge = 1.4826*hso::getMedian(errors_ls);
    }
    else if(!errors_pt.empty() && errors_ls.empty())
    {
        huber_corner = 1.4826*hso::getMedian(errors_pt);
        huber_edge   = 0.5 / pFrame->cam_->errorMultiplier2();
    }
    else
    {
        assert(false);
    }
}
Eigen::MatrixXd Marginalize(const Eigen::MatrixXd &H, const int &start, const int &end)
{
    
    const int a = start;
    const int b = end-start+1;
    const int c = H.cols() - (end+1);

    Eigen::MatrixXd Hn = Eigen::MatrixXd::Zero(H.rows(),H.cols());
    if(a>0)
    {
        Hn.block(0,0,a,a) = H.block(0,0,a,a);
        Hn.block(0,a+c,a,b) = H.block(0,a,a,b);
        Hn.block(a+c,0,b,a) = H.block(a,0,b,a);
    }
    if(a>0 && c>0)
    {
        Hn.block(0,a,a,c) = H.block(0,a+b,a,c);
        Hn.block(a,0,c,a) = H.block(a+b,0,c,a);
    }
    if(c>0)
    {
        Hn.block(a,a,c,c) = H.block(a+b,a+b,c,c);
        Hn.block(a,a+c,c,b) = H.block(a+b,a,c,b);
        Hn.block(a+c,a,b,c) = H.block(a,a+b,b,c);
    }
    Hn.block(a+c,a+c,b,b) = H.block(a,a,b,b);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Hn.block(a+c,a+c,b,b),Eigen::ComputeThinU | Eigen::ComputeThinV); 
    Eigen::JacobiSVD<Eigen::MatrixXd>::SingularValuesType singularValues_inv=svd.singularValues();     
    for (int i=0; i<b; ++i)
    {
        if (singularValues_inv(i)>1e-6)
            singularValues_inv(i)=1.0/singularValues_inv(i);
        else singularValues_inv(i)=0;
    }
    Eigen::MatrixXd invHb = svd.matrixV()*singularValues_inv.asDiagonal()*svd.matrixU().transpose();  
    Hn.block(0,0,a+c,a+c) = Hn.block(0,0,a+c,a+c) - Hn.block(0,a+c,a+c,b)*invHb*Hn.block(a+c,0,b,a+c);
    Hn.block(a+c,a+c,b,b) = Eigen::MatrixXd::Zero(b,b);
    Hn.block(0,a+c,a+c,b) = Eigen::MatrixXd::Zero(a+c,b);
    Hn.block(a+c,0,b,a+c) = Eigen::MatrixXd::Zero(b,a+c);

    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(H.rows(),H.cols());
    if(a>0)
    {
        res.block(0,0,a,a) = Hn.block(0,0,a,a);
        res.block(0,a,a,b) = Hn.block(0,a+c,a,b);
        res.block(a,0,b,a) = Hn.block(a+c,0,b,a);
    }
    if(a>0 && c>0)
    {
        res.block(0,a+b,a,c) = Hn.block(0,a,a,c);
        res.block(a+b,0,c,a) = Hn.block(a,0,c,a);
    }
    if(c>0)
    {
        res.block(a+b,a+b,c,c) = Hn.block(a,a,c,c);
        res.block(a+b,a,c,b) = Hn.block(a,a+c,c,b);
        res.block(a,a+b,b,c) = Hn.block(a+c,a,b,c);
    }

    res.block(a,a,b,b) = Hn.block(a+c,a+c,b,b);

    return res;
}


} // namespace ba
} // namespace vihso


