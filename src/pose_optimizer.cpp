#include <stdexcept>
#include <vihso/pose_optimizer.h>
#include <vihso/frame.h>
#include <vihso/feature.h>
#include <vihso/point.h>
#include <vihso/config.h>
#include "vihso/vikit/math_utils.h"
#include "vihso/vikit/robust_cost.h"
namespace vihso {
namespace pose_optimizer {
void optimizeLevenbergMarquardt3rd( const double reproj_thresh, 
                                    const size_t n_iter,        
                                    const bool verbose,         
                                    FramePtr& frame,            
                                    double& estimated_scale,    
                                    double& error_init,         
                                    double& error_final,        
                                    size_t& num_obs)            
{
    double chi2=0.0, rho=0, mu=0.1, nu=2.0;
    bool stop = false;
    int n_trials = 0;
    const int n_trials_max = 5;
    vector<double> chi2_vec_init;  
    vector<double> chi2_vec_final;  
    chi2_vec_init.reserve(frame->fts_.size());
    chi2_vec_final.reserve(frame->fts_.size());
    hso::robust_cost::HuberWeightFunction weight_function;
    Matrix6d A; A.setZero();
    Vector6d b; b.setZero();
    vector<float> errors_pt;    
    vector<float> errors_ls;    
    errors_pt.reserve(frame->fts_.size());
    errors_ls.reserve(frame->fts_.size());
    vector<Vector3d> v_host;    
    v_host.reserve(frame->fts_.size());
    for(Features::iterator it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
    {
        if((*it)->point == NULL) continue;
        Feature* ft = *it;
        Frame* host = ft->point->hostFeature_->frame;
        Vector3d pHost = ft->point->hostFeature_->f * (1.0/ft->point->GetIdist());
		const Sophus::SE3 Ttw = frame->GetPoseSE3();
		const Sophus::SE3 Twh = host->GetPoseInverseSE3();
        SE3 Tth = Ttw * Twh;
        Vector3d pTarget = Tth * pHost;
        Vector2d e = hso::project2d(ft->f) - hso::project2d(pTarget);   
        e *= 1.0 / (1<<ft->level);                                      
        if(ft->type == Feature::EDGELET)
        {
            float error_ls = ft->grad.transpose()*e;
            errors_ls.push_back(fabs(error_ls));
            chi2_vec_init.push_back(error_ls*error_ls);
        }
        else
        {
            float error_pt = e.norm();
            errors_pt.push_back(error_pt);
            chi2_vec_init.push_back(error_pt*error_pt);
        }
        v_host.push_back(pHost);
    }
    if(errors_pt.empty() && errors_ls.empty()) return;
    hso::robust_cost::MADScaleEstimator scale_estimator;    
    float estimated_scale_pt;   
    float estimated_scale_ls;   
    if(!errors_pt.empty() && !errors_ls.empty())        
    {
        estimated_scale_pt = scale_estimator.compute(errors_pt);    
        estimated_scale_ls = scale_estimator.compute(errors_ls);
    }
    else if(!errors_pt.empty() && errors_ls.empty())    
    {
        estimated_scale_pt = scale_estimator.compute(errors_pt);
        estimated_scale_ls = 0.5*estimated_scale_pt;
    }
    else if(errors_pt.empty() && !errors_ls.empty())    
    {
        estimated_scale_ls = scale_estimator.compute(errors_ls);
        estimated_scale_pt = 2*estimated_scale_ls;
    }
    else
    {
        assert(false);
    }
    estimated_scale = estimated_scale_pt;
    int idx_host = 0;   
    for(Features::iterator it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
    {
        if((*it)->point == NULL) continue;
        Frame* host = (*it)->point->hostFeature_->frame;
        Vector3d pHost = v_host[idx_host];
		const Sophus::SE3 Ttw = frame->GetPoseSE3();
		const Sophus::SE3 Twh = host->GetPoseInverseSE3();
        SE3 Tth = Ttw * Twh;
        Vector3d pTarget = Tth * pHost;
        Vector2d e = hso::project2d((*it)->f) - hso::project2d(pTarget);
        e *= 1.0 / (1<<(*it)->level);
        if((*it)->type == Feature::EDGELET)
        {
            double error_ls = (*it)->grad.transpose()*e;
            double weight = weight_function.value(fabs(error_ls)/estimated_scale_ls);
            if((*it)->point->type_ == Point::TYPE_TEMPORARY) weight *= 0.5;
            chi2 += error_ls*error_ls * weight;
        }
        else
        {
            double error_pt = e.norm();
            double weight = weight_function.value(error_pt/estimated_scale_pt);
            if((*it)->point->type_ == Point::TYPE_TEMPORARY) weight *= 0.5;
            chi2 += error_pt*error_pt * weight;
        }
        ++idx_host;
    }
    num_obs = errors_pt.size()+errors_ls.size();
    for(size_t iter=0; iter<n_iter; iter++) 
    {
        rho = 0;        
        n_trials = 0;   
        do
        {
            SE3 T_new;
            double new_chi2 = 0.0;
            A.setZero();
            b.setZero();
            idx_host = 0;
            for(Features::iterator it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
            {
                if((*it)->point == NULL) continue;
                Frame* host = (*it)->point->hostFeature_->frame;
                Vector3d pHost = v_host[idx_host];
				const Sophus::SE3 Ttw = frame->GetPoseSE3();
				const Sophus::SE3 Twh = host->GetPoseInverseSE3();
				SE3 Tth = Ttw * Twh;
                Vector3d pTarget = Tth * pHost;
                Matrix26d J;
                Frame::jacobian_xyz2uv(pTarget, J);
                Vector2d e = hso::project2d((*it)->f) - hso::project2d(pTarget);
                double sqrt_inv_cov = 1.0 / (1<<(*it)->level);
                e *= sqrt_inv_cov;
                J *= sqrt_inv_cov;
                if((*it)->type == Feature::EDGELET)
                {
                    Matrix<double,1,6> J_edge = (*it)->grad.transpose()*J;
                    double e_edge = (*it)->grad.transpose()*e;
                    double weight = weight_function.value(fabs(e_edge)/estimated_scale_ls);
                    if((*it)->point->type_ == Point::TYPE_TEMPORARY) weight *= 0.5;
                    A.noalias() += J_edge.transpose()*J_edge*weight;
                    b.noalias() -= J_edge.transpose()*e_edge*weight;
                }
                else
                {
                    double weight = weight_function.value(e.norm()/estimated_scale_pt);
                    if((*it)->point->type_ == Point::TYPE_TEMPORARY) weight *= 0.5;
                    A.noalias() += J.transpose()*J*weight;
                    b.noalias() -= J.transpose()*e*weight;
                }
                ++idx_host;
            }
            A += (A.diagonal()*mu).asDiagonal();    
            const Vector6d dT(A.ldlt().solve(b));
            if(!(bool) std::isnan((double)dT[0]))
            {
                T_new = SE3::exp(dT)*frame->GetPoseSE3();
                idx_host = 0;
                for(Features::iterator it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
                {
                    if((*it)->point == NULL) continue;
                    Frame* host = (*it)->point->hostFeature_->frame;
                    Vector3d pHost = v_host[idx_host];
                    SE3 Tth = T_new * host->GetPoseInverseSE3();
                    Vector3d pTarget = Tth * pHost;
                    Vector2d e = hso::project2d((*it)->f) - hso::project2d(pTarget);
                    double sqrt_inv_cov = 1.0 / (1<<(*it)->level);
                    e *= sqrt_inv_cov;
                    if((*it)->type == Feature::EDGELET)
                    {
                        double error_ls = (*it)->grad.transpose()*e;
                        double weight = weight_function.value(fabs(error_ls)/estimated_scale_ls);
                        if((*it)->point->type_ == Point::TYPE_TEMPORARY) weight *= 0.5;
                        new_chi2 += error_ls*error_ls * weight;
                    }
                    else
                    {
                        double error_pt = e.norm();
                        double weight = weight_function.value(error_pt/estimated_scale_pt);
                        if((*it)->point->type_ == Point::TYPE_TEMPORARY) weight *= 0.5;
                        new_chi2 += error_pt*error_pt * weight;
                    }
                    ++idx_host;
                }
                rho = chi2 - new_chi2;
            }
            else
                rho = -1;
            if(rho>0)
            {
                frame->SetPose(T_new);
                chi2 = new_chi2;
                stop = hso::norm_max(dT) <= EPS;
                mu *= std::max(1./3., std::min(1.-std::pow(2*rho-1,3), 2./3.)); 
                nu = 2.;
                if(verbose)
                {
                    cout << "It. " << iter
                    << "\t Trial " << n_trials
                    << "\t Success"
                    << "\t new_chi2 = " << new_chi2
                    << "\t mu = " << mu
                    << "\t nu = " << nu
                    << endl;
                }
            }
            else
            {
                mu *= nu;
                nu *= 2.;
                if(mu < 0.0001) mu = 0.0001;
                ++n_trials;
                if(n_trials >= n_trials_max) stop = true;
                if(verbose)
                {
                    cout << "It. " << iter
                    << "\t Trial " << n_trials
                    << "\t Failure"
                    << "\t new_chi2 = " << new_chi2
                    << "\t mu = " << mu
                    << "\t nu = " << nu
                    << endl;
                }
            }
        }while(!(rho>0 || stop));
        if (stop) break;
    }
    const float pixel_variance=1.0;
    frame->Cov_ = pixel_variance*(A*std::pow(frame->cam_->errorMultiplier2(),2)).inverse();
    const float reproj_thresh_scaled_pt = (frame->fts_.size() < 80)?
                                            sqrt(5.991)/frame->cam_->errorMultiplier2() :   
                                            reproj_thresh/frame->cam_->errorMultiplier2();  
    const float reproj_thresh_scaled_ls = 1.3 / frame->cam_->errorMultiplier2();            
    size_t n_deleted_refs = 0;  
    idx_host = 0;
    for(Features::iterator it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
    {
        if((*it)->point == NULL) continue;
        Frame* host = (*it)->point->hostFeature_->frame;
        Vector3d pHost = v_host[idx_host];
		const Sophus::SE3 Ttw = frame->GetPoseSE3();
		const Sophus::SE3 Twh = host->GetPoseInverseSE3();
        SE3 Tth = Ttw * Twh;
        Vector3d pTarget = Tth * pHost;
        Vector2d e = hso::project2d((*it)->f) - hso::project2d(pTarget);
        double sqrt_inv_cov = 1.0 / (1<<(*it)->level);
        e *= sqrt_inv_cov;
        if((*it)->type == Feature::EDGELET)
        {
            double error_ls = (*it)->grad.transpose()*e;
            if(fabs(error_ls) > reproj_thresh_scaled_ls)
            {
                ++n_deleted_refs;
                (*it)->point = NULL;
            }
            chi2_vec_final.push_back(error_ls*error_ls);
        }
        else
        {
            float error_pt = e.norm();
            if(error_pt > reproj_thresh_scaled_pt)
            {
                ++n_deleted_refs;
                (*it)->point = NULL;
            }
            chi2_vec_final.push_back(error_pt*error_pt);
        }
        ++idx_host;
    }
    error_init=0.0;
    error_final=0.0;
    if(!chi2_vec_init.empty())
        error_init = sqrt(hso::getMedian(chi2_vec_init))*frame->cam_->errorMultiplier2();
    if(!chi2_vec_final.empty())
        error_final = sqrt(hso::getMedian(chi2_vec_final))*frame->cam_->errorMultiplier2();
    estimated_scale *= frame->cam_->errorMultiplier2();
    num_obs -= n_deleted_refs;
    if(verbose)
        std::cout   << "n obs = " << num_obs
                    << "\t n deleted obs = " << n_deleted_refs
                    << "\t scale = " << estimated_scale
                    << "\t error init = " << error_init
                    << "\t error end = " << error_final << std::endl;
    frame->m_error_in_px = error_final<1.5? 1.0 : 1.5/error_final;
}
} 
} 
