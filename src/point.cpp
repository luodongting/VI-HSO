#include <stdexcept>
#include <vihso/point.h>
#include <vihso/frame.h>
#include <vihso/feature.h>
#include <vihso/config.h>
#include "vihso/vikit/math_utils.h"

namespace vihso 
{
	
int Point::point_counter_ = 0;
mutex Point::mGlobalMutex;
Point::Point(const Vector3d& pos) : id_(point_counter_++), pos_(pos), type_(TYPE_UNKNOWN),                            
                                    last_published_ts_(0), last_structure_optim_(0), last_projected_kf_id_(-1), n_failed_reproj_(0), n_succeeded_reproj_(0),
                                    isBad_(false), v_pt_(NULL), vPoint_(NULL), nBA_(0), normal_set_(false),
                                    nApplyScaled(0), nGetAllMapPoints(0), nFullBAmaxKFID(0), mnBALocalForKF(0)
{}
Point::Point(const Vector3d& pos, Feature* ftr) :   id_(point_counter_++), pos_(pos), type_(TYPE_UNKNOWN),                            
                                                    last_published_ts_(0), last_structure_optim_(0), last_projected_kf_id_(-1), n_failed_reproj_(0), n_succeeded_reproj_(0),
                                                    isBad_(false), v_pt_(NULL), vPoint_(NULL), nBA_(0), normal_set_(false),
                                                    nApplyScaled(0), nGetAllMapPoints(0), nFullBAmaxKFID(0), mnBALocalForKF(0)
{
    addFrameRef(ftr);
}
Point::~Point()
{}
void Point::addFrameRef(Feature* ftr)
{
    unique_lock<mutex> lock(mMutexFeatures);
    obs_.push_front(ftr);
}
bool Point::deleteFrameRef(Frame* frame)
{
    unique_lock<mutex> lock(mMutexFeatures);
    for(auto it=obs_.begin(), ite=obs_.end(); it!=ite; ++it)
    {
        if((*it)->frame == frame)
        {
            obs_.erase(it);
            return true;
        } 
    }
    return false;
}
Feature* Point::findFrameRef(Frame* frame)
{
    list<Feature*> observations;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        observations = GetObservations();
    }
    for(auto it=observations.begin(), ite=observations.end(); it!=ite; ++it)
    {
        if((*it)->frame == frame)
            return *it;
    }
    return NULL;    
}
void Point::initNormal()
{
    list<Feature*> observations = GetObservations();
    assert(!observations.empty());
    const Feature* ftr = observations.back();
    assert(ftr->frame != NULL);
    normal_ = ftr->frame->GetRotation().transpose()*(-ftr->f);
    Vector3d pose = GetWorldPos();
    normal_information_ = DiagonalMatrix<double,3,3>(pow(20/(pose-ftr->frame->GetCameraCenter()).norm(),2), 1.0, 1.0);
    normal_set_ = true;
}
bool Point::getCloseViewObs(const Vector3d& framepos, Feature* &ftr)
{
    Vector3d pose = GetWorldPos();
    list<Feature*> observations = GetObservations();
    Vector3d obs_dir(framepos - pose);
    obs_dir.normalize();
    auto min_it=observations.begin();
    double min_cos_angle = 0;
    for(auto it=observations.begin(), ite=observations.end(); it!=ite; ++it)
    {
        Vector3d dir((*it)->frame->GetCameraCenter() - pose);
        dir.normalize();
        double cos_angle = obs_dir.dot(dir);
        if(cos_angle > min_cos_angle)
        {
            min_cos_angle = cos_angle;
            min_it = it;
        }
    }
    ftr = *min_it;
    if(min_cos_angle < 0.5)
        return false; 
    return true;
}
void Point::optimize(const size_t n_iter)
{
    list<Feature*> observations = GetObservations();
    double idist = GetIdist();
    double old_idist = idist;  
    double chi2 = 0.0;  
    double H=0,b=0;
    if(observations.size() < 5) return;
    Frame* host = hostFeature_->frame;  
    for(size_t i=0; i<n_iter; i++)  
    {
        H=b=0;
        double new_chi2 = 0.0;  
        for(auto it=observations.begin(); it!=observations.end(); ++it) 
        {
			Feature* ft = *it;  
            if(ft->frame->id_ == host->id_)	continue;
            Frame* target = ft->frame;  
			const Sophus::SE3 Ttw = target->GetPoseSE3();
			const Sophus::SE3 Twh = host->GetPoseInverseSE3();
            SE3 Tth = Ttw * Twh;	
            Vector3d pTarget = Tth * (hostFeature_->f*(1.0/idist));    
            Vector2d e(hso::project2d(ft->f) - hso::project2d(pTarget));    
            Vector2d J; 
            jacobian_id2uv(pTarget, Tth, idist, hostFeature_->f, J);
            if((*it)->type == Feature::EDGELET)
            {
                double e_edge = (*it)->grad.transpose() * e;    
                double J_edge = (*it)->grad.transpose() * J;    
                new_chi2 += e_edge * e_edge;
                H += J_edge*J_edge;
                b -= J_edge*e_edge;
            }
            else
            {
                new_chi2 += e.squaredNorm();
                H += J.transpose() * J;
                b -= J.transpose() * e;
            }
        }
        const double id_step = b/H;
        if((i > 0 && new_chi2 > chi2) || (bool) std::isnan(id_step))    
        {
            #ifdef POINT_OPTIMIZER_DEBUG
                cout << "it " << i << "\t FAILURE \t new_chi2 = " << new_chi2 << endl;
            #endif
            idist = old_idist;
            break;
        }
        double new_id = idist+id_step;
        old_idist = idist;
        idist = new_id;
        chi2 = new_chi2;
        #ifdef POINT_OPTIMIZER_DEBUG
            cout << "it " << i
            << "\t Success \t new_chi2 = " << new_chi2
            << "\t norm(b) = " << fabs(id_step)
            << endl;
        #endif
        if(fabs(id_step) < 0.00001) break;
    }
    #ifdef POINT_OPTIMIZER_DEBUG
        cout << endl;
    #endif
    UpdatePoseidist(idist);
}
void Point::optimizeLM(const size_t n_iter)
{
    double chi2 = 0.0;  
    double rho = 0;
    double mu = 0.1;
    double nu = 2.0;
    bool stop = false;
    int n_trials = 0;
    Matrix3d A;
    Vector3d b;
    const int n_trials_max = 5;
    list<Feature*> observations = GetObservations();
    Vector3d pose = GetWorldPos();
    for(auto it=observations.begin(); it!=observations.end(); ++it)
    {
        const Vector3d p_in_f((*it)->frame->GetPoseSE3() * pose); 
        const Vector2d e(hso::project2d((*it)->f) - hso::project2d(p_in_f));
        chi2 += e.squaredNorm();    
    }
    for(size_t iter = 0; iter < n_iter; iter++)
    {
        rho = 0;
        n_trials = 0;
        do
        {
            Vector3d new_pos;
            double new_chi2 = 0.0;
            A.setZero();
            b.setZero();
            for(auto it=observations.begin(); it!=observations.end(); ++it)
            {
                Matrix23d J;
                const Vector3d p_in_f((*it)->frame->GetPoseSE3() * pose);
                Point::jacobian_xyz2uv(p_in_f, (*it)->frame->GetRotation(), J);
                const Vector2d e(hso::project2d((*it)->f) - hso::project2d(p_in_f));
                A.noalias() += J.transpose() * J ;
                b.noalias() -= J.transpose() * e ;
            }
            A += (A.diagonal()*mu).asDiagonal();
            const Vector3d dp(A.ldlt().solve(b));
        if(!(bool)std::isnan((double)dp[0]))
        {
            new_pos = pose + dp;
            for(auto it=observations.begin(); it!=observations.end(); ++it)
            {
                Matrix23d J;
                const Vector3d p_in_f((*it)->frame->GetPoseSE3() * new_pos);
                const Vector2d e(hso::project2d((*it)->f) - hso::project2d(p_in_f));
                new_chi2 += e.squaredNorm();
            }
            rho = chi2 - new_chi2;
        }
        else
        {
            #ifdef POINT_OPTIMIZER_DEBUG
                cout << "Matrix is close to singular!" << endl;
                cout << "H = " << A << endl;
                cout << "Jres = " << b << endl;
            #endif
            rho = -1;
        }
        if(rho > 0)
        {
            SetWorldPos(new_pos);
            chi2 = new_chi2;
            stop = hso::norm_max(dp) <= EPS;
            mu *= std::max(1./3., std::min(1.-std::pow(2*rho-1,3), 2./3.));
            nu = 2.;
            #ifdef POINT_OPTIMIZER_DEBUG
                cout << "It. " << iter
                << "\t Trial " << n_trials
                << "\t Success"
                << "\t new_chi2 = " << new_chi2
                << "\t mu = " << mu
                << "\t nu = " << nu
                << endl;
            #endif
        }
        else
        {
            mu *= nu;
            nu *= 2.;
            ++n_trials;
            if (n_trials >= n_trials_max) stop = true;
            #ifdef POINT_OPTIMIZER_DEBUG
                cout << "It. " << iter
                << "\t Trial " << n_trials
                << "\t Failure"
                << "\t new_chi2 = " << new_chi2
                << "\t mu = " << mu
                << "\t nu = " << nu
                << endl;
            #endif
        }
        }while(!(rho>0 || stop));
        if (stop) break;
    }
    #ifdef POINT_OPTIMIZER_DEBUG
        cout << "======================" << endl;
    #endif
}
void Point::optimizeID(const size_t n_iter)
{
    double idist = GetIdist();
    double oldEnergy = 0;
    double rho = 0;
    double mu = 0.1;
    double nu = 2.0;
    bool stop = false;
    int n_trials = 0;
    const int n_trials_max = 5;
    double H = 0;
    double b = 0;
    const float idistOld = idist;
    Frame* host = hostFeature_->frame;
    list<Feature*> observations = GetObservations();
    for(auto it=observations.begin(); it!=observations.end(); ++it)
    {
        Feature* ft = *it;
        if(ft->frame->id_ == host->id_)
            continue;
        Frame* target = ft->frame;
		const Sophus::SE3 Ttw = target->GetPoseSE3();
		const Sophus::SE3 Twh = host->GetPoseInverseSE3();
        SE3 Tth = Ttw * Twh;
        Vector3d pTarget = Tth * (hostFeature_->f*(1.0/idist));
        Vector2d e(hso::project2d(ft->f) - hso::project2d(pTarget));
        oldEnergy += e.squaredNorm();
    }
    assert(oldEnergy > 0);
    for(size_t iter = 0; iter < n_iter; iter++)
    {
        rho = 0;
        n_trials = 0;
        do
        {
            double newid;
            double newEnergy = 0.0;
            H = 0; b = 0;
            for(auto it=observations.begin(); it!=observations.end(); ++it)
            {
                Feature* ft = *it;
                if(ft->frame->id_ == host->id_)
                    continue;
                Frame* target = ft->frame;
				const Sophus::SE3 Ttw = target->GetPoseSE3();
				const Sophus::SE3 Twh = host->GetPoseInverseSE3();
				SE3 Tth = Ttw * Twh;
                Vector3d pTarget = Tth * (hostFeature_->f*(1.0/idist));
                Vector2d e(hso::project2d(ft->f) - hso::project2d(pTarget));
                Vector2d Juvdd;
                jacobian_id2uv(pTarget, Tth, idist, hostFeature_->f, Juvdd);
                H += Juvdd.transpose()*Juvdd;
                b -= Juvdd.transpose()*e;
            }
            H *= 1+mu;
            double step = (1.0/H)*b;
            if(!(bool)std::isnan(step))
            {
                newid = idist+step;
                for(auto it=observations.begin(); it!=observations.end(); ++it)
                {
                    Feature* ft = *it;
                    if(ft->frame->id_ == host->id_)
                        continue;
                    Frame* target = ft->frame;
					const Sophus::SE3 Ttw = target->GetPoseSE3();
					const Sophus::SE3 Twh = host->GetPoseInverseSE3();
            		SE3 Tth = Ttw * Twh;
                    Vector3d pTarget = Tth * (hostFeature_->f*(1.0/newid));
                    Vector2d e(hso::project2d(ft->f) - hso::project2d(pTarget)); 
                    newEnergy += e.squaredNorm();
                }
                rho = oldEnergy - newEnergy;
            }
            else
            {
                #ifdef POINT_OPTIMIZER_DEBUG
                    cout << "Matrix is close to singular!" << endl;
                    cout << "H = " << H << endl;
                    cout << "b = " << b << endl;
                    cout << "Energy = " << oldEnergy << endl;
                #endif
                rho = -1;
            }
            if(rho > 0)
            {
                idist = newid;
                oldEnergy = newEnergy;
                stop = (step <= EPS);
                mu *= std::max(1./3., std::min(1.-std::pow(2*rho-1,3), 2./3.));
                nu = 2.;
                #ifdef POINT_OPTIMIZER_DEBUG
                    cout << "It. " << iter
                    << "\t Trial " << n_trials
                    << "\t Success"
                    << "\t new_chi2 = " << newEnergy
                    << "\t mu = " << mu
                    << "\t nu = " << nu
                    << endl;
                #endif
            }
            else
            {
                mu *= nu;
                nu *= 2.;
                ++n_trials;
                if (n_trials >= n_trials_max) stop = true;
                #ifdef POINT_OPTIMIZER_DEBUG
                    cout << "It. " << iter
                    << "\t Trial " << n_trials
                    << "\t Failure"
                    << "\t new_chi2 = " << newEnergy
                    << "\t mu = " << mu
                    << "\t nu = " << nu
                    << endl;
                #endif
            }   
        }while(!(rho>0 || stop));
        if(stop) break;
    }
    UpdatePoseidist(idist);
    #ifdef POINT_OPTIMIZER_DEBUG
        cout << "Before = " << idistOld << "\t" << "After = " << idist_ << endl;
    #endif
    #ifdef POINT_OPTIMIZER_DEBUG
        cout << "======================" << endl;
    #endif
}
void Point::SetWorldPos(const Vector3d &pos)
{
	unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    pos_ = pos; 
}
void Point::SetIdist(const double &idist)
{
	unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    idist_ = idist; 
}
Vector3d Point::GetWorldPos()
{
    unique_lock<mutex> lock(mMutexPos);
    return pos_; 
}
double Point::GetIdist()
{
    unique_lock<mutex> lock(mMutexPos);
    return idist_;  
}
void Point::UpdatePose()
{
    unique_lock<mutex> lock(mMutexPos);
    if(isBad_) return;
    Vector3d pHost = hostFeature_->f*(1.0/idist_);          
    pos_ = hostFeature_->frame->GetPoseInverseSE3() * pHost;   
}
void Point::UpdatePoseScale(const double &scale)
{
	unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    if(isBad_) return;
    idist_ = idist_/scale;  
    Vector3d pHost = hostFeature_->f*(1.0/idist_);          
    pos_ = hostFeature_->frame->GetPoseInverseSE3() * pHost;   
}
void Point::UpdatePoseidist(const double &idist)
{
	unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    if(isBad_) return;
    idist_ = idist;  
    Vector3d pHost = hostFeature_->f*(1.0/idist_);          
    pos_ = hostFeature_->frame->GetPoseInverseSE3() * pHost;   
}
list<vihso::Feature*> Point::GetObservations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return obs_;
}
size_t Point::GetNumofObs()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return obs_.size();
}
void Point::ClearObservations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    obs_.clear();
}
} 
