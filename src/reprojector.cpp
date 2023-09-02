#include <algorithm>
#include <stdexcept>
#include <vihso/reprojector.h>
#include <vihso/frame.h>
#include <vihso/point.h>
#include <vihso/feature.h>
#include <vihso/map.h>
#include <vihso/config.h>
#include <vihso/depth_filter.h>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include "vihso/camera.h"
namespace vihso {
Reprojector::Reprojector(hso::AbstractCamera* cam, Map* pMap) : 
	mpMap(pMap), sum_seed_(0), sum_temp_(0), nFeatures_(0)
{
    initializeGrid(cam);
}
Reprojector::~Reprojector()
{
    std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell* c){ delete c; });
    std::for_each(grid_.seeds.begin(), grid_.seeds.end(), [&](Sell* s){ delete s; });
}
inline int Reprojector::caculateGridSize(const int wight, const int height, const int N)
{
    return floorf(sqrt(float(wight*height)/N)*0.6);
}
void Reprojector::initializeGrid(hso::AbstractCamera* cam)
{
    grid_.cell_size = caculateGridSize(cam->width(), cam->height(), Config::maxFts());  
    grid_.grid_n_cols = std::ceil(static_cast<double>(cam->width()) /grid_.cell_size);
    grid_.grid_n_rows = std::ceil(static_cast<double>(cam->height())/grid_.cell_size);
    grid_.cells.resize(grid_.grid_n_cols*grid_.grid_n_rows);
    grid_.seeds.resize(grid_.grid_n_cols*grid_.grid_n_rows);
    std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell* &c){ c = new Cell; });
    std::for_each(grid_.seeds.begin(), grid_.seeds.end(), [&](Sell* &s){ s = new Sell; });
    grid_.cell_order.resize(grid_.cells.size());
    for(size_t i=0; i<grid_.cells.size(); ++i)
        grid_.cell_order[i] = i;
    std::random_shuffle(grid_.cell_order.begin(), grid_.cell_order.end()); 
}
void Reprojector::resetGrid()
{
    n_matches_ = 0;n_trials_ = 0;n_seeds_ = 0;n_filters_ = 0;
    std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell* c){ c->clear(); });
    std::for_each(grid_.seeds.begin(), grid_.seeds.end(), [&](Sell* s){ s->clear(); });
    std::random_shuffle(grid_.cell_order.begin(), grid_.cell_order.end());
    nFeatures_ = 0;
}
void Reprojector::reprojectMap( FramePtr frame,                                                 
                                std::vector< std::pair<FramePtr, std::size_t> >& overlap_kfs)   
{
    resetGrid();
    std::vector< pair<Vector2d, Point*> > allPixelToDistribute;
    VIHSO_START_TIMER("reproject_kfs");
    if(!mpMap->mCandidatesManager.mlTemporaryPoints.empty())    
    {
        DepthFilter::lock_t lock(depth_filter_->seeds_mut_);
        size_t n = 0;
        auto ite = mpMap->mCandidatesManager.mlTemporaryPoints.begin();
        while(ite != mpMap->mCandidatesManager.mlTemporaryPoints.end())
        {
            if(ite->first->seedStates_ == 0) 
            {
                ite++;
                continue;
            } 
            unique_lock<mutex> lock(mpMap->mCandidatesManager.mMutexCandidates);
            mpMap->safeDeleteTempPoint(*ite);
            ite = mpMap->mCandidatesManager.mlTemporaryPoints.erase(ite);
            n++;
        }
        sum_seed_ -= n;
    }
    overlap_kfs.reserve(options_.max_n_kfs);
    FramePtr LastFrame = frame->m_last_frame;
    size_t nCovisibilityGraph = 0; 
    for(vector<Frame*>::iterator it = LastFrame->connectedKeyFrames.begin(); it != LastFrame->connectedKeyFrames.end(); ++it)
    {
        Frame* repframe = *it;
        FramePtr repFrame = NULL;   
        if(!mpMap->getKeyframeById(repframe->id_, repFrame))  
            continue;
        if(repFrame->lastReprojectFrameId_ == frame->id_)   
            continue;
        repFrame->lastReprojectFrameId_ = frame->id_;       
        overlap_kfs.push_back(pair<FramePtr,size_t>(repFrame,0));   
        for(auto ite = repFrame->fts_.begin(); ite != repFrame->fts_.end(); ++ite)
        {
            if((*ite)->point == NULL)
                continue;
            if((*ite)->point->type_ == Point::TYPE_TEMPORARY)
                continue;
            if((*ite)->point->last_projected_kf_id_ == frame->id_)
                continue;
            (*ite)->point->last_projected_kf_id_ = frame->id_;
            if(reprojectPoint(frame, (*ite)->point, allPixelToDistribute))
                overlap_kfs.back().second++;    
        }
        nCovisibilityGraph++;
    }
    LastFrame->connectedKeyFrames.clear();
    list< pair<FramePtr,double> > close_kfs;
    mpMap->GetListCovisibleKeyFrames(frame, close_kfs);
    close_kfs.sort(boost::bind(&std::pair<FramePtr, double>::second, _1) < boost::bind(&std::pair<FramePtr, double>::second, _2));
    size_t n = nCovisibilityGraph;
    for(auto it_frame=close_kfs.begin(), ite_frame=close_kfs.end(); it_frame!=ite_frame && n<15; ++it_frame)    
    {
        FramePtr ref_frame = it_frame->first;
        if(ref_frame->lastReprojectFrameId_ == frame->id_)
            continue;
        ref_frame->lastReprojectFrameId_ = frame->id_;
        overlap_kfs.push_back(pair<FramePtr,size_t>(ref_frame,0));
        for(auto it_ftr=ref_frame->fts_.begin(), ite_ftr=ref_frame->fts_.end(); it_ftr!=ite_ftr; ++it_ftr)
        {
            if((*it_ftr)->point == NULL)
                continue;
            if((*it_ftr)->point->type_ == Point::TYPE_TEMPORARY)
                continue;
            assert((*it_ftr)->point->type_ != Point::TYPE_DELETED);
            if((*it_ftr)->point->last_projected_kf_id_ == frame->id_)
                continue;
            (*it_ftr)->point->last_projected_kf_id_ = frame->id_;
            if(reprojectPoint(frame, (*it_ftr)->point, allPixelToDistribute))
                overlap_kfs.back().second++;
        }
        ++n;
    }
    VIHSO_STOP_TIMER("reproject_kfs");
    VIHSO_START_TIMER("reproject_candidates");    
    {
        unique_lock<mutex> lock(mpMap->mCandidatesManager.mMutexCandidates); 
        auto it=mpMap->mCandidatesManager.mlCandidatePoints.begin();
        while(it!=mpMap->mCandidatesManager.mlCandidatePoints.end())
        {
            if(!reprojectPoint(frame, it->first, allPixelToDistribute))
            {
                it->first->n_failed_reproj_ += 3;   
                if(it->first->n_failed_reproj_ > 30)
                {
                    mpMap->mCandidatesManager.deleteCandidate(*it);
                    it = mpMap->mCandidatesManager.mlCandidatePoints.erase(it);
                    continue;
                }
            }
            ++it;
        }
    } 
    VIHSO_STOP_TIMER("reproject_candidates");
    auto itk = mpMap->mCandidatesManager.mlTemporaryPoints.begin();
    while(itk != mpMap->mCandidatesManager.mlTemporaryPoints.end()) 
    {
        if(itk->first->isBad_){itk++;  continue;}
        assert(itk->first->last_projected_kf_id_ != frame->id_);
        itk->first->last_projected_kf_id_ = frame->id_;
        Point* tempPoint = itk->first;
        Feature* tempFeature = itk->second;
        double idist = tempPoint->GetIdist();
		Sophus::SE3 Twc = tempFeature->frame->GetPoseInverseSE3();
        Vector3d pose = Twc * (tempFeature->f*(1.0/idist));
        tempPoint->SetWorldPos(pose);
        if(!reprojectPoint(frame, itk->first, allPixelToDistribute))
        {
            itk->first->n_failed_reproj_ += 3;
            if(itk->first->n_failed_reproj_ > 30)
                itk->first->isBad_ = true;
        }
        itk++;
    }
    VIHSO_START_TIMER("feature_align");
    if(allPixelToDistribute.size() < (size_t)Config::maxFts()+50) 
    {
        reprojectCellAll(allPixelToDistribute, frame);
    }
    else 
    {
        for(size_t i=0; i<grid_.cells.size(); ++i)
        {
            if(reprojectCell(*grid_.cells.at(grid_.cell_order[i]), frame, false, false))    
            {
                ++n_matches_;
            }
            if(n_matches_ >= (size_t) Config::maxFts())
                break;
        }
        if(n_matches_ < (size_t) Config::maxFts())
        {
            for(size_t i=grid_.cells.size()-1; i>0; --i) 
            {
                if(reprojectCell(*grid_.cells.at(grid_.cell_order[i]), frame, true, false)) 
                {
                    ++n_matches_;
                }
                if(n_matches_ >= (size_t) Config::maxFts())
                    break;
            }
        }
        if(n_matches_ < (size_t) Config::maxFts())
        {
            for(size_t i=0; i<grid_.cells.size(); ++i)
            {
                reprojectCell(*grid_.cells.at(grid_.cell_order[i]), frame, true, true);
                if(n_matches_ >= (size_t) Config::maxFts())
                    break;
            }
        }
    }
    if(n_matches_ < 100 && options_.reproject_unconverged_seeds)
    {
        DepthFilter::lock_t lock(depth_filter_->seeds_mut_);
        for(auto it = depth_filter_->seeds_.begin(); it != depth_filter_->seeds_.end(); ++it)
        {
            if(sqrt(it->sigma2) < it->z_range/options_.reproject_seed_thresh && !it->haveReprojected)
                reprojectorSeed(frame, *it, it);   
        }
        for(size_t i=0; i<grid_.seeds.size(); ++i)
        {
            if(reprojectorSeeds(*grid_.seeds.at(grid_.cell_order[i]), frame))
            {
                ++n_matches_;
            }
            if(n_matches_ >= (size_t) Config::maxFts())
                break;     
        }
    }
    VIHSO_STOP_TIMER("feature_align");
}
void Reprojector::reprojectNeighbors(FramePtr curFrame, FramePtr refKF, list<pair<FramePtr,double>> &lNeighborKFs)
{
    resetGrid();
    std::vector< pair<Vector2d, Point*> > allPixelToDistribute;
    if(!mpMap->mCandidatesManager.mlTemporaryPoints.empty())    
    {
        DepthFilter::lock_t lock(depth_filter_->seeds_mut_);
        size_t n = 0;
        auto ite = mpMap->mCandidatesManager.mlTemporaryPoints.begin();
        while(ite != mpMap->mCandidatesManager.mlTemporaryPoints.end())
        {
            if(ite->first->seedStates_ == 0) 
            {
                ite++;
                continue;
            } 
            unique_lock<mutex> lock(mpMap->mCandidatesManager.mMutexCandidates);
            mpMap->safeDeleteTempPoint(*ite);
            ite = mpMap->mCandidatesManager.mlTemporaryPoints.erase(ite);
            n++;
        }
        sum_seed_ -= n;
    }
    std::vector< std::pair<FramePtr, std::size_t> > overlap_kfs;
	overlap_kfs.reserve(options_.max_n_kfs);
    size_t nCovisibilityGraph = 0;
	std::vector<vihso::Frame*> vRefCovs = refKF->GetBestCovisibilityKeyFrames(5);
    for(vector<Frame*>::iterator it = vRefCovs.begin(); it != vRefCovs.end(); ++it)
    {
        int KFid = (*it)->id_;
        FramePtr pRefCovKF = NULL;   
        if(!mpMap->getKeyframeById(KFid, pRefCovKF))			continue;
        if(pRefCovKF->lastReprojectFrameId_ == curFrame->id_)	continue;
        pRefCovKF->lastReprojectFrameId_ = curFrame->id_;
        overlap_kfs.push_back(pair<FramePtr,size_t>(pRefCovKF,0));   
        for(auto ite = pRefCovKF->fts_.begin(); ite != pRefCovKF->fts_.end(); ++ite)
        {
            if((*ite)->point == NULL)									continue;
            if((*ite)->point->type_ == Point::TYPE_TEMPORARY)			continue;
            if((*ite)->point->last_projected_kf_id_ == curFrame->id_)	continue;
            (*ite)->point->last_projected_kf_id_ = curFrame->id_;
            if(reprojectPoint(curFrame, (*ite)->point, allPixelToDistribute))
                overlap_kfs.back().second++;    
        }
        nCovisibilityGraph++;
    }
    size_t n = 0;
    for(auto lit=lNeighborKFs.begin(), litend=lNeighborKFs.end(); lit!=litend && n<15; ++lit)
    {
        FramePtr pCovKF = lit->first;
        if(pCovKF->lastReprojectFrameId_ == curFrame->id_)	continue;
        pCovKF->lastReprojectFrameId_ = curFrame->id_;
        overlap_kfs.push_back(pair<FramePtr,size_t>(pCovKF,0));
        for(auto it_ftr=pCovKF->fts_.begin(), ite_ftr=pCovKF->fts_.end(); it_ftr!=ite_ftr; ++it_ftr)
        {
            if((*it_ftr)->point == NULL)                					continue;
            if((*it_ftr)->point->type_ == Point::TYPE_TEMPORARY)			continue;
            assert((*it_ftr)->point->type_ != Point::TYPE_DELETED);
            if((*it_ftr)->point->last_projected_kf_id_ == curFrame->id_)	continue;
            (*it_ftr)->point->last_projected_kf_id_ = curFrame->id_;
            if(reprojectPoint(curFrame, (*it_ftr)->point, allPixelToDistribute))
                overlap_kfs.back().second++;
        }
        ++n;
    }
    {
        unique_lock<mutex> lock(mpMap->mCandidatesManager.mMutexCandidates); 
        auto it=mpMap->mCandidatesManager.mlCandidatePoints.begin();
        while(it!=mpMap->mCandidatesManager.mlCandidatePoints.end())
        {
            if(!reprojectPoint(curFrame, it->first, allPixelToDistribute))
            {
                it->first->n_failed_reproj_ += 3;   
                if(it->first->n_failed_reproj_ > 30)
                {
                    mpMap->mCandidatesManager.deleteCandidate(*it);
                    it = mpMap->mCandidatesManager.mlCandidatePoints.erase(it);
                    continue;
                }
            }
            ++it;
        }
    }
    auto itk = mpMap->mCandidatesManager.mlTemporaryPoints.begin();
    while(itk != mpMap->mCandidatesManager.mlTemporaryPoints.end())
    {
        if(itk->first->isBad_){itk++;  continue;}
        assert(itk->first->last_projected_kf_id_ != curFrame->id_);
        itk->first->last_projected_kf_id_ = curFrame->id_;
        Point* tempPoint = itk->first;
        Feature* tempFeature = itk->second;
        double idist = tempPoint->GetIdist();
		Sophus::SE3 Twc = tempFeature->frame->GetPoseInverseSE3();
        Vector3d pose = Twc * (tempFeature->f*(1.0/idist));
        tempPoint->SetWorldPos(pose);
        if(!reprojectPoint(curFrame, itk->first, allPixelToDistribute))
        {
            itk->first->n_failed_reproj_ += 3;
            if(itk->first->n_failed_reproj_ > 30)
                itk->first->isBad_ = true;
        }
        itk++;
    }
	size_t nMaxFts = 300;	
	reprojectCellAll(allPixelToDistribute, curFrame);
    if(n_matches_ < 100)
    {
        DepthFilter::lock_t lock(depth_filter_->seeds_mut_);
        for(auto it = depth_filter_->seeds_.begin(); it != depth_filter_->seeds_.end(); ++it)
        {
            if(sqrt(it->sigma2) < it->z_range/options_.reproject_seed_thresh && !it->haveReprojected)
                reprojectorSeed(curFrame, *it, it);   
        }
        for(size_t i=0; i<grid_.seeds.size(); ++i)
        {
            if(reprojectorSeeds(*grid_.seeds.at(grid_.cell_order[i]), curFrame))
            {
                ++n_matches_;
            }
            if(n_matches_ >= nMaxFts)
                break;     
        }
    }
}
bool Reprojector::pointQualityComparator(Candidate& lhs, Candidate& rhs)
{
    if(lhs.pt->type_ != rhs.pt->type_)
        return (lhs.pt->type_ > rhs.pt->type_);
    else
    {
        if(lhs.pt->ftr_type_ > rhs.pt->ftr_type_)
            return true;
        return false;
    }
}
bool Reprojector::seedComparator(SeedCandidate& lhs, SeedCandidate& rhs)
{
  return (lhs.seed.sigma2 < rhs.seed.sigma2);
}
bool Reprojector::reprojectCell(Cell& cell, FramePtr frame, bool is_2nd, bool is_3rd)
{   
    if(cell.empty()) return false;
    if(!is_2nd) 
        cell.sort(boost::bind(&Reprojector::pointQualityComparator, _1, _2));
    Cell::iterator it=cell.begin();
    int succees = 0;
    while(it!=cell.end())
    {
        ++n_trials_;
        if(it->pt->type_ == Point::TYPE_DELETED)
        {
            it = cell.erase(it);
            continue;
        }
        if(!matcher_.findMatchDirect(*it->pt, *frame, it->px))
        {
            it->pt->n_failed_reproj_++;
            if(it->pt->type_ == Point::TYPE_UNKNOWN && it->pt->n_failed_reproj_ > 15)       
                mpMap->safeDeletePoint(it->pt);
            if(it->pt->type_ == Point::TYPE_CANDIDATE  && it->pt->n_failed_reproj_ > 30)    
                mpMap->mCandidatesManager.deleteCandidatePoint(it->pt);
            if(it->pt->type_ == Point::TYPE_TEMPORARY && it->pt->n_failed_reproj_ > 30)     
                it->pt->isBad_ = true;
            it = cell.erase(it);
            continue;
        }
        it->pt->n_succeeded_reproj_++;
        if(it->pt->type_ == Point::TYPE_UNKNOWN && it->pt->n_succeeded_reproj_ > 10)    
            it->pt->type_ = Point::TYPE_GOOD;
        Feature* new_feature = new Feature(frame.get(), it->px, matcher_.search_level_);
        frame->addFeature(new_feature);
        new_feature->point = it->pt;    
        if(matcher_.ref_ftr_->type == Feature::EDGELET)
        {
            new_feature->type = Feature::EDGELET;
            new_feature->grad = matcher_.A_cur_ref_*matcher_.ref_ftr_->grad;
            new_feature->grad.normalize();
        }
        else if(matcher_.ref_ftr_->type == Feature::GRADIENT)
            new_feature->type = Feature::GRADIENT;
        else
            new_feature->type = Feature::CORNER;
        it = cell.erase(it);
        if(!is_3rd)
            return true;    
        else
        {
            succees++;
            n_matches_++;
            if((succees >= 3) || (n_matches_ >= (size_t)Config::maxFts())) 
                return true;
        }
    }
    return false;
}
bool Reprojector::reprojectorSeeds(Sell& sell, FramePtr frame)
{
    sell.sort(boost::bind(&Reprojector::seedComparator, _1, _2));
    Sell::iterator it=sell.begin();
    while(it != sell.end())
    {
        if(matcher_.findMatchSeed(it->seed, *frame, it->px))
        {
            assert(it->seed.ftr->point == NULL);
            ++n_seeds_;
            sum_seed_++;
            Vector3d pHost = it->seed.ftr->f*(1./it->seed.mu);
			Sophus::SE3 Twc = it->seed.ftr->frame->GetPoseInverseSE3();
            Vector3d xyz_world(Twc * pHost);
            Point* point = new Point(xyz_world, it->seed.ftr);
            point->SetIdist(it->seed.mu);
            point->hostFeature_ = it->seed.ftr;
            point->type_ = Point::TYPE_TEMPORARY;
            if(it->seed.ftr->type == Feature::EDGELET)
                point->ftr_type_ = Point::FEATURE_EDGELET;
            else if(it->seed.ftr->type == Feature::CORNER)
                point->ftr_type_ = Point::FEATURE_CORNER;
            else
                point->ftr_type_ = Point::FEATURE_GRADIENT;
            Feature* new_feature = new Feature(frame.get(), it->px, matcher_.search_level_);      
            if(matcher_.ref_ftr_->type == Feature::EDGELET)
            {
                new_feature->type = Feature::EDGELET;
                new_feature->grad = matcher_.A_cur_ref_*matcher_.ref_ftr_->grad;
                new_feature->grad.normalize();
            }
            else if(matcher_.ref_ftr_->type == Feature::GRADIENT)
                new_feature->type = Feature::GRADIENT;
            else
                new_feature->type = Feature::CORNER;
            new_feature->point = point;
            frame->addFeature(new_feature);
            it->seed.haveReprojected = true;
            it->seed.temp = point;
            point->seedStates_ = 0;
            mpMap->mCandidatesManager.addPauseSeedPoint(point);
            it = sell.erase(it);
            return true;
        }
        else
            ++it;
    }
    return false;
}
bool Reprojector::reprojectPoint(FramePtr frame, Point* point, vector< pair<Vector2d, Point*> >& cells)
{
    Vector3d pHost = point->hostFeature_->f * (1.0/point->GetIdist());
	Sophus::SE3 Ttw = frame->GetPoseSE3();
	Sophus::SE3 Twh = point->hostFeature_->frame->GetPoseInverseSE3();
    Vector3d pTarget = (Ttw * Twh)*pHost;
    if(pTarget[2] < 0.00001) return false;    
    Vector2d px(frame->cam_->world2cam(pTarget));
    if(frame->cam_->isInFrame(px.cast<int>(), 8)) 
    {
        const int k = static_cast<int>(px[1]/grid_.cell_size)*grid_.grid_n_cols
                    + static_cast<int>(px[0]/grid_.cell_size);
        grid_.cells.at(k)->push_back(Candidate(point, px));
        cells.push_back(make_pair(px, point));
        nFeatures_++;
        return true;
    }
    return false;
}
bool Reprojector::reprojectorSeed(FramePtr frame, Seed& seed, list< Seed, aligned_allocator<Seed> >::iterator index)
{
	Sophus::SE3 Ttw = frame->GetPoseSE3();
	Sophus::SE3 Twh = seed.ftr->frame->GetPoseInverseSE3();
    SE3 Tth = Ttw * Twh;
    Vector3d pTarget = Tth*(1.0/seed.mu * seed.ftr->f);
    if(pTarget[2] < 0.001) return false;
    Vector2d px(frame->cam_->world2cam(pTarget));
    if(frame->cam_->isInFrame(px.cast<int>(), 8))
    {
        const int k = static_cast<int>(px[1]/grid_.cell_size)*grid_.grid_n_cols
                    + static_cast<int>(px[0]/grid_.cell_size);
        grid_.seeds.at(k)->push_back(SeedCandidate(seed, px, index));
        return true;
    }
    return false;
}
void Reprojector::reprojectCellAll(vector< pair<Vector2d, Point*> >& cell, FramePtr frame)
{
    if(cell.empty()) return;
    vector< pair<Vector2d, Point*> >::iterator it = cell.begin();
    while(it != cell.end())
    {
        ++n_trials_;
        if(it->second->type_ == Point::TYPE_DELETED)
        {
            it = cell.erase(it);
            continue;
        }
        if(!matcher_.findMatchDirect(*(it->second), *frame, it->first))
        {
            it->second->n_failed_reproj_++;
            if(it->second->type_ == Point::TYPE_UNKNOWN && it->second->n_failed_reproj_ > 15)
                mpMap->safeDeletePoint(it->second);
            if(it->second->type_ == Point::TYPE_CANDIDATE && it->second->n_failed_reproj_ > 30)
                mpMap->mCandidatesManager.deleteCandidatePoint(it->second);
            if(it->second->type_ == Point::TYPE_TEMPORARY && it->second->n_failed_reproj_ > 30)
                it->second->isBad_ = true;
            it = cell.erase(it);
            continue;
        }
        it->second->n_succeeded_reproj_++;
        if(it->second->type_ == Point::TYPE_UNKNOWN && it->second->n_succeeded_reproj_ > 10)
            it->second->type_ = Point::TYPE_GOOD;
        Feature* new_feature = new Feature(frame.get(), it->first, matcher_.search_level_);
        frame->addFeature(new_feature);
        new_feature->point = it->second;
        if(matcher_.ref_ftr_->type == Feature::EDGELET)
        {
            new_feature->type = Feature::EDGELET;
            new_feature->grad = matcher_.A_cur_ref_*matcher_.ref_ftr_->grad;
            new_feature->grad.normalize();
        }
        else if(matcher_.ref_ftr_->type == Feature::GRADIENT)
            new_feature->type = Feature::GRADIENT;
        else
            new_feature->type = Feature::CORNER;
        it = cell.erase(it);
        n_matches_++;
        if(n_matches_ >= (size_t)Config::maxFts()) return;
    }
}
} 
