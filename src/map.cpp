#include <set>
#include <vihso/map.h>
#include <vihso/point.h>
#include <vihso/frame.h>
#include <vihso/feature.h>
#include <boost/bind.hpp>

namespace vihso {

int Map::getpoint_counter_ = 0;

Map::Map(): mbImuInitialized(false), mbIMU_BA1(false), mbIMU_BA2(false), mnMapChange(0),mnMapChangeNotified(0)
{
}

Map::~Map()
{
	reset();
}

void Map::reset()
{
	mCandidatesManager.reset();
	keyframes_.clear();
	emptyTrash();
	nKFsInMap=0;
}

bool Map::safeDeleteFrame(FramePtr frame)
{
	bool found = false;
	for(auto it=keyframes_.begin(), ite=keyframes_.end(); it!=ite; ++it)
	{
		if(*it == frame)
		{
			std::for_each((*it)->fts_.begin(), (*it)->fts_.end(), [&](Feature* ftr){
			removePtFrameRef(it->get(), ftr);
			});
			keyframes_.erase(it);
			found = true;
			break;
		}
	}

	mCandidatesManager.removeFrameCandidates(frame);

	if(found)
	return true;

	VIHSO_ERROR_STREAM("Tried to delete Keyframe in map which was not there.");
	return false;
}

bool Map::safeDeleteFrameID(int id)
{
	bool found = false;
	FramePtr delete_frame = NULL;
	for(auto it=keyframes_.begin(), ite=keyframes_.end(); it!=ite; ++it)
	{
		if((*it)->id_ == id)
		{
			delete_frame = *it;

			std::for_each((*it)->fts_.begin(), (*it)->fts_.end(), [&](Feature* ftr){
			removePtFrameRef(it->get(), ftr);
			});
			keyframes_.erase(it);
			found = true;
			break;
		}
	}

	mCandidatesManager.removeFrameCandidates(delete_frame);

	if(found)
	return true;

	VIHSO_ERROR_STREAM("Tried to delete Keyframe in map which was not there.");
	return false;
}

void Map::removePtFrameRef(Frame* frame, Feature* ftr)
{
	if(ftr->point == NULL)
		return; 

	Point* pt = ftr->point;
	ftr->point = NULL;
	if(pt->GetNumofObs() <= 2)
	{
		safeDeletePoint(pt);
		return;
	}
	pt->deleteFrameRef(frame);  
	frame->removeKeyPoint(ftr); 
}

void Map::safeDeletePoint(Point* pt)
{
  list<Feature*> observations = pt->GetObservations();
  std::for_each(observations.begin(), observations.end(), 
  				[&](Feature* ftr)
				{
    				ftr->point=NULL;
    				ftr->frame->removeKeyPoint(ftr);
  				});
  pt->ClearObservations();
  deletePoint(pt);
}

void Map::safeDeleteTempPoint(pair<Point*, Feature*>& p)
{
    Point* pTempMP = p.first;
    Feature* pft = p.second;
    if(pTempMP->seedStates_ == -1)
    {
        if(pTempMP->isBad_)
            safeDeletePoint(pTempMP);
        else
        {
            assert(pTempMP->hostFeature_ == pft);

            pTempMP->UpdatePose();
            
            if(pTempMP->GetNumofObs() == 1)
            {
                pTempMP->type_ = Point::TYPE_CANDIDATE;
                pTempMP->n_failed_reproj_ = 0;
                pTempMP->n_succeeded_reproj_ = 0;
                list<Feature*> observations = pTempMP->GetObservations();
                mCandidatesManager.mlCandidatePoints.push_back(CandidatesManager::CandidatePoint(pTempMP, observations.front()));
            }
            else
            {
                pTempMP->type_ = Point::TYPE_UNKNOWN;
                pTempMP->n_failed_reproj_ = 0;
                pTempMP->n_succeeded_reproj_ = 0;
                pft->frame->addFeature(pft);
            }
        }
    }
    else 
    {
        list<Feature*> observations = pTempMP->GetObservations();
        assert(pTempMP->seedStates_ == 1 && 
               observations.back()->point->id_ == pft->point->id_);

        for(auto it = observations.begin(); it != observations.end(); ++it) 
        {
          if((*it)->point->id_ != pft->point->id_)
            {
                (*it)->point=NULL;
                (*it)->frame->removeKeyPoint(*it);
            }
        }    

        pTempMP->ClearObservations();
        deletePoint(pTempMP);
    }
}

void Map::deletePoint(Point* pt)
{
	pt->type_ = Point::TYPE_DELETED;
	mlTrashPoints.push_back(pt);
}

void Map::addKeyframe(FramePtr new_keyframe)
{
  	keyframes_.push_back(new_keyframe);
	nKFsInMap++;
}


void Map::GetListCovisibleKeyFrames(FramePtr& pFrame, list< pair<FramePtr,double> >& CovisibleKFs)
{
    for(auto kf : keyframes_)
    {   
        for(auto keypoint : kf->key_pts_)
        {   
            if(keypoint == nullptr) continue;
  
            assert(keypoint->point != NULL);
            if(pFrame->isVisible(keypoint->point->GetWorldPos()))
            {
				double dist = (pFrame->GetTranslation()-kf->GetTranslation()).norm();
                CovisibleKFs.push_back( std::make_pair(kf, dist)); 
                break; 
            }
        }
    }
}

FramePtr Map::GetClosestKeyframe(FramePtr& pFrame)
{
	list< pair<FramePtr,double> > close_kfs;
	GetListCovisibleKeyFrames(pFrame, close_kfs);
	if(close_kfs.empty())
	{
		return nullptr;
	}

	close_kfs.sort(boost::bind(&std::pair<FramePtr, double>::second, _1) <
					boost::bind(&std::pair<FramePtr, double>::second, _2));

	if(close_kfs.front().first != pFrame)
		return close_kfs.front().first;
	close_kfs.pop_front();
	return close_kfs.front().first;
}

FramePtr Map::GetFurthestKeyframe(const Vector3d& pos)
{
	FramePtr furthest_kf;
	double maxdist = 0.0;
	for(auto it=keyframes_.begin(), ite=keyframes_.end(); it!=ite; ++it)
	{
		double dist = ((*it)->GetCameraCenter()-pos).norm();
		if(dist > maxdist) {
		maxdist = dist;
		furthest_kf = *it;
		}
	}
	return furthest_kf;
}

bool Map::getKeyframeById(const int id, FramePtr& frame)
{
    bool found = false;
    for(auto it=keyframes_.begin(), ite=keyframes_.end(); it!=ite; ++it)
        if((*it)->id_ == id) 
        {
            found = true;
            frame = *it;
            break;
        }
    return found;
}

void Map::TransformMap(const Matrix3d& R, const Vector3d& t, const double& s)
{
	for(auto it=keyframes_.begin(), ite=keyframes_.end(); it!=ite; ++it)
	{
		Vector3d pos = s*R*(*it)->GetCameraCenter() + t;
		Matrix3d Rwc = (*it)->GetRotation().inverse();
		Matrix3d rot = R*Rwc;
		(*it)->SetPose(SE3(rot, pos).inverse());

		for(auto ftr=(*it)->fts_.begin(); ftr!=(*it)->fts_.end(); ++ftr)
		{
			if((*ftr)->point == NULL)
				continue;
			if((*ftr)->point->last_published_ts_ == -1000)
				continue;
			(*ftr)->point->last_published_ts_ = -1000;
			const Vector3d pose = s*R*(*ftr)->point->GetWorldPos() + t;
			(*ftr)->point->SetWorldPos(pose);
		}
	}
}

void Map::emptyTrash()
{
	std::for_each(	mlTrashPoints.begin(), 
					mlTrashPoints.end(), 
					[&](Point*& pt)
					{pt=NULL;});
	mlTrashPoints.clear();
	mCandidatesManager.emptyTrash();
}

void Map::SetImuInitialized()
{
    mbImuInitialized = true;
}
bool Map::isImuInitialized()
{
    return mbImuInitialized;
}
void Map::SetIniertialBA1()
{
    mbIMU_BA1 = true;
}
void Map::SetIniertialBA2()
{
    mbIMU_BA2 = true;
}
bool Map::isIniertialBA1()
{
    return mbIMU_BA1;
}
bool Map::isIniertialBA2()
{
    return mbIMU_BA2;
}

std::vector<FramePtr> Map::GetAllKeyFrames()
{ 
	unique_lock<mutex> lock(mMutexMap);
	std::vector<FramePtr> vKFs;
	for(list<vihso::FramePtr>::iterator it=keyframes_.begin(),itend=keyframes_.end(); it!=itend; it++)
	{
		vKFs.push_back(*it);
	}
	return vKFs;
}
std::list<FramePtr> Map::GetAllKeyFramesList()
{
	unique_lock<mutex> lock(mMutexMap);
	return keyframes_;
}
std::vector<FramePtr> Map::GetAllKeyFramesFromSec()
{ 
	unique_lock<mutex> lock(mMutexMap);
	list<vihso::FramePtr>::iterator pKF = keyframes_.begin();
	pKF++;
	return std::vector<FramePtr>(pKF,keyframes_.end());
}
int Map::GetMaxKFid()
{
	unique_lock<mutex> lock(mMutexMap);
	FramePtr lastKF = keyframes_.back();
	return lastKF->GetKeyFrameID();
}
size_t Map::GetNumOfKF()
{
	unique_lock<mutex> lock(mMutexMap);
	return keyframes_.size();
}
size_t Map::GetNumOfMP()
{
	unique_lock<mutex> lock(mMutexMap);
	std::set<vihso::Point*> sMPs;
	for(list<vihso::FramePtr>::iterator it_kf = keyframes_.begin(); it_kf != keyframes_.end(); ++it_kf)
	{
		for(Features::iterator it_pt=(*it_kf)->fts_.begin(); it_pt!=(*it_kf)->fts_.end(); ++it_pt)
		{
		if((*it_pt)->point == NULL) continue;
		sMPs.insert((*it_pt)->point);
		}
	}
	return sMPs.size();
}

void Map::ApplyScaledRotation(const Eigen::Matrix3d &Rgw, const double s, const bool bScaledVel, const Eigen::Vector3d t)
{
  
	unique_lock<mutex> lock(mMutexMap);

	Eigen::Matrix4d Txw = Eigen::Matrix4d::Identity();
	Txw.block<3,3>(0,0) = Rgw;

	Eigen::Matrix4d Tyx = Eigen::Matrix4d::Identity();  
	Eigen::Matrix4d Tyw = Tyx*Txw;  
	Tyw.block<3,1>(0,3) = Tyw.block<3,1>(0,3) + t;
	Eigen::Matrix3d Ryw = Tyw.block<3,3>(0,0);
	Eigen::Vector3d tyw = Tyw.block<3,1>(0,3);


	for(list<FramePtr>::iterator sit=keyframes_.begin(); sit!=keyframes_.end(); sit++)
	{
		FramePtr pKF = *sit;

		Eigen::Matrix4d Twc = pKF->GetPoseInverse();  
		Twc.block<3,1>(0,3)*=s;                       
		Eigen::Matrix4d Tyc = Tyw*Twc;                
		Eigen::Matrix4d Tcy = Tyc.inverse();          

		pKF->SetPose(Sophus::SE3(Tcy.block<3,3>(0,0), Tcy.block<3,1>(0,3)));

		Eigen::Vector3d Vw = pKF->GetVelocity();
		if(!bScaledVel)
		pKF->SetVelocity(Ryw*Vw);
		else
		pKF->SetVelocity(Ryw*Vw*s);

		if(!pKF->mlRelativeFrame.empty())
		{
		for(std::list<std::pair<double, Eigen::Matrix4d>>::iterator lit=pKF->mlRelativeFrame.begin();
			lit!=pKF->mlRelativeFrame.end(); lit++)
		{
			(*lit).second.block<3,1>(0,3) *=s;
		}
		}
	
		for(list<Feature*>::iterator itfeat = pKF->fts_.begin(); itfeat != pKF->fts_.end(); itfeat++)
		{
			if((*itfeat)->point == nullptr)
				continue;
			vihso::Point* pPoint = (*itfeat)->point;

			if(pPoint->nApplyScaled == mnMapChange)
				continue;
			else
			{
				pPoint->UpdatePoseScale(s);
				pPoint->nApplyScaled = mnMapChange;
			}
		}
	}
	mCandidatesManager.ApplyScaledRotationCandidate(Ryw, tyw, s, mnMapChange);
}

void  CandidatesManager::ApplyScaledRotationCandidate(const Eigen::Matrix3d& Ryw, Eigen::Vector3d& tyw, const double s, const int nMapChange)
{
	unique_lock<mutex> lock(mMutexCandidates);

	for(list<CandidatePoint>::iterator itC = mlCandidatePoints.begin(); itC != mlCandidatePoints.end(); itC++)
	{
		if((*itC).first == nullptr)
		continue;

		vihso::Point* pPoint = (*itC).first;
		if(pPoint->nApplyScaled != nMapChange)
		{
			pPoint->UpdatePoseScale(s);
			pPoint->nApplyScaled = nMapChange;
		}  
		else
		continue;
	}

	for(list< pair<Point*, Feature*> >::iterator itC = mlTemporaryPoints.begin(); itC != mlTemporaryPoints.end(); itC++)
	{
		if((*itC).first == nullptr)
		continue;
		
		vihso::Point* pPoint = (*itC).first;
		if(pPoint->nApplyScaled != nMapChange)
		{
		pPoint->UpdatePoseScale(s);
		pPoint->nApplyScaled = nMapChange;
		}  
		else
		continue;
	}
}

std::vector<Point*> Map::GetAllMapPoints()
{
	unique_lock<mutex> lock(mMutexMap);
	getpoint_counter_++;
	std::vector<Point*> vMapPoints;
	vMapPoints.reserve(keyframes_.size()*200);

	for(list<vihso::FramePtr>::iterator itf=keyframes_.begin(); itf!=keyframes_.end(); itf++)  
	{
		FramePtr pKF = (*itf);
		for(Features::iterator itfeat=pKF->fts_.begin(); itfeat!=pKF->fts_.end(); itfeat++)
		{
		if((*itfeat)->point == nullptr)
			continue;
		vihso::Point* pMP = (*itfeat)->point;

		if(pMP->nGetAllMapPoints == getpoint_counter_)  
			continue;
		else    
		{
			vMapPoints.push_back(pMP);
			pMP->nGetAllMapPoints = getpoint_counter_;
		}
		}
	}
	return vMapPoints;
}

list<pair<Point*,Feature*>> Map::GetAllCandidates()
{
 	unique_lock<mutex> lock(mCandidatesManager.mMutexCandidates);
    return mCandidatesManager.mlCandidatePoints;
}

void Map::IncreaseChangeIndex()
{
    mnMapChange++;
}
int Map::GetMapChangeIndex()
{
    return mnMapChange;
}
void Map::SetLastMapChange(int currentChangeId)
{
    mnMapChangeNotified = currentChangeId;
}
int Map::GetLastMapChange()
{
    return mnMapChangeNotified;
}

CandidatesManager::CandidatesManager()
{}

CandidatesManager::~CandidatesManager()
{
  	reset();
}

void CandidatesManager::newCandidatePoint(Point* point, double depth_sigma2)
{
	point->type_ = Point::TYPE_CANDIDATE;
	unique_lock<mutex> lock(mMutexCandidates);

	list<Feature*> observations = point->GetObservations();
	mlCandidatePoints.push_back(CandidatePoint(point, observations.front()));
}

void CandidatesManager::addPauseSeedPoint(Point* point)
{
	assert(point->type_ == Point::TYPE_TEMPORARY);
	unique_lock<mutex> lock(mMutexCandidates);

	list<Feature*> observations = point->GetObservations();
	assert(point->hostFeature_ == observations.front());
	mlTemporaryPoints.push_back(make_pair(point, observations.front()));
}

void CandidatesManager::addCandidatePointToFrame(FramePtr pFrame)
{
  	addCandidatePointToFrame(pFrame.get());
}
void CandidatesManager::addCandidatePointToFrame(Frame* frame)
{
    unique_lock<mutex> lock(mMutexCandidates);
    LCandidatePoints::iterator it=mlCandidatePoints.begin();
    while(it != mlCandidatePoints.end()) 
    {
        list<Feature*> observations = it->first->GetObservations();
        if(observations.front()->frame == frame)
        {
            assert(it->first->GetNumofObs() == 2);
            it->first->type_ = Point::TYPE_UNKNOWN; 
            it->first->n_failed_reproj_ = 0;
            it->second->frame->addFeature(it->second);  
            it = mlCandidatePoints.erase(it);
        }
        else
            ++it;
    }
}

bool CandidatesManager::deleteCandidatePoint(Point* point)
{
    unique_lock<mutex> lock(mMutexCandidates);
    for(auto it=mlCandidatePoints.begin(), ite=mlCandidatePoints.end(); it!=ite; ++it)
    {
        if(it->first == point)
        {
            deleteCandidate(*it);
            mlCandidatePoints.erase(it);
            return true;
        }
    }
    return false;
}

void CandidatesManager::changeCandidatePosition(Frame* frame)
{
    unique_lock<mutex> lock(mMutexCandidates);
    
    for(LCandidatePoints::iterator it = mlCandidatePoints.begin(); it != mlCandidatePoints.end(); ++it)
    {
        Point* point = it->first;
        Feature* ft = it->second;
		assert(point != NULL);
        assert(point->type_ == Point::TYPE_CANDIDATE );
		assert(point->GetNumofObs() == 1 );
		assert(point->vPoint_ == NULL);

        if(ft->frame->id_ == frame->id_)
        {
          const double idist = point->GetIdist();
		  const Sophus::SE3 Twc = frame->GetPoseInverseSE3();
          const Vector3d pose = Twc * (ft->f * (1.0/idist));
          point->SetWorldPos(pose);
        }
    }
}

void CandidatesManager::removeFrameCandidates(FramePtr pFrame)
{
  	removeFrameCandidates(pFrame.get());
}
void CandidatesManager::removeFrameCandidates(Frame* frame)
{
	unique_lock<mutex> lock(mMutexCandidates);
	auto it=mlCandidatePoints.begin();
	while(it!=mlCandidatePoints.end())
	{
		if(it->second->frame == frame)
		{
			deleteCandidate(*it);
			it = mlCandidatePoints.erase(it);
		}
		else
			++it;
	}
}

void CandidatesManager::reset()
{
	unique_lock<mutex> lock(mMutexCandidates);
	std::for_each(mlCandidatePoints.begin(), mlCandidatePoints.end(), [&](CandidatePoint& c){
		delete c.first;
		delete c.second;
	});
	mlCandidatePoints.clear();
}

void CandidatesManager::deleteCandidate(CandidatePoint& c)
{
	delete c.second; c.second=NULL;
	c.first->type_ = Point::TYPE_DELETED;
	mlTrashPoints.push_back(c.first);
}

void CandidatesManager::emptyTrash()
{
	std::for_each(mlTrashPoints.begin(), mlTrashPoints.end(), [&](Point*& p)
	{
		p=NULL;
	});
	mlTrashPoints.clear();
}

namespace map_debug {

void mapValidation(Map* map, int id)
{
	for(auto it=map->keyframes_.begin(); it!=map->keyframes_.end(); ++it)
		frameValidation(it->get(), id);
}

void frameValidation(Frame* frame, int id)
{
	for(auto it = frame->fts_.begin(); it!=frame->fts_.end(); ++it)
	{
		if((*it)->point==NULL)
		continue;

		if((*it)->point->type_ == Point::TYPE_DELETED)
		printf("ERROR DataValidation %i: Referenced point was deleted.\n", id);

		if(!(*it)->point->findFrameRef(frame))
		printf("ERROR DataValidation %i: Frame has reference but point does not have a reference back.\n", id);

		pointValidation((*it)->point, id);
	}
	for(auto it=frame->key_pts_.begin(); it!=frame->key_pts_.end(); ++it)
		if(*it != NULL)
		if((*it)->point == NULL)
			printf("ERROR DataValidation %i: KeyPoints not correct!\n", id);
}

void pointValidation(Point* point, int id)
{
	list<Feature*> observations = point->GetObservations();
	for(auto it=observations.begin(); it!=observations.end(); ++it)
	{
		bool found=false;
		for(auto it_ftr=(*it)->frame->fts_.begin(); it_ftr!=(*it)->frame->fts_.end(); ++it_ftr)
		if((*it_ftr)->point == point) {
		found=true; break;
		}
		if(!found)
		printf("ERROR DataValidation %i: Point %i has inconsistent reference in frame %i, is candidate = %i\n", id, point->id_, (*it)->frame->id_, (int) point->type_);
	}
}

void mapStatistics(Map* map)
{
	size_t n_pt_obs(0);
	for(auto it=map->keyframes_.begin(); it!=map->keyframes_.end(); ++it)
		n_pt_obs += (*it)->nObs();
	printf("\n\nMap Statistics: Frame avg. point obs = %f\n", (float) n_pt_obs/map->size());

	size_t n_frame_obs(0);
	size_t n_pts(0);
	std::set<Point*> points;
	for(auto it=map->keyframes_.begin(); it!=map->keyframes_.end(); ++it)
	{
		for(auto ftr=(*it)->fts_.begin(); ftr!=(*it)->fts_.end(); ++ftr)
		{
		if((*ftr)->point == NULL)
			continue;
		if(points.insert((*ftr)->point).second) {
			++n_pts;
			n_frame_obs += (*ftr)->point->GetNumofObs();
		}
		}
	}
	printf("Map Statistics: Point avg. frame obs = %f\n\n", (float) n_frame_obs/n_pts);
}

} // namespace map_debug
} // namespace vihso
