#ifndef VIHSO_MAP_H_
#define VIHSO_MAP_H_

#include <queue>
#include <boost/noncopyable.hpp>
#include <boost/thread.hpp>
#include <vihso/global.h>

namespace vihso {

class Point;
class Feature;
class Seed;

class CandidatesManager
{
public:
	typedef pair<Point*, Feature*> CandidatePoint;	
	typedef list<CandidatePoint> LCandidatePoints;  

	LCandidatePoints mlCandidatePoints;				
	list<pair<Point*, Feature*>> mlTemporaryPoints;	
	list<Point*> mlTrashPoints;
	
	std::mutex mMutexCandidates;  

public:

	CandidatesManager();
	~CandidatesManager();

	void newCandidatePoint(Point* point, double depth_sigma2);
	void addPauseSeedPoint(Point* point);
	void addCandidatePointToFrame(FramePtr pFrame);
	void addCandidatePointToFrame(Frame* Frame);
	bool deleteCandidatePoint(Point* point);
	void removeFrameCandidates(FramePtr pFrame);
	void removeFrameCandidates(Frame* Frame);
	void reset();
	void deleteCandidate(CandidatePoint& c);
	void emptyTrash();
	void changeCandidatePosition(Frame* frame);
	void  ApplyScaledRotationCandidate(const Eigen::Matrix3d& Ryw_, Eigen::Vector3d& tyw_, const double s, const int nMapChange);

};

class Map : boost::noncopyable
{
public:

	list< FramePtr > keyframes_;          
	list< Point* > mlTrashPoints;
	CandidatesManager mCandidatesManager;

public:

	Map();
	~Map();

	void reset();
	void safeDeletePoint(Point* pt);
	void deletePoint(Point* pt);
	bool safeDeleteFrame(FramePtr pFrame);
	bool safeDeleteFrameID(int id);
	void removePtFrameRef(Frame* pFrame, Feature* ftr);
	void addKeyframe(FramePtr new_keyframe);
	void GetListCovisibleKeyFrames(FramePtr& pFrame, list< pair<FramePtr,double> >& CovisibleKFs);
	
	FramePtr GetClosestKeyframe(FramePtr& pFrame);
	FramePtr GetFurthestKeyframe(const Vector3d& pos);

	bool getKeyframeById(const int id, FramePtr& pFrame);
	void TransformMap(const Matrix3d& R, const Vector3d& t, const double& s);
	void emptyTrash();
	inline FramePtr lastKeyframe() { unique_lock<mutex> lock(mMutexMap); return keyframes_.back(); }  
	inline size_t size() { unique_lock<mutex> lock(mMutexMap); return keyframes_.size(); }      

	void safeDeleteTempPoint(pair<Point*, Feature*>& p);

	static int getpoint_counter_;  
	bool mbImuInitialized;  
	bool mbIMU_BA1;        
	bool mbIMU_BA2;      
	int mnMapChange;        
	int mnMapChangeNotified;

	std::mutex mMutexMap;  
	std::mutex mMutexMapUpdate;

	int nKFsInMap=0;	

	void SetImuInitialized();
	void SetIniertialBA1();
	void SetIniertialBA2();

	bool isImuInitialized();
	bool isIniertialBA1();
	bool isIniertialBA2();

	std::vector<FramePtr> GetAllKeyFrames();
	std::list<FramePtr> GetAllKeyFramesList();
	std::vector<FramePtr> GetAllKeyFramesFromSec();
	size_t GetNumOfKF();
	size_t GetNumOfMP();
	int GetMaxKFid();
	std::vector<Point*> GetAllMapPoints();
	list<pair<Point*,Feature*>> GetAllCandidates();

	void ApplyScaledRotation(const Eigen::Matrix3d &Rgw, const double s, const bool bScaledVel=false, const Eigen::Vector3d t=Eigen::Vector3d::Zero());
	void IncreaseChangeIndex();
	int GetMapChangeIndex();
	void SetLastMapChange(int currentChangeId);
	int GetLastMapChange();

};

namespace map_debug 
{
	void mapStatistics(Map* map);
	void mapValidation(Map* map, int id);
	void frameValidation(Frame* frame, int id);
	void pointValidation(Point* point, int id);

} // namespace map_debug


} // namespace vihso

#endif // VIHSO_MAP_H_
