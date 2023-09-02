#ifndef VIHSO_REPROJECTION_H_
#define VIHSO_REPROJECTION_H_
#include <vihso/global.h>
#include <vihso/matcher.h>
#include "vihso/camera.h"
namespace vihso {
class Map;
class Point;
class DepthFilter;
struct Seed;
class Reprojector
{
 public:
  	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	struct Options  
	{
		size_t max_n_kfs; 
		bool find_match_direct;
		bool reproject_unconverged_seeds;
		float reproject_seed_thresh;
		Options()
		: max_n_kfs(10),
		find_match_direct(true),
		reproject_unconverged_seeds(true),
		reproject_seed_thresh(86)
		{}
	} options_;
	Reprojector(hso::AbstractCamera* cam, Map* pMap);
	~Reprojector();
	int caculateGridSize(const int wight, const int height, const int N);
	void reprojectMap(FramePtr frame, std::vector< std::pair<FramePtr,std::size_t> >& overlap_kfs);
	void reprojectNeighbors(FramePtr curFrame, FramePtr refKF, list<pair<FramePtr,double>> &lNeighborKFs);
 private:
	struct Candidate  
	{
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		Point* pt;       
		Vector2d px;     
		Candidate(Point* pt, Vector2d& px) :  pt(pt), px(px) 
		{}
	};
	typedef std::list<Candidate > Cell;       
	typedef std::vector<Cell*> CandidateGrid; 
	struct SeedCandidate
	{
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		Seed& seed;
		Vector2d px;
		list< Seed, aligned_allocator<Seed> >::iterator index;
		SeedCandidate(Seed& _seed, Vector2d _uv, 
		list< Seed, aligned_allocator<Seed> >::iterator _i):seed(_seed), px(_uv), index(_i)
		{}
	};
	typedef std::list<SeedCandidate> Sell;    
	typedef std::vector<Sell*> SeedGrid;      
	struct Grid 
	{
		CandidateGrid cells;    
		SeedGrid seeds;         
		vector<int> cell_order; 
		int cell_size;    
		int grid_n_cols;  
		int grid_n_rows;  
		int cell_size_w;
		int cell_size_h;
	};
	static bool pointQualityComparator(Candidate& lhs, Candidate& rhs); 
	static bool seedComparator(SeedCandidate& lhs, SeedCandidate& rhs);
	void initializeGrid(hso::AbstractCamera* cam);
	void resetGrid();
	bool reprojectCell(Cell& cell, FramePtr frame, bool is_2nd = false, bool is_3rd = false); 
	bool reprojectorSeeds(Sell& sell, FramePtr frame);
	bool reprojectPoint(FramePtr frame, Point* point, vector< pair<Vector2d, Point*> >& cells); 
	bool reprojectorSeed(FramePtr frame, Seed& seed, list< Seed, aligned_allocator<Seed> >::iterator index);
	void reprojectCellAll(vector< pair<Vector2d, Point*> >& cell, FramePtr frame);
 public:
	size_t n_matches_;  
	size_t n_trials_;   
	size_t n_seeds_;    
	size_t n_filters_;
	DepthFilter* depth_filter_;
 private:
	Grid grid_;
	Matcher matcher_;
	Map* mpMap;
	size_t sum_seed_; 	
	size_t sum_temp_;
	size_t nFeatures_;	
};
} 
#endif 
