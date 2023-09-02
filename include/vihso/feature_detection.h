#ifndef VIHSO_FEATURE_DETECTION_H_
#define VIHSO_FEATURE_DETECTION_H_

#include <vihso/global.h>
#include <vihso/frame.h>
#include <fast/fast.h>

#include <string>

namespace vihso {

namespace feature_detection 
{
enum FeatureSpecies
{
    kCornerHigh,  
    kEdgeLet,     
    kGrad,
    kOccur
};

struct KeyPoint
{   
    float x;  
    float y;
    float response;
    int level;  
    FeatureSpecies species; 
    int gx;
    int gy;

    KeyPoint(float _x, float _y, float _response, int _level, FeatureSpecies _species): 
        x(_x), y(_y), response(_response), level(_level), species(_species)
    {
    }

    KeyPoint(): x(0), y(0), response(0), level(0), species(kCornerHigh)
    {
    }
};

class ExtractorNode
{
public:

    ExtractorNode():bNoMore(false){}
    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
    {
        const int halfX = ceil(static_cast<float>(UR.x-UL.x)/2);
        const int halfY = ceil(static_cast<float>(BR.y-UL.y)/2);

        n1.UL = UL;
        n1.UR = cv::Point2i(UL.x+halfX,UL.y);
        n1.BL = cv::Point2i(UL.x,UL.y+halfY);
        n1.BR = cv::Point2i(UL.x+halfX,UL.y+halfY);
        n1.vKeys.reserve(vKeys.size());

        n2.UL = n1.UR;
        n2.UR = UR;
        n2.BL = n1.BR;
        n2.BR = cv::Point2i(UR.x,UL.y+halfY);
        n2.vKeys.reserve(vKeys.size());

        n3.UL = n1.BL;
        n3.UR = n1.BR;
        n3.BL = BL;
        n3.BR = cv::Point2i(n1.BR.x,BL.y);
        n3.vKeys.reserve(vKeys.size());

        n4.UL = n3.UR;
        n4.UR = n2.BR;
        n4.BL = n3.BR;
        n4.BR = BR;
        n4.vKeys.reserve(vKeys.size());

        for(size_t i=0;i<vKeys.size();i++)
        {
            const float u = vKeys[i].x;
            const float v = vKeys[i].y;
            if(u<n1.UR.x)
            {
                if(v<n1.BR.y)
                    n1.vKeys.push_back(vKeys[i]);
                else
                    n3.vKeys.push_back(vKeys[i]);
            }
            else if(v<n1.BR.y)
                n2.vKeys.push_back(vKeys[i]);
            else
                n4.vKeys.push_back(vKeys[i]);
        }

        if(n1.vKeys.size()==1)
            n1.bNoMore = true;
        if(n2.vKeys.size()==1)
            n2.bNoMore = true;
        if(n3.vKeys.size()==1)
            n3.bNoMore = true;
        if(n4.vKeys.size()==1)
            n4.bNoMore = true;
    }

    std::vector<KeyPoint> vKeys;  
    cv::Point2i UL, UR, BL, BR;   
    bool bNoMore;                 
    std::list<ExtractorNode>::iterator lit;
};


class FeatureExtractor
{
public:

    FeatureExtractor(const int width, const int height, const int cellSize, const int levels, bool isInit=false);       
    void detect(Frame* frame, const float initThresh, const float minThresh, Features& fts, int nSetNumFts=0, Frame* last_frame=NULL);
    void resetGrid(); 
    void setGridOccpuancy(const Vector2d& px, Feature* occurFeature); 
    void setExistingFeatures(const Features& fts);

    inline int getCellIndex(int x, int y, int level) 
    {
        return static_cast<int>(y/vGrids_[level]*vGridCols_[level] + x/vGridRows_[level]);
    }

protected:

    void fastDetect(const ImgPyr& img_pyr);                           
    void fastDetectMT(const ImgPyr& img_pyr);                         
    void fastDetectST(const cv::Mat& imageLevel, const int Level);    

    void edgeLetDetectMT(const ImgPyr& img_pyr);                      
    void edgeLetDetectST(const cv::Mat& imageLevel, const int Level);

	void orbDetectMT(const ImgPyr& img_pyr);                      
    void orbDetectST(const cv::Mat& imageLevel, const int Level); 

    void gradDetect(const ImgPyr& img_pyr);                          
    void gradDetectMT(const ImgPyr& img_pyr);                        
    void gradDetectST(const cv::Mat& imageLevel, const int Level);   

    void fillingHole(const cv::Mat& imageLevel, const int Level);    

    vector<KeyPoint> computeKeyPointsOctTree(
        const vector<KeyPoint>& toDistributeKeys, 
        const int &minX, const int &maxX, const int &minY, const int &maxY, const int &level);

    void findEpiHole();
    bool edgeletFilter(int u_level, int v_level, short gx, short gy, int level, double& angle); 


protected:

    Frame* frame_;  
    Frame* m_last_frame = NULL; 

    int width_;     
    int height_;    
    std::vector<int> vecWidth_;  
    std::vector<int> vecHeight_; 
    int cellSize_;  
    int nLevels_;   
    int nFeatures_;     
    int extFeatures_;
    int needFeatures_;  
    int initThresh_;  
    int minThresh_;   
    int nCols_; 
    int nRows_; 
    bool m_egde_filter;
    Vector2d epi_hole;  

    std::vector<KeyPoint> allFeturesToDistribute_;      
    std::vector<vector<KeyPoint> > featurePerLevel_;    
    static const int gridSize_ = 8; 
    std::vector<int> vGrids_;       
    std::vector<int> vGridCols_;   
    std::vector<int> vGridRows_;   
    std::vector<vector<bool> > haveFeatures_; 
    std::vector<KeyPoint> resultFeatures_;  
    bool isInit_; 

    int n_corners, n_edgeLet;           
    double timer_corners,timer_edgeLet; 

};


} // namespace feature_detection
} // namespace vihso

#endif // VIHSO_FEATURE_DETECTION_H_
