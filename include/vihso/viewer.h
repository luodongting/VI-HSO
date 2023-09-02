#pragma once
#include <mutex>
#include <list>
#include <opencv2/core/core.hpp>
#include <sophus/se3.h>
#include <pangolin/pangolin.h>
#include <vihso/map.h>
namespace vihso 
{
    class Frame;
    class Map;
    class FrameHandlerMono; 
}
namespace hso {
class Viewer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Viewer(vihso::FrameHandlerMono* vo, vihso::Map* pMap);
    void run();
    bool CheckFinish();
    void DrawKeyFrames(const bool bDrawTrajactor, const bool bDrawKF);
    void DrawMapRegionPoints();
    void DrawMapSeeds();
    void DrawConnections(const bool bDrawGraph, const bool bDrawSpanningTree, const bool bDrawConstraints);
    cv::Mat DrawFrame();
    void DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText);
    void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M, pangolin::OpenGlMatrix &MOw);
    void DrawCurrentCamera(pangolin::OpenGlMatrix &Twc);
    void SuphusToOpenGLMatrix(Sophus::SE3 &T, pangolin::OpenGlMatrix &M);
private:
    vihso::FrameHandlerMono* _vo;
    vihso::Map* mpMap;
    std::mutex mMutexCurrentPose;
    std::mutex mMutexIMG;
    std::vector< Sophus::SE3 > _pos;
    std::vector< Sophus::SE3 > _KFpos;
    std::list<double> _qProcessTimes;   
    Sophus::SE3  _CurrentPoseTwc ;
    int _drawedframeID=0;
    void SetFinish();
    bool mbFinished;
    std::mutex mMutexFinish;
    bool mbStopTrack;
    float mKeyFrameSize;
    float mKeyFrameLineWidth;
    float mGraphLineWidth;
    float mPointSize;
    float mCameraSize;
    float mCameraLineWidth;
    float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;
};
}