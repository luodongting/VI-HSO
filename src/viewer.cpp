#include <vihso/viewer.h>
#include <vihso/frame_handler_mono.h>
#include <vihso/map.h>
#include <vihso/frame.h>
#include <vihso/feature.h>
#include <vihso/point.h>
#include <vihso/depth_filter.h>
#include <pangolin/gl/gltext.h>
#include <opencv4/opencv2/imgproc/types_c.h>
#include <vihso/vikit/colortable.h>
#include <vihso/config.h>
using namespace vihso;
namespace hso {
Viewer::Viewer(vihso::FrameHandlerMono* vo, vihso::Map* pMap): _vo(vo), mpMap(pMap)
{
    mbFinished = false;
    mViewpointX =  0;	
    mViewpointY = -4;	
    mViewpointZ =  -4;	
    mViewpointF =  500;
    mKeyFrameSize = 0.05;
    mKeyFrameLineWidth = 1.6;
	mGraphLineWidth = 1.4;
    mCameraSize = 0.08;
    mCameraLineWidth = 3.0;
    mPointSize = 3.0;
    mbStopTrack = false;
}
bool Viewer::CheckFinish()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    return mbFinished;
}
void Viewer::SetFinish()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    mbFinished = true;
}
void Viewer::DrawKeyFrames(const bool bDrawTrajactor, const bool bDrawKF)
{
    vihso::FramePtr lastframe = _vo->lastFrame();
    if(lastframe != NULL && lastframe->id_ != _drawedframeID)
    {
        _drawedframeID = lastframe->id_ ;
        _CurrentPoseTwc = lastframe->GetPoseInverseSE3();
        _pos.push_back(_CurrentPoseTwc);
    }
    const list<FramePtr> lpKFs = _vo->mpMap->GetAllKeyFramesList();
    _KFpos.clear();
    for(list<FramePtr>::const_iterator it=lpKFs.begin(),itend=lpKFs.end(); it!=itend; it++)
    {
        if(*it != NULL)
            _KFpos.push_back((*it)->GetPoseInverseSE3());
    }
    if(_vo->mbMapViewMonitoring)
    {
        _vo->mbMapViewMonitoring = false;
        vector<Sophus::SE3>().swap(_pos);
        _pos.assign(_KFpos.begin(), _KFpos.end());
    }
    if(_pos.empty()) 
        return;
    if(bDrawTrajactor)
    { 
        glColor3f(1.0,0.0,0.0);
        glLineWidth(mKeyFrameLineWidth);
        glBegin(GL_LINES);
        for(size_t i = 1; i<_pos.size();i++)
        {   
            Sophus::SE3 TCurrent = _pos[i];
            glVertex3d(TCurrent.translation()[0], TCurrent.translation()[1], TCurrent.translation()[2]);
            Sophus::SE3 TLast = _pos[i-1];
            glVertex3d(TLast.translation()[0], TLast.translation()[1], TLast.translation()[2]);
        }
        glEnd();
    }
    const float &w = mCameraSize*0.5;
    const float h = w*0.75;
    const float z = w*0.6;
    if(bDrawKF)
    {
        for(size_t i = 0; i<_KFpos.size();i++)
        {
            Sophus::SE3 Twc_ = _KFpos[i];
            pangolin::OpenGlMatrix Twc_KF;
            SuphusToOpenGLMatrix(Twc_, Twc_KF);
            glPushMatrix();
            #ifdef HAVE_GLES
                    glMultMatrixf(Twc.m);
            #else
                    glMultMatrixd(Twc_KF.m);
            #endif
            glLineWidth(mCameraLineWidth*0.5);
            glColor3f(0.0f,0.0f,1.0f);
            glBegin(GL_LINES);
            glVertex3f(0,0,0);
            glVertex3f(w,h,z);
            glVertex3f(0,0,0);
            glVertex3f(w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,h,z);
            glVertex3f(w,h,z);
            glVertex3f(w,-h,z);
            glVertex3f(-w,h,z);
            glVertex3f(-w,-h,z);
            glVertex3f(-w,h,z);
            glVertex3f(w,h,z);
            glVertex3f(-w,-h,z);
            glVertex3f(w,-h,z);
            glEnd();
            glPopMatrix();
        }
    }
}
void Viewer::DrawMapRegionPoints()
{
    glPointSize(mPointSize);
    glBegin(GL_POINTS);
	list<vihso::FramePtr> lKFs = _vo->mpMap->GetAllKeyFramesList();
    for(auto kf = lKFs.begin(); kf != lKFs.end(); ++kf)
	{
		for(auto& ft: (*kf)->fts_)
        {
            if(ft->point == NULL) continue;
            float color = float(ft->point->color_) / 255;
            if(color > 0.9) color = 0.9;
            glColor3f(color,color,color);
            Eigen::Vector3d Pw = ft->point->GetWorldPos();
            glVertex3f( Pw[0],Pw[1],Pw[2]);
        }
	}  
    glEnd();
    glPointSize(mPointSize+1.0f);
    glBegin(GL_POINTS);
    glColor3f(1.0,0.0,0.0);
    vihso::FramePtr lastframe = _vo->lastFrame();
    for(vihso::Features::iterator it=lastframe->fts_.begin(); it!=lastframe->fts_.end(); ++it)
    {
        if((*it)->point == NULL) continue;
        Eigen::Vector3d Pw = (*it)->point->GetWorldPos();
        glVertex3f( Pw[0],Pw[1],Pw[2]);
    }
    glEnd();
}
void Viewer::DrawConnections(const bool bDrawGraph, const bool bDrawSpanningTree, const bool bDrawConstraints)
{
    const list<FramePtr> lpKFs = _vo->mpMap->GetAllKeyFramesList();
    if(bDrawGraph)
    {
        glLineWidth(mGraphLineWidth);
        glColor4f(0.0f,1.0f,0.0f,0.6f);
        glBegin(GL_LINES);
		for(list<FramePtr>::const_iterator it=lpKFs.begin(),itend=lpKFs.end(); it!=itend; it++)
        {
			FramePtr pKF = *it;
            const vector<Frame*> vCovKFs = pKF->GetCovisiblesByWeight(50);
            Eigen::Vector3d Ow = pKF->GetCameraCenter();
            if(!vCovKFs.empty())
            {
                for(vector<Frame*>::const_iterator vit=vCovKFs.begin(), vend=vCovKFs.end(); vit!=vend; vit++)
                {
                    if((*vit)->keyFrameId_ < pKF->keyFrameId_)	continue;
                    Eigen::Vector3d Ow2 = (*vit)->GetCameraCenter();
                    glVertex3f(Ow.x(),Ow.y(),Ow.z());
                    glVertex3f(Ow2.x(),Ow2.y(),Ow2.z());
                }
            }
        }
        glEnd();
    }
	if(bDrawSpanningTree)
    {
        glLineWidth(mGraphLineWidth);
        glColor4f(0.0f,1.0f,0.0f,0.6f);
        glBegin(GL_LINES);
		for(list<FramePtr>::const_iterator it=lpKFs.begin(),itend=lpKFs.end(); it!=itend; it++)
        {
			FramePtr pKF = *it;
			Eigen::Vector3d Ow = pKF->GetCameraCenter();
            Frame* pParent = pKF->GetParent();
            if(pParent)
            {
                Eigen::Vector3d Owp = pParent->GetCameraCenter();
                glVertex3f(Ow.x(),Ow.y(),Ow.z());
                glVertex3f(Owp.x(),Owp.y(),Owp.z());
            }
        }
        glEnd();
    }
	if(bDrawConstraints)
    {
		set<Frame*> LocalMap = _vo->LocalMap_;
		Vector3d posCurrent(_vo->lastFrame()->GetCameraCenter());
		if(LocalMap.empty()) return;
		glLineWidth(mGraphLineWidth);
		glColor4f(0.0f,1.0f,0.0f,0.6f);
		glBegin(GL_LINES);
		for(set<Frame*>::iterator it = LocalMap.begin(); it != LocalMap.end(); ++it)
		{
			Frame* target = *it;
			if(target->id_ == _vo->lastFrame()->id_) continue;
			Vector3d posTarget(target->GetCameraCenter());
			glVertex3d(posCurrent[0], posCurrent[1], posCurrent[2]);
			glVertex3d(posTarget[0], posTarget[1], posTarget[2]);
		}
		glEnd();
	}
}
void Viewer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
{
    const float &w = mCameraSize;
    const float h = w*0.75;
    const float z = w*0.6;
    glPushMatrix();
    #ifdef HAVE_GLES
            glMultMatrixf(Twc.m);
    #else
            glMultMatrixd(Twc.m);
    #endif
    glLineWidth(mCameraLineWidth);
    glColor3f(0.0f,1.0f,0.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);
    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);
    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);
    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();
    glPopMatrix();
}
void Viewer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M, pangolin::OpenGlMatrix &MOw)
{
    if(_drawedframeID != 0)
    {
        Eigen::Matrix3d Rwc = _CurrentPoseTwc.rotation_matrix();
        Eigen::Vector3d twc = _CurrentPoseTwc.translation();
        M.m[0] = Rwc(0,0);
        M.m[1] = Rwc(1,0);
        M.m[2] = Rwc(2,0);
        M.m[3] = 0.0;
        M.m[4] = Rwc(0,1);
        M.m[5] = Rwc(1,1);
        M.m[6] = Rwc(2,1);
        M.m[7] = 0.0;
        M.m[8] = Rwc(0,2);
        M.m[9] = Rwc(1,2);
        M.m[10] = Rwc(2,2);
        M.m[11] = 0.0;
        M.m[12] = twc[0];
        M.m[13] = twc[1];
        M.m[14] = twc[2];
        M.m[15] = 1.0;
        MOw.SetIdentity();
        MOw.m[12] = twc[0];
        MOw.m[13] = twc[1];
        MOw.m[14] = twc[2];
    }
    else
    {
        M.SetIdentity();
        MOw.SetIdentity();
    }
}
void Viewer::run()
{
    mbFinished = false;
    pangolin::CreateWindowAndBind("VI-HSO", 1024,768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);     
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);  
    pangolin::OpenGlRenderState s_cam
    (
        pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000),
        pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0)
    );
    pangolin::View& d_cam = pangolin::CreateDisplay()
          .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
          .SetHandler(new pangolin::Handler3D(s_cam));
    pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",true,true);
    pangolin::Var<bool> menuCamView("menu.Camera View",false,false);
    pangolin::Var<bool> menuTopView("menu.Top View",false,false);
    pangolin::Var<bool> menuShowTrajactory("menu.Show Trajactory",true,true);
    pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames",true,true);
    pangolin::Var<bool> menuShowPoints("menu.Show Points",true,true);
    pangolin::Var<bool> menuShowGraph("menu.Show Graph",false,true);
	pangolin::Var<bool> menuShowSpanningTree("menu.Show SpanningTree",false,true);
	pangolin::Var<bool> menuShowConstrains("menu.Show Constrains",false,true);
    pangolin::Var<bool> menuStepByStep("menu.Step By Step",false,true);
    pangolin::Var<bool> menuStep("menu.Step",false,false);
    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();
    pangolin::OpenGlMatrix Ow;
    Ow.SetIdentity();
    bool bFollow = true;
    bool bCameraView = true;
    bool bStepByStep = false;
    string FileName = _vo->mFILE_NAME;
    cv::namedWindow(FileName);
    while(!CheckFinish())
    {
        usleep(10000);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        GetCurrentOpenGLCameraMatrix(Twc, Ow);
        if(mbStopTrack)
        {
            menuStepByStep = true;
            mbStopTrack = false;
        }
        if(menuFollowCamera && bFollow)
        {
            if(bCameraView)
                s_cam.Follow(Twc);
            else
                s_cam.Follow(Ow);
        }
        else if(menuFollowCamera && !bFollow)
        {
            if(bCameraView)
            {
                s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000));
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
                s_cam.Follow(Twc);
            }
            else
            {
                s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024,768,3000,3000,512,389,0.1,1000));
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0,0.01,10, 0,0,0,0.0,0.0, 1.0));
                s_cam.Follow(Ow);
            }
            bFollow = true;
        }
        else if(!menuFollowCamera && bFollow)
        {
            bFollow = false;
        }
        if(menuCamView)
        {
            menuCamView = false;
            bCameraView = true;
            s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,10000));
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
            s_cam.Follow(Twc);
        }
        if(menuTopView && _vo->mpMap->isImuInitialized())
        {
            menuTopView = false;
            bCameraView = false;
            s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024,768,3000,3000,512,389,0.1,10000));
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0,0.01,50, 0,0,0,0.0,0.0, 1.0));
            s_cam.Follow(Ow);
        }
        if(menuStepByStep && !bStepByStep)
        {
            _vo->SetStepByStep(true);
            bStepByStep = true;
        }
        else if(!menuStepByStep && bStepByStep)
        {
            _vo->SetStepByStep(false);
            bStepByStep = false;
        }
        if(menuStep)
        {
            _vo->mbStep = true;
            menuStep = false;
        }
        d_cam.Activate(s_cam);
        glClearColor(1.0f,1.0f,1.0f,0.5f);
        DrawCurrentCamera(Twc);
        DrawKeyFrames(menuShowTrajactory, menuShowKeyFrames);
        if(menuShowPoints) 
            DrawMapRegionPoints();
        if(menuShowGraph || menuShowSpanningTree || menuShowConstrains)
			DrawConnections(menuShowGraph, menuShowSpanningTree, menuShowConstrains);
        pangolin::FinishFrame();
        _qProcessTimes.push_back(_vo->last_frame_->mProcessTime);
        if(_qProcessTimes.size()>100)    
            _qProcessTimes.pop_front();
        cv::Mat tracking_img = DrawFrame();
		cv::imshow (FileName, tracking_img);
    }
    pangolin::BindToContext("VI_HSO");
    std::cout<<"pangolin close"<<std::endl;
	SetFinish();
}
void Viewer::SuphusToOpenGLMatrix(Sophus::SE3 &T, pangolin::OpenGlMatrix &M)
{
    Eigen::Matrix3d Rwc = T.rotation_matrix();
    Eigen::Vector3d twc = T.translation();
    M.m[0] = Rwc(0,0);
    M.m[1] = Rwc(1,0);
    M.m[2] = Rwc(2,0);
    M.m[3] = 0.0;
    M.m[4] = Rwc(0,1);
    M.m[5] = Rwc(1,1);
    M.m[6] = Rwc(2,1);
    M.m[7] = 0.0;
    M.m[8] = Rwc(0,2);
    M.m[9] = Rwc(1,2);
    M.m[10] = Rwc(2,2);
    M.m[11] = 0.0;
    M.m[12] = twc[0];
    M.m[13] = twc[1];
    M.m[14] = twc[2];
    M.m[15] = 1.0;
}
cv::Mat Viewer::DrawFrame()
{
    cv::Mat im;
    vihso::FramePtr pFrame = _vo->lastFrame();
    int state; 
    {
        unique_lock<mutex> lock(mMutexIMG);
        state=_vo->mState;
        pFrame->img_pyr_[0].copyTo(im);
        cvtColor(im,im,CV_GRAY2RGB);
        if(pFrame != NULL)
        {
            if((state == FrameHandlerBase::NO_IMAGES_YET) || (state == FrameHandlerBase::NOT_INITIALIZED))
            {
                im = _vo->klt_homography_init_.cvImgshow.clone();
            }
            else
            {
				Eigen::Vector3d Ow = pFrame->GetCameraCenter();
				list<Feature*> lfts = pFrame->GetFeatures();
                for(auto& ft:lfts)
                {
                    if(ft->point == NULL) continue;
					int len = round((ft->point->GetWorldPos()-Ow).norm()/15 * 200); 
					if(len>200) 
						len=200;
					cv::Scalar sc(255*PesudoPalette[len][2], 255*PesudoPalette[len][1], 255*PesudoPalette[len][0]);
                    if(ft->type == vihso::Feature::EDGELET)
					{					
        				cv::rectangle(im, cv::Point2f(ft->px.x()-3, ft->px.y()-3), cv::Point2f(ft->px.x()+3, ft->px.y()+3), cv::Scalar ( 255,0,0 ), FILLED);
					}			 
                    else
					{
						cv::rectangle(im, cv::Point2f(ft->px.x()-3, ft->px.y()-3), cv::Point2f(ft->px.x()+3, ft->px.y()+3), cv::Scalar ( 0,255,0 ), FILLED);
					}
                }
            }
        }
    }
    cv::Mat imWithInfo;
    DrawTextInfo(im,state, imWithInfo);
    return imWithInfo;
}
void Viewer::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
{
    stringstream s;
    if(nState==FrameHandlerBase::NO_IMAGES_YET)
        s << " WAITING FOR IMAGES";
    else if(nState==FrameHandlerBase::NOT_INITIALIZED)
        s << " TRYING TO IMG INITIALIZE ";
    else if(nState==FrameHandlerBase::ONLY_VISIUAL_INIT)
        s << " TRYING TO IMU INITIALIZE ";
    else if(nState==FrameHandlerBase::OK)
    {
        s << "SLAM MODE |  ";
        size_t nKFs = _vo->mpMap->GetNumOfKF();
		int frame_id = _vo->last_frame_->id_;
        size_t nMPs = _vo->mpMap->GetNumOfMP();
        size_t mnTracked = _vo->last_frame_->m_n_inliers;
        double v = _vo->last_frame_->mVw.norm();
        double dProcessTimes=0.0;
        for(list<double>::iterator it=_qProcessTimes.begin(),itend=_qProcessTimes.end(); it!=itend; it++)
        {
            dProcessTimes += (*it);
        }
        dProcessTimes = dProcessTimes/_qProcessTimes.size();
        s  << "Frame ID: " << frame_id ;
    }
    else if(nState==FrameHandlerBase::LOST)
    {
        s << " TRACK LOST. TRYING TO RELOCALIZE ";
    }
    else if(nState==FrameHandlerBase::SYSTEM_NOT_READY)
    {
        s << " LOADING configuration file. PLEASE WAIT...";
    }
    int baseline=0;
    cv::Size textSize = cv::getTextSize(    s.str(),                    
                                            cv::FONT_HERSHEY_PLAIN,     
                                            1,                          
                                            1,                          
                                            &baseline);
    imText = cv::Mat(im.rows+textSize.height+10,im.cols,im.type());
    im.copyTo(imText.rowRange(0,im.rows).colRange(0,im.cols));
    imText.rowRange(im.rows,imText.rows) = cv::Mat::zeros(textSize.height+10,im.cols,im.type());
    cv::putText(imText,s.str(),cv::Point(5,imText.rows-5), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,255,255), 1, 8);
}
}
