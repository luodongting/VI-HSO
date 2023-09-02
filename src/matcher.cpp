#include <cstdlib>
#include <vihso/matcher.h>
#include <vihso/frame.h>
#include <vihso/feature.h>
#include <vihso/point.h>
#include <vihso/config.h>
#include <vihso/feature_alignment.h>
#include <vihso/depth_filter.h>
#include "vihso/camera.h"
#include "vihso/vikit/math_utils.h"
#include "vihso/vikit/vision.h"
#include "vihso/vikit/patch_score.h"
#include <cmath>
#define LIGHT_THRESHOLD 30.0f
namespace vihso {
namespace warp {
void getWarpMatrixAffine(const hso::AbstractCamera& cam_ref,
                         const hso::AbstractCamera& cam_cur,
                         const Vector2d& px_ref,
                         const Vector3d& f_ref,
                         const double depth_ref,
                         const SE3& T_cur_ref,
                         const int level_ref,   
                         Matrix2d& A_cur_ref)
{
    const int halfpatch_size = 5;
    const Vector3d xyz_ref(f_ref*depth_ref);    
    const int ratio = (1<<level_ref);
    Vector3d xyz_du_ref(cam_ref.cam2world(px_ref + Vector2d(halfpatch_size,0)*ratio));  
    Vector3d xyz_dv_ref(cam_ref.cam2world(px_ref + Vector2d(0,halfpatch_size)*ratio));
    xyz_du_ref *= xyz_ref[2]/xyz_du_ref[2]; 
    xyz_dv_ref *= xyz_ref[2]/xyz_dv_ref[2]; 
    const Vector2d px_cur(cam_cur.world2cam(T_cur_ref*(xyz_ref)));  
    const Vector2d px_du(cam_cur.world2cam(T_cur_ref*(xyz_du_ref)));
    const Vector2d px_dv(cam_cur.world2cam(T_cur_ref*(xyz_dv_ref)));
    A_cur_ref.col(0) = (px_du - px_cur)/halfpatch_size; 
    A_cur_ref.col(1) = (px_dv - px_cur)/halfpatch_size;
}
int getBestSearchLevel(const Matrix2d& A_cur_ref, const int max_level)
{
    int search_level = 0;
    double D = A_cur_ref.determinant(); 
    while(D > 3.0 && search_level < max_level)   
    {
        search_level += 1;
        D *= 0.25;   
    }
    return search_level;
}
void warpAffine(const Matrix2d& A_cur_ref,   
                const cv::Mat& img_ref,      
                const Vector2d& px_ref,      
                const int level_ref,         
                const int search_level,      
                const int halfpatch_size,    
                uint8_t* patch)              
{
  const int patch_size = halfpatch_size*2;
  const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();  
  if(isnan(A_ref_cur(0,0)))  
  {
    printf("Affine warp is NaN, probably camera has no translation\n");  
    return;
  }
  uint8_t* patch_ptr = patch;
  const Vector2f px_ref_pyr = px_ref.cast<float>() / (1<<level_ref);     
  for (int y=0; y<patch_size; ++y)
    for (int x=0; x<patch_size; ++x, ++patch_ptr)    
    {
      Vector2f px_patch(x-halfpatch_size, y-halfpatch_size); 
      px_patch *= (1<<search_level);         
      const Vector2f px(A_ref_cur*px_patch + px_ref_pyr);
      if (px[0]<0 || px[1]<0 || px[0]>=img_ref.cols-1 || px[1]>=img_ref.rows-1)
        *patch_ptr = 0;
      else
        *patch_ptr = (uint8_t) hso::interpolateMat_8u(img_ref, px[0], px[1]);
    }
}
void warpAffine(
    const Matrix2d& A_cur_ref,
    const cv::Mat& img_ref,
    const Vector2d& px_ref,
    const int level_ref,
    const int search_level,
    const int halfpatch_size,
    float* patch)
{
    const int patch_size = halfpatch_size*2;
    const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
    if(isnan(A_ref_cur(0,0)))
    {
        printf("Affine warp is NaN, probably camera has no translation\n");  
        return;
    }
    float* patch_ptr = patch;
    const Vector2f px_ref_pyr = (px_ref / (1<<level_ref)).cast<float>();
    const float scaleTarget = (1<<search_level);
    for (int y=0; y<patch_size; ++y)
        for (int x=0; x<patch_size; ++x, ++patch_ptr)
        {   
            Vector2f px_patch(x-halfpatch_size, y-halfpatch_size);
            px_patch *= scaleTarget;
            const Vector2f px(A_ref_cur*px_patch + px_ref_pyr);
            if (px[0]<0 || px[1]<0 || px[0]>=img_ref.cols-1 || px[1]>=img_ref.rows-1)
                *patch_ptr = 0;
            else
                *patch_ptr = hso::interpolateMat_8u(img_ref, px[0], px[1]);
        }
}
void createPatch(
  float* patch,                  
  const Vector2d& px_scaled,    
  const Frame* cur_frame,
  const int halfpatch_size,      
  const int level)      
{
    int patch_size = halfpatch_size*2;
    int stride = cur_frame->img_pyr_[level].cols;
    float u_cur = px_scaled[0];
    float v_cur = px_scaled[1];
    int ui = floorf(u_cur);
    int vi = floorf(v_cur);
    float subpix_u_ref = u_cur - ui;
    float subpix_v_ref = v_cur - vi;
    float w_ref_tl = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);
    float w_ref_tr = subpix_u_ref * (1.0-subpix_v_ref);
    float w_ref_bl = (1.0-subpix_u_ref) * subpix_v_ref;
    float w_ref_br = 1.0-w_ref_tl-w_ref_tr-w_ref_bl;
    float* patch_ptr = patch;
    for(int y = 0; y < patch_size; ++y)
    {
        uint8_t* cur_patch_ptr = cur_frame->img_pyr_[level].data
                                + (vi - halfpatch_size + y) * stride
                                + (ui - halfpatch_size);
        for(int x = 0; x < patch_size; ++x, ++patch_ptr, ++cur_patch_ptr)
        {
            float intensity = w_ref_tl*cur_patch_ptr[0]      + w_ref_tr*cur_patch_ptr[1] + 
                              w_ref_bl*cur_patch_ptr[stride] + w_ref_br*cur_patch_ptr[stride+1];
            *patch_ptr = intensity;          
        }
    }
}
void convertPatchFloat2Int(
    float* patch_f,
    uint8_t* patch_i,
    const int patch_size)
{
  const int patch_area = patch_size*patch_size;
  float* patch_f_ptr = patch_f;
  uint8_t* patch_i_ptr = patch_i;
  for(int i = 0; i < patch_area; ++i, ++patch_f_ptr, ++patch_i_ptr)
  {
    *patch_i_ptr = (uint8_t)*patch_f_ptr;
  }
}
void createPatchFromPatchWithBorder(
    uint8_t* patch,
    uint8_t* patch_with_border,
    const int patch_size)
{
  uint8_t* ref_patch_ptr = patch;
  for(int y=1; y<patch_size+1; ++y, ref_patch_ptr += patch_size)
  {
    uint8_t* ref_patch_border_ptr = patch_with_border + y*(patch_size+2) + 1;
    for(int x=0; x<patch_size; ++x)
      ref_patch_ptr[x] = ref_patch_border_ptr[x];
  }
}
void createPatchFromPatchWithBorder(
    float* patch,
    float* patch_with_border,
    const int patch_size)
{
  float* ref_patch_ptr = patch;
  for(int y=1; y<patch_size+1; ++y, ref_patch_ptr += patch_size)
  {
    float* ref_patch_border_ptr = patch_with_border + y*(patch_size+2) + 1;
    for(int x=0; x<patch_size; ++x)
      ref_patch_ptr[x] = ref_patch_border_ptr[x];
  }
}
}  
bool depthFromTriangulation(
    const SE3& T_search_ref,
    const Vector3d& f_ref,
    const Vector3d& f_cur,
    double& depth)
{
    Matrix<double,3,2> A; A << T_search_ref.rotation_matrix() * f_ref, f_cur;
    const Matrix2d AtA = A.transpose()*A;
    if(AtA.determinant() < 0.000001)
        return false;
    const Vector2d depth2 = - AtA.inverse()*A.transpose()*T_search_ref.translation();
    depth = fabs(depth2[0]);
    return true;
}
void Matcher::createPatchFromPatchWithBorder()
{
  uint8_t* ref_patch_ptr = patch_;
  for(int y=1; y<patch_size_+1; ++y, ref_patch_ptr += patch_size_)
  {
    uint8_t* ref_patch_border_ptr = patch_with_border_ + y*(patch_size_+2) + 1;
    for(int x=0; x<patch_size_; ++x)
      ref_patch_ptr[x] = ref_patch_border_ptr[x];
  }
}
/**
 * inverse compositional Lucas-Kanade algorithm
 * 图像配准，优化3d点在当前帧中的投影像素坐标。参考帧为3d点观测帧中与当前帧视差最小的一帧
 * @param pt        世界点
 * @param cur_frame 当前帧
 * @param px_cur    input/output 世界点投影到当前帧的像素点（输入初始估计，输出精确估计）
*/
bool Matcher::findMatchDirect(Point& pt, Frame& cur_frame, Vector2d& px_cur)
{
    /*
    bool find_last = false;
    const int pt_id = pt.id_;
    if(cur_frame.lastFrameAndAffine_.frame->id_ != cur_frame.lastKeyFrameAndAffine_.frame->id_)
    {
        FramePtr LastFrame = cur_frame.lastFrameAndAffine_.frame;
        for(Features::iterator it = LastFrame->fts_.begin(); it != LastFrame->fts_.end(); ++it)
        {
            if((*it)->point == NULL) continue;
            if((*it)->point->id_ == pt_id) {
                find_last = true;
                ref_ftr_ = *it;
                break;
            }
        }
    }
    */
    if(!pt.getCloseViewObs(cur_frame.GetCameraCenter(), ref_ftr_))
    {
            return false;
    }
    if(!ref_ftr_->frame->cam_->isInFrame((ref_ftr_->px/(1<<ref_ftr_->level)).cast<int>(), halfpatch_size_+2, ref_ftr_->level))
        return false;
	Sophus::SE3 Tcw = cur_frame.GetPoseSE3();
	Sophus::SE3 Twr = ref_ftr_->frame->GetPoseInverseSE3();	
    SE3 T_c_r =  Tcw * Twr;
    if(ref_ftr_->frame->id_ == pt.hostFeature_->frame->id_)
    {
        double idist = pt.GetIdist();
        warp::getWarpMatrixAffine(  *ref_ftr_->frame->cam_, *cur_frame.cam_, ref_ftr_->px, ref_ftr_->f,
                                    1.0/idist, T_c_r, ref_ftr_->level, A_cur_ref_);
    }
    else
    {
        const Vector3d PointPose = pt.GetWorldPos();
        warp::getWarpMatrixAffine(  *ref_ftr_->frame->cam_, *cur_frame.cam_, ref_ftr_->px, ref_ftr_->f,
                                    (ref_ftr_->frame->GetCameraCenter() - PointPose).norm(), T_c_r, ref_ftr_->level, A_cur_ref_);
    }
    search_level_ = warp::getBestSearchLevel(A_cur_ref_, Config::nPyrLevels()-1);
    float patch_with_border_f_temp[(patch_size_+2)*(patch_size_+2)]; 
    warp::warpAffine(A_cur_ref_, ref_ftr_->frame->img_pyr_[ref_ftr_->level], ref_ftr_->px, 
                    ref_ftr_->level, search_level_, halfpatch_size_+1, patch_with_border_f_temp);
    const int patchWithBorderArea = (patch_size_+2)*(patch_size_+2);
    if(cur_frame.keyFrameId_ - ref_ftr_->frame->keyFrameId_ < 4) 
    {
        float exposure_rat = cur_frame.m_exposure_time/ref_ftr_->frame->m_exposure_time;
        if(fabsf(exposure_rat*128 - 128) > LIGHT_THRESHOLD)  
        {
            float* patch_ptr = patch_with_border_f_;
            float* patch_temp = patch_with_border_f_temp;
            for(int i = 0; i < patchWithBorderArea; ++i, ++patch_ptr, ++patch_temp)
            {
                float intensity = (*patch_temp) * exposure_rat;
                *patch_ptr = intensity;
            }
        }
        else
            std::memcpy(patch_with_border_f_, patch_with_border_f_temp, patchWithBorderArea*sizeof(float));
    }
    else
        std::memcpy(patch_with_border_f_, patch_with_border_f_temp, patchWithBorderArea*sizeof(float));
    warp::createPatchFromPatchWithBorder(patch_f_, patch_with_border_f_, patch_size_);   
    Vector2d px_scaled(px_cur/(1<<search_level_));
    const Vector2d px_scaled_orig = px_scaled;
    float patchNCC[patch_size_*patch_size_];
    bool alignResult;
    if(ref_ftr_->type == Feature::EDGELET)  
    {
        Vector2d dir_cur(A_cur_ref_*ref_ftr_->grad);     
        dir_cur.normalize();     
        alignResult = feature_alignment::align1D(   cur_frame.img_pyr_[search_level_],   
                                                    dir_cur.cast<float>(),               
                                                    patch_with_border_f_,
                                                    patch_f_,
                                                    options_.align_max_iter,             
                                                    px_scaled,                           
                                                    h_inv_,                              
                                                    patchNCC);                           
        if(alignResult)  
            alignResult = checkNormal(cur_frame, search_level_, px_scaled, dir_cur, Config::edgeLetCosAngle());  
    }
    else                 
    {
        alignResult = feature_alignment::align2D(
            cur_frame.img_pyr_[search_level_], patch_with_border_f_, 
            patch_f_, options_.align_max_iter, px_scaled, false, patchNCC);
    }
    if(alignResult) 
        alignResult = checkNCC(patch_f_, patchNCC, 0.7);
    if(alignResult)
        alignResult = (px_scaled_orig-px_scaled).norm() < 20;    
    px_cur = px_scaled * (1<<search_level_);
    return alignResult;
}
bool Matcher::checkNCC(float* patch1, float* patch2, float thresh)   
{
    const int NCC_area = patch_size_*patch_size_;    
    float mean1 = 0, mean2 = 0;
    for(int i = 0; i < NCC_area; ++i)
    {
        mean1 += patch1[i];
        mean2 += patch2[i];
    }
    mean1 /= NCC_area;
    mean2 /= NCC_area;
    float numerator = 0, demoniator1 = 0, demoniator2 = 0;
    for (int i = 0; i < NCC_area; i++)
    {
        float patch1_mean = patch1[i]-mean1;
        float patch2_mean = patch2[i]-mean2;
        numerator   += patch1_mean*patch2_mean;
        demoniator1 += patch1_mean*patch1_mean;
        demoniator2 += patch2_mean*patch2_mean;
    }
    return (numerator / (sqrt(demoniator1*demoniator2)+1e-12)) > thresh;
}
bool Matcher::checkNormal(const Frame& frame, int level, Vector2d pxLevel, Vector2d normal, float thresh)
{
    float uf = pxLevel[0];
    float vf = pxLevel[1];
    int ui = floorf(pxLevel[0]);
    int vi = floorf(pxLevel[1]);
    float subpix_x = uf-ui;
    float subpix_y = vf-vi;
    float wTL = (1.0-subpix_x)*(1.0-subpix_y);
    float wTR = subpix_x * (1.0-subpix_y);
    float wBL = (1.0-subpix_x)*subpix_y;
    float wBR = 1.0-wTL-wTR-wBL;
	if(frame.sobelX_.empty())
	{
		cout << "Matcher::checkNormal\tKFid=" << frame.keyFrameId_  << "\tFrameid=" << frame.id_ << endl;
		return false;
	}
    short gx00 = frame.sobelX_[level].at<short>(vi,ui);
    short gx10 = frame.sobelX_[level].at<short>(vi,ui+1);
    short gx01 = frame.sobelX_[level].at<short>(vi+1,ui);
    short gx11 = frame.sobelX_[level].at<short>(vi+1,ui+1);
    short gy00 = frame.sobelY_[level].at<short>(vi,ui);
    short gy10 = frame.sobelY_[level].at<short>(vi,ui+1);
    short gy01 = frame.sobelY_[level].at<short>(vi+1,ui);
    short gy11 = frame.sobelY_[level].at<short>(vi+1,ui+1);
    Vector2d n00(gx00, gy00);
    Vector2d n10(gx10, gy10);
    Vector2d n01(gx01, gy01);
    Vector2d n11(gx11, gy11);
    Vector2d n = wTL*n00 + wTR*n10 + wBL*n01 + wBR*n11;  
    n.normalize();
    return (normal.dot(n) > thresh);
}
bool Matcher::findMatchSeed(const Seed& seed, Frame& cur_frame, Vector2d& px_cur, float ncc_thresh)
{
	Sophus::SE3 Twseed = seed.ftr->frame->GetPoseInverseSE3();
    Vector3d seed_pos(Twseed * (1.0/seed.mu * seed.ftr->f));    
    Vector3d ref_dir(seed.ftr->frame->GetCameraCenter() - seed_pos); ref_dir.normalize();            
    Vector3d cur_dir(cur_frame.GetCameraCenter() - seed_pos); cur_dir.normalize();                   
    const double cos_angle = ref_dir.dot(cur_dir);
    if(cos_angle < 0.5) return false;
    ref_ftr_ = seed.ftr;
    if(!ref_ftr_->frame->cam_->isInFrame((ref_ftr_->px/(1<<ref_ftr_->level)).cast<int>(), halfpatch_size_+2, ref_ftr_->level))
    {
		return false;
	}   
	Sophus::SE3 Tcw = cur_frame.GetPoseSE3();
	Sophus::SE3 Twr = ref_ftr_->frame->GetPoseInverseSE3();	
    SE3 T_c_r =  Tcw * Twr;	 
    warp::getWarpMatrixAffine(
        *ref_ftr_->frame->cam_, *cur_frame.cam_, ref_ftr_->px, ref_ftr_->f,
        1./seed.mu, T_c_r, ref_ftr_->level, A_cur_ref_);
    search_level_ = warp::getBestSearchLevel(A_cur_ref_, Config::nPyrLevels()-1);
    warp::warpAffine(
        A_cur_ref_, ref_ftr_->frame->img_pyr_[ref_ftr_->level], ref_ftr_->px,
        ref_ftr_->level, search_level_, halfpatch_size_+1, patch_with_border_f_);
    const int patchWithBorderArea = (patch_size_+2)*(patch_size_+2);
    float exposure_rat = cur_frame.m_exposure_time/ref_ftr_->frame->m_exposure_time;
    if(fabsf(exposure_rat*128 - 128) > LIGHT_THRESHOLD) 
    {
        float* patch_ptr = patch_with_border_f_;
        for(int i=0; i<patchWithBorderArea; ++i, ++patch_ptr)
        {
            float intensity = (*patch_ptr)*exposure_rat;
            *patch_ptr = intensity;
        }
    }
    warp::createPatchFromPatchWithBorder(patch_f_, patch_with_border_f_, patch_size_);
    Vector2d px_scaled(px_cur/(1<<search_level_));
    const Vector2d px_scaled_orig = px_scaled;
    bool success = false;
    float patchNCC[patch_size_*patch_size_];
    if(ref_ftr_->type == Feature::EDGELET)
    {
        Vector2d dir_cur(A_cur_ref_*ref_ftr_->grad);
        dir_cur.normalize();
        success = feature_alignment::align1D(cur_frame.img_pyr_[search_level_], dir_cur.cast<float>(), 
            								patch_with_border_f_, patch_f_, options_.align_max_iter, px_scaled, h_inv_, patchNCC);
        if(success) 
            success = checkNormal(cur_frame, search_level_, px_scaled, dir_cur, Config::edgeLetCosAngle());
    }
    else
	{
		success = feature_alignment::align2D(cur_frame.img_pyr_[search_level_], patch_with_border_f_, patch_f_,
												options_.align_max_iter, px_scaled, false, patchNCC);
	}
    if(success) 
        success = checkNCC(patch_f_, patchNCC, 0.8); 
    if(success)
        success = (px_scaled_orig-px_scaled).norm() < 20;
    px_cur = px_scaled * (1<<search_level_);
    return success;
}
#define MIN_DEPTH 0.05  
#define MAX_EPL_LENGTH_CROP 100.0  
#define MIN_EPL_LENGTH_CROP (2.0)  
#define GRADIENT_SAMPLE_DIST 1.0
#define SAMPLE_POINT_TO_BORDER 8
#define MAX_ERROR_STEREO (1300.0f)  
#define MIN_DISTANCE_ERROR_STEREO (1.5f)  
#define STEREO_EPL_VAR_FAC 2.0f
int Matcher::doLineStereo(  Frame& ref_frame,			 
                            Frame& cur_frame,     		 
                            const Feature& ref_ftr,      
                            const double min_idepth,     
                            const double prior_idepth,   
                            const double max_idepth,     
                            double& result_depth,        
                            Vector2i& EPL_start,         
                            Vector2i& EPL_end)           
{
	Sophus::SE3 Tcw = cur_frame.GetPoseSE3();
	Sophus::SE3 Twr = ref_frame.GetPoseInverseSE3();	
    SE3 T_cur_ref = Tcw * Twr;   
    warp::getWarpMatrixAffine(*ref_frame.cam_, *cur_frame.cam_, ref_ftr.px, ref_ftr.f,
                                prior_idepth, T_cur_ref, ref_ftr.level, A_cur_ref_);
    search_level_ = warp::getBestSearchLevel(A_cur_ref_, Config::nPyrLevels()-1);
    warp::warpAffine(A_cur_ref_, ref_frame.img_pyr_[ref_ftr.level], ref_ftr.px,
                    ref_ftr.level, search_level_, halfpatch_size_+1, patch_with_border_f_);
    float exposure_rat = cur_frame.m_exposure_time/ref_frame.m_exposure_time;
    if(fabsf(exposure_rat*128 - 128) > LIGHT_THRESHOLD) 
    {
        float* patch_ptr = patch_with_border_f_;
        for(int i=0; i<100; ++i, ++patch_ptr)
        {
            float intensity = (*patch_ptr) * exposure_rat;
            *patch_ptr = intensity;
        }
    }
    warp::createPatchFromPatchWithBorder(patch_f_, patch_with_border_f_, patch_size_);
    Vector3d pClose(T_cur_ref * (ref_ftr.f*min_idepth));
    pClose = pClose / pClose[2];  
    Vector3d pFar(T_cur_ref * (ref_ftr.f*max_idepth));
    if(pFar[2] < 0.001 || max_idepth < min_idepth) 
    {
        return -1;   
    }
    pFar = pFar / pFar[2];  
    if(isnanf((float)(pFar[0]+pClose[0])))
    {
        return -1;
    }
    Vector2d px_close(cur_frame.cam_->world2cam(Vector2d(pClose[0], pClose[1]))); 
    EPL_start = px_close.cast<int>();        
    px_close = px_close/(1<<search_level_);  
    Vector2d px_far(cur_frame.cam_->world2cam(Vector2d(pFar[0], pFar[1]))); 
    EPL_end = px_far.cast<int>();            
    px_far = px_far/(1<<search_level_);      
    double incx = px_close[0] - px_far[0];
    double incy = px_close[1] - px_far[1];
    double eplLength = sqrt(incx*incx+incy*incy);    
    if(!(eplLength > 0) || std::isinf(eplLength)) 
    {
        return -1;
    }
    if(eplLength > MAX_EPL_LENGTH_CROP)  
    {
        px_close[0] = px_far[0] + incx*MAX_EPL_LENGTH_CROP/eplLength;
        px_close[1] = px_far[1] + incy*MAX_EPL_LENGTH_CROP/eplLength;
    }
    incx *= GRADIENT_SAMPLE_DIST/eplLength;  
    incy *= GRADIENT_SAMPLE_DIST/eplLength;
    px_far[0] -= incx;
    px_far[1] -= incy;
    px_close[0] += incx;
    px_close[1] += incy;
    if(eplLength < MIN_EPL_LENGTH_CROP)  
    {
        double pad = (MIN_EPL_LENGTH_CROP - (eplLength)) / 2.0f;
        px_far[0] -= incx*pad;
        px_far[1] -= incy*pad;
        px_close[0] += incx*pad;
        px_close[1] += incy*pad;
    }
    double epi_grad_angle=1;
    if((ref_ftr.type == Feature::GRADIENT || ref_ftr.type == Feature::EDGELET) && options_.epi_search_edgelet_filtering)
    {
        const Vector2d grad_cur = (A_cur_ref_ * ref_ftr.grad).normalized();          
        const double cosangle = fabs(grad_cur.dot((px_close-px_far).normalized()));  
        epi_grad_angle = cosangle;
        if(cosangle < options_.epi_search_edgelet_max_angle) return -1;  
    }
    double cpx = px_far[0];
    double cpy = px_far[1];
    hso::patch_score::ZMNCC_F<halfpatch_size_> patchScore(patch_f_);     
    float zmncc_best = 0.1;
    float zmncc_second = zmncc_best;
    Vector2d uv_best, uv_second, px;
    Vector2i pxi;
    float patch_f[patch_size_*patch_size_];      
    int loopCounter = 0;
    int loopCBest=-1, loopCSecond =-1;
    while(((incx < 0) == (cpx > px_close[0]) && (incy < 0) == (cpy > px_close[1])) || loopCounter == 0)
    {
        px  = Vector2d(cpx, cpy);
        if(!cur_frame.cam_->isInFrame(px.cast<int>(), patch_size_, search_level_))
        {
            cpx += incx;
            cpy += incy;
            loopCounter++;
            continue;
        }
        warp::createPatch(patch_f, px, &cur_frame, halfpatch_size_, search_level_);  
        float zmncc = patchScore.computeScore(patch_f);
        if(zmncc > zmncc_best)
        {
            zmncc_second = zmncc_best;
            uv_best = px;
            zmncc_best = zmncc;
            loopCSecond = loopCBest;
            loopCBest = loopCounter;
        }
        else if(zmncc > zmncc_second)
        {
            zmncc_second = zmncc;
            loopCSecond = loopCounter;
        }
        cpx += incx;
        cpy += incy;
        loopCounter++;
    }
    if(abs(loopCBest - loopCSecond) > 1.0f && MIN_DISTANCE_ERROR_STEREO * zmncc_second > zmncc_best)
        return -4;
     Vector2d uv_best_0 = uv_best*(1<<search_level_);     
    if(zmncc_best > 0.8)     
    {
        px_cur_ = uv_best_0;
        Vector2d px_scaled(px_cur_/(1<<search_level_));
        bool result = KLTLimited1D(
                    cur_frame.img_pyr_[search_level_], patch_with_border_f_, patch_f_,
                    options_.align_max_iter, px_scaled, (px_close-px_far).normalized(), NULL, false);
        float patch2D[patch_size_*patch_size_];  
        if(!result)
        {
            Vector2d px_2d(px_cur_/(1<<search_level_));
            if(ref_ftr.type != Feature::EDGELET)
                result = KLTLimited2D(
                    cur_frame.img_pyr_[search_level_], patch_with_border_f_, patch_f_,
                    options_.align_max_iter, px_2d, patch2D, false);
            else
            {
                Vector2d dir_cur(A_cur_ref_*ref_ftr.grad);
                dir_cur.normalize();
                result = KLTLimited1D(
                    cur_frame.img_pyr_[search_level_], patch_with_border_f_, patch_f_,
                    options_.align_max_iter, px_2d, dir_cur, patch2D, false);
                if(result) result = checkNormal(cur_frame, search_level_, px_2d, dir_cur, 0.7);
            }
            px_scaled = px_2d;
        }
        else
        {
            if(ref_ftr.type != Feature::EDGELET)
                result = KLTLimited2D(
                    cur_frame.img_pyr_[search_level_], patch_with_border_f_, patch_f_,
                    options_.align_max_iter, px_scaled, patch2D, false);
            else
            {
                Vector2d dir_cur(A_cur_ref_*ref_ftr.grad);
                dir_cur.normalize();
                result = KLTLimited1D(
                    cur_frame.img_pyr_[search_level_], patch_with_border_f_, patch_f_,
                    options_.align_max_iter, px_scaled, dir_cur, patch2D, false);
                if(result) result = checkNormal(cur_frame, search_level_, px_scaled, dir_cur, 0.7);
            }
        }
        if(result)
           result = checkNCC(patch_f_, patch2D, 0.8);
        if(result)
        {
            px_cur_ = px_scaled*(1<<search_level_);  
            if(depthFromTriangulation(T_cur_ref, ref_ftr.f, cur_frame.cam_->cam2world(px_cur_), result_depth))
                return 1;
            return -2;   
        }
        return -3;       
    }
    return -4;           
}
bool Matcher::findEpipolarMatchPrevious(Frame& ref_frame, Frame& cur_frame, const Feature& ref_ftr,
										const double d_estimate, const double d_min, const double d_max, double& depth)
{
	Sophus::SE3 Tcw = cur_frame.GetPoseSE3();
	Sophus::SE3 Twr = ref_frame.GetPoseInverseSE3();	
    SE3 T_cur_ref = Tcw * Twr;   
    Vector2d A = hso::project2d(T_cur_ref * (ref_ftr.f*d_min));
    Vector2d B = hso::project2d(T_cur_ref * (ref_ftr.f*d_max));
    epi_dir_ = A - B;
    warp::getWarpMatrixAffine(
        *ref_frame.cam_, *cur_frame.cam_, ref_ftr.px, ref_ftr.f,
        d_estimate, T_cur_ref, ref_ftr.level, A_cur_ref_);
    search_level_ = warp::getBestSearchLevel(A_cur_ref_, Config::nPyrLevels()-1);
    Vector2d px_A(cur_frame.cam_->world2cam(A));
    Vector2d px_B(cur_frame.cam_->world2cam(B));
    epi_length_ = (px_A-px_B).norm() / (1<<search_level_);
    warp::warpAffine(
        A_cur_ref_, ref_frame.img_pyr_[ref_ftr.level], ref_ftr.px,
        ref_ftr.level, search_level_, halfpatch_size_+1, patch_with_border_f_);
    double epi_grad_angle=1;
    if((ref_ftr.type == Feature::GRADIENT || ref_ftr.type == Feature::EDGELET) && options_.epi_search_edgelet_filtering)
    {
        const Vector2d grad_cur = (A_cur_ref_ * ref_ftr.grad).normalized();
        const double cosangle = fabs(grad_cur.dot(epi_dir_.normalized()));
        epi_grad_angle = cosangle;
        if(cosangle < options_.epi_search_edgelet_max_angle) return false;
    }
    float exposure_rat = cur_frame.m_exposure_time/ref_frame.m_exposure_time;
    if(fabsf(exposure_rat*128 - 128) > LIGHT_THRESHOLD)
    {
        float* patch_ptr = patch_with_border_f_;
        for(int i=0; i<100; ++i, ++patch_ptr)
        {
            float intensity = (*patch_ptr) * exposure_rat;
            *patch_ptr = intensity;
        }
    }
    warp::createPatchFromPatchWithBorder(patch_f_, patch_with_border_f_, patch_size_);
    if(epi_length_ < 2.0)
    {
        px_cur_ = (px_A+px_B)/2.0;
        Vector2d px_scaled(px_cur_/(1<<search_level_));
        bool res = KLTLimited1D(
                    cur_frame.img_pyr_[search_level_], patch_with_border_f_, patch_f_,
                    options_.align_max_iter, px_scaled, (px_A-px_B).normalized(), NULL, false);
        float patch2D[patch_size_*patch_size_];
        if(!res)
        {
            Vector2d px_2d(px_cur_/(1<<search_level_));
            if(ref_ftr.type != Feature::EDGELET)
                res = KLTLimited2D(
                    cur_frame.img_pyr_[search_level_], patch_with_border_f_, patch_f_,
                    options_.align_max_iter, px_2d, patch2D, false);
            else
            {
                Vector2d dir_cur(A_cur_ref_*ref_ftr.grad);
                dir_cur.normalize();
                res = KLTLimited1D(
                    cur_frame.img_pyr_[search_level_], patch_with_border_f_, patch_f_,
                    options_.align_max_iter, px_2d, dir_cur, patch2D, false);
                if(res) res = checkNormal(cur_frame, search_level_, px_2d, dir_cur, 0.7);
            }
            px_scaled = px_2d;
        }
        else
        {
            if(ref_ftr.type != Feature::EDGELET)
                res = KLTLimited2D(
                    cur_frame.img_pyr_[search_level_], patch_with_border_f_, patch_f_,
                    options_.align_max_iter, px_scaled, patch2D, false);
            else
            {
                Vector2d dir_cur(A_cur_ref_*ref_ftr.grad);
                dir_cur.normalize();
                res = KLTLimited1D(
                    cur_frame.img_pyr_[search_level_], patch_with_border_f_, patch_f_,
                    options_.align_max_iter, px_scaled, dir_cur, patch2D, false);
                if(res) res = checkNormal(cur_frame, search_level_, px_scaled, dir_cur, 0.7);
            }
        }
        if(res) res = checkNCC(patch_f_, patch2D, 0.8);
        if(res)
        {
            px_cur_ = px_scaled*(1<<search_level_);
            if(depthFromTriangulation(T_cur_ref, ref_ftr.f, cur_frame.cam_->cam2world(px_cur_), depth))
                return true;
        }
        return false;
    }
    size_t n_steps = epi_length_/0.7; 
    Vector2d step = epi_dir_/n_steps;
    if(n_steps > options_.max_epi_search_steps) 
        return false;
    hso::patch_score::ZMNCC_F<halfpatch_size_> patchScore(patch_f_);
    float zmncc_best = 0.1f;
    float zmncc_second = zmncc_best;
    size_t bestCounter = 0;
    size_t secondCounter = 0;
    Vector2d uv_best, px, px_scaled;
    Vector2i pxi;
    Vector2d uv = B-step;
    ++n_steps;
    float patch_f[patch_size_*patch_size_];
    for(size_t i=0; i<n_steps; ++i, uv+=step)
    {
        px = cur_frame.cam_->world2cam(uv);
        px_scaled = px / (1<<search_level_);
        if(!cur_frame.cam_->isInFrame(px_scaled.cast<int>(), patch_size_, search_level_))
            continue;
        warp::createPatch(patch_f, px_scaled, &cur_frame, halfpatch_size_, search_level_);
        float zmncc = patchScore.computeScore(patch_f);
        if(zmncc > zmncc_best)
        {
            zmncc_second = zmncc_best;
            secondCounter = bestCounter;
            zmncc_best = zmncc;
            bestCounter = i;
            uv_best = uv; 
        }
        else if(zmncc > zmncc_second)
        {
            zmncc_second = zmncc;
            secondCounter = i;
        }
    }
    int countsize = bestCounter - secondCounter;
    if(std::abs(countsize) > 1.0f && MIN_DISTANCE_ERROR_STEREO * zmncc_second > zmncc_best) 
        return false;
    if(zmncc_best > 0.8)
    {
        px_cur_ = cur_frame.cam_->world2cam(uv_best);
        Vector2d px_scaled(px_cur_/(1<<search_level_));
        bool res = KLTLimited1D(
                    cur_frame.img_pyr_[search_level_], patch_with_border_f_, patch_f_,
                    options_.align_max_iter, px_scaled, (px_A-px_B).normalized(), NULL, false);
        float patch2D[patch_size_*patch_size_];
        if(!res)
        {
            Vector2d px_2d(px_cur_/(1<<search_level_));
            if(ref_ftr.type != Feature::EDGELET)
                res = KLTLimited2D(
                    cur_frame.img_pyr_[search_level_], patch_with_border_f_, patch_f_,
                    options_.align_max_iter, px_2d, patch2D, false);
            else
            {
                Vector2d dir_cur(A_cur_ref_*ref_ftr.grad);
                dir_cur.normalize();
                res = KLTLimited1D(
                    cur_frame.img_pyr_[search_level_], patch_with_border_f_, patch_f_,
                    options_.align_max_iter, px_2d, dir_cur, patch2D, false);
                if(res) res = checkNormal(cur_frame, search_level_, px_2d, dir_cur, 0.7);
            }
            px_scaled = px_2d;
        }
        else
        {
            if(ref_ftr.type != Feature::EDGELET)
                res = KLTLimited2D(
                    cur_frame.img_pyr_[search_level_], patch_with_border_f_, patch_f_,
                    options_.align_max_iter, px_scaled, patch2D, false);
            else
            {
                Vector2d dir_cur(A_cur_ref_*ref_ftr.grad);
                dir_cur.normalize();
                res = KLTLimited1D(
                    cur_frame.img_pyr_[search_level_], patch_with_border_f_, patch_f_,
                    options_.align_max_iter, px_scaled, dir_cur, patch2D, false);
                if(res) res = checkNormal(cur_frame, search_level_, px_scaled, dir_cur, 0.7);
            }
        }
        if(res) res = checkNCC(patch_f_, patch2D, 0.8);
        if(res)
        {
            px_cur_ = px_scaled*(1<<search_level_);
            if(depthFromTriangulation(T_cur_ref, ref_ftr.f, cur_frame.cam_->cam2world(px_cur_), depth))
                return true;
        }
        return false;
    }
    return false;
}
bool Matcher::KLTLimited2D( const cv::Mat& targetImg, 
                            float* hostPatchWithBorder, 
                            float* hostPatch,
                            const int n_iter, 
                            Vector2d& targetPxEstimate,
                            float* targetPatch, 
                            bool debugPrint)
{
    const int halfPatchSize = 4;
    const int patchSize = 8;
    const int patchArea = 64;
    float host_dx[patchArea], host_dy[patchArea];
    Matrix3f H; H.setZero();
    float grad_weight[patchArea];
    const int hostStep = patchSize+2;
    float* it_dx = host_dx;
    float* it_dy = host_dy;
    float* it_weight = grad_weight;
    Vector3f J; 
    for(int y=0; y<patchSize; ++y)
    {
        float* it = hostPatchWithBorder + (y+1)*hostStep + 1;
        for(int x=0; x<patchSize; ++x, ++it, ++it_dx, ++it_dy, ++it_weight)
        {
            J[0] = 0.5 * (it[1] - it[-1]);                   
            J[1] = 0.5 * (it[hostStep] - it[-hostStep]);     
            J[2] = 1;                                        
            *it_dx = J[0];
            *it_dy = J[1];
            *it_weight = sqrtf(250.0/(250.0+(J[0]*J[0]+J[1]*J[1])));
            H += J*J.transpose()*(*it_weight);
        }
    }
    for(int i=0;i<3;i++) H(i,i) *= (1+0.001);
    Matrix3f Hinv = H.inverse();
    float mean_diff = 0;
    float bestU = targetPxEstimate.x();
    float bestV = targetPxEstimate.y();
    const int cur_step = targetImg.step.p[0];
    float bestEnergy = 1e8;
    Vector3f step; step.setZero();
    Vector3f stepBack; stepBack.setZero();
    Vector3f Jres; Jres.setZero();
    float uBak=bestU, vBak=bestV, meanBak=mean_diff;
    for(int iter = 0; iter<n_iter; ++iter)
    {
        float* cur_patch_ptr = targetPatch;
        int u_r = floor(bestU);
        int v_r = floor(bestV);
        if(u_r < halfPatchSize || v_r < halfPatchSize || u_r >= targetImg.cols-halfPatchSize || v_r >= targetImg.rows-halfPatchSize)
            break;
        if(isnan(bestU) || isnan(bestV))  
            return false;
        float subpix_x = bestU-u_r;
        float subpix_y = bestV-v_r;
        float wTL = (1.0-subpix_x)*(1.0-subpix_y);
        float wTR = subpix_x * (1.0-subpix_y);
        float wBL = (1.0-subpix_x)*subpix_y;
        float wBR = subpix_x * subpix_y;
        float* it_ref = hostPatch;
        float* it_ref_dx = host_dx;
        float* it_ref_dy = host_dy;
        float* it_weight = grad_weight;
        float energy = 0.0;
        Jres.setZero();
        for(int y=0; y<patchSize; ++y)
        {
            uint8_t* it = (uint8_t*)targetImg.data + (v_r+y-halfPatchSize)*cur_step + u_r-halfPatchSize;
            for(int x=0; x<patchSize; ++x, ++it, ++it_ref, ++it_ref_dx, ++it_ref_dy, ++cur_patch_ptr, ++it_weight)
            {
                float search_pixel = wTL*it[0] + wTR*it[1] + wBL*it[cur_step] + wBR*it[cur_step+1];
                if(!std::isfinite(search_pixel)) {energy+=1e5; continue;}
                float res = search_pixel - (*it_ref) + mean_diff;
                Jres[0] -= res*(*it_ref_dx)*(*it_weight);
                Jres[1] -= res*(*it_ref_dy)*(*it_weight);
                Jres[2] -= res*(*it_weight);
                energy += res*res*(*it_weight);
                *cur_patch_ptr = search_pixel;
            }
        }
        if(energy > bestEnergy)
        {
            stepBack*=0.5;
            bestU = uBak + stepBack[0];
            bestV = vBak + stepBack[1];
            mean_diff = meanBak + stepBack[2];
            if(debugPrint)
                printf("GN BACK %d: E %f. id-step %f. UV %f %f -> %f %f.\n",
                        iter, energy, stepBack[0],
                        uBak, vBak, bestU, bestV);
        }
        else
        {
            step = Hinv * Jres;
            if(step[0] < -0.5) step[0] = -0.5;
            else if(step[0] > 0.5) step[0] = 0.5;
            if(step[1] < -0.5) step[1] = -0.5;
            else if(step[1] > 0.5) step[1] = 0.5;
            if(!std::isfinite(step[0])) step.setZero();
            uBak=bestU;
            vBak=bestV;
            meanBak=mean_diff;
            stepBack=step;
            bestU += step[0];
            bestV += step[1];
            mean_diff += step[2];
            bestEnergy = energy;
            if(debugPrint)
                printf("GN step %d: E %f. id-step %f. UV %f %f -> %f %f.\n",
                        iter, energy, step[0],
                        uBak, vBak, bestU, bestV);
        }
        if(stepBack[0]*stepBack[1] < 0.01*0.01)
        {
            if(debugPrint) 
                cout << "converged." << endl;
            break;
        } 
    }
    targetPxEstimate << bestU, bestV;
    if(bestEnergy > 650*patchArea) return false;
    return true;
}
bool Matcher::KLTLimited1D( const cv::Mat& targetImg,    
                            float* hostPatchWithBorder,  
                            float* hostPatch,            
                            const int n_iter,            
                            Vector2d& targetPxEstimate,  
                            const Vector2d& direct,      
                            float* targetPatch, 
                            bool debugPrint)
{
    const int halfPatchSize = 4;
    const int patchSize = 8;
    const int patchArea = 64;
    float host_d[patchArea];
    Matrix2f H; H.setZero();
    float grad_weight[patchArea];
    const int hostStep = patchSize+2;
    float* it_dv = host_d;
    float* it_weight = grad_weight;
    Vector2f J; 
    for(int y=0; y<patchSize; ++y)
    {
        float* it = hostPatchWithBorder + (y+1)*hostStep + 1;
        for(int x=0; x<patchSize; ++x, ++it, ++it_dv, ++it_weight)
        {
            J[0] = 0.5 * (direct[0]*(it[1] - it[-1]) + direct[1]*(it[hostStep] - it[-hostStep]));
            J[1] = 1;
            *it_dv = J[0];
            *it_weight = sqrtf(250.0/(250.0+(J[0]*J[0])));
            H += J*J.transpose()*(*it_weight);
        }
    }
    for(int i=0;i<2;i++) H(i,i) *= (1+0.001);
    Matrix2f Hinv = H.inverse();
    float mean_diff = 0;
    float bestU = targetPxEstimate.x();
    float bestV = targetPxEstimate.y();
    const int cur_step = targetImg.step.p[0];
    float bestEnergy = 1e8;
    Vector2f step; step.setZero();
    Vector2f stepBack; stepBack.setZero();
    Vector2f Jres; Jres.setZero();
    float uBak=bestU, vBak=bestV, meanBak=mean_diff;
    for(int iter = 0; iter<n_iter; ++iter)
    {
        float* cur_patch_ptr = targetPatch;
        int u_r = floor(bestU);
        int v_r = floor(bestV);
        if(u_r < halfPatchSize || v_r < halfPatchSize || u_r >= targetImg.cols-halfPatchSize || v_r >= targetImg.rows-halfPatchSize)
            break;
        if(isnan(bestU) || isnan(bestV))  
            return false;
        float subpix_x = bestU-u_r;
        float subpix_y = bestV-v_r;
        float wTL = (1.0-subpix_x)*(1.0-subpix_y);
        float wTR = subpix_x * (1.0-subpix_y);
        float wBL = (1.0-subpix_x)*subpix_y;
        float wBR = subpix_x * subpix_y;
        float* it_ref = hostPatch;   
        float* it_ref_dv = host_d;
        float* it_weight = grad_weight;
        float energy = 0.0;
        Jres.setZero();
        for(int y=0; y<patchSize; ++y)
        {
            uint8_t* it = (uint8_t*)targetImg.data + (v_r+y-halfPatchSize)*cur_step + u_r-halfPatchSize;
            for(int x=0; x<patchSize; ++x, ++it, ++it_ref, ++it_ref_dv, ++it_weight)
            {
                float search_pixel = wTL*it[0] + wTR*it[1] + wBL*it[cur_step] + wBR*it[cur_step+1];
                if(!std::isfinite(search_pixel)) {energy+=1e5; continue;}
                float res = search_pixel - (*it_ref) + mean_diff;
                Jres[0] -= res*(*it_ref_dv)*(*it_weight);    
                Jres[1] -= res*(*it_weight);                 
                energy += res*res*(*it_weight);
                if(targetPatch != NULL)
                {
                    *cur_patch_ptr = search_pixel;
                    ++cur_patch_ptr;
                }
            }
        }
        if(energy > bestEnergy)  
        {
            stepBack*=0.5;
            bestU = uBak + stepBack[0]*direct[0];
            bestV = vBak + stepBack[0]*direct[1];
            mean_diff = meanBak + stepBack[1];
            if(debugPrint)
                printf("GN BACK %d: E %f. id-step %f. UV %f %f -> %f %f.\n",
                        iter, energy, stepBack[0],
                        uBak, vBak, bestU, bestV);
        }
        else
        {
            step = Hinv * Jres;          
            if(step[0] < -0.5) step[0] = -0.5;
            else if(step[0] > 0.5) step[0] = 0.5;
            if(!std::isfinite(step[0])) step.setZero();
            uBak=bestU;
            vBak=bestV;
            meanBak=mean_diff;
            stepBack=step;
            bestU += step[0]*direct[0];  
            bestV += step[0]*direct[1];  
            mean_diff += step[1];        
            bestEnergy = energy;
            if(debugPrint)
                printf("GN step %d: E %f. id-step %f. UV %f %f -> %f %f.\n",
                        iter, energy, step[0],
                        uBak, vBak, bestU, bestV);
        }
        if(fabsf(stepBack[0]) < 0.01)
        {
            if(debugPrint) 
                cout << "converged." << endl;
            break;
        } 
    }
    targetPxEstimate << bestU, bestV;
    if(bestEnergy > 650*patchArea) return false;
    return true;
}
} // namespace vihso