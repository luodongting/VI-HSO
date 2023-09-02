#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif
#include <vihso/feature_alignment.h>
#include <vihso/feature.h>
#include <vihso/point.h>
#include <vihso/matcher.h>

namespace vihso {
namespace feature_alignment {

#define SUBPIX_VERBOSE 0

bool align1D(
    const cv::Mat& cur_img,       
    const Vector2f& dir,          
    float* ref_patch_with_border, 
    float* ref_patch,             
    const int n_iter,
    Vector2d& cur_px_estimate,    
    double& h_inv,
    float* cur_patch)             
{
    const int halfpatch_size_ = 4;
    const int patch_size = 8;
    const int patch_area = 64;
    bool converged=false;

    float ref_patch_dv[patch_area]; 
    Matrix2f H; H.setZero();
    float grad_weight[patch_area];
    const int ref_step = patch_size+2;  
    float* it_dv = ref_patch_dv;        
    float* it_weight = grad_weight;    
    Vector2f J;
    for(int y=0; y<patch_size; ++y)
    {
        float* it = ref_patch_with_border + (y+1)*ref_step + 1;
        for(int x=0; x<patch_size; ++x, ++it, ++it_dv, ++it_weight)
        {
            J[0] = 0.5*(dir[0]*(it[1] - it[-1]) + dir[1]*(it[ref_step] - it[-ref_step])); 
            J[1] = 1.;    
            *it_dv = J[0];
            *it_weight = sqrtf(250.0/(250.0+J[0]*J[0])); 
            H += J*J.transpose()*(*it_weight);
        }
    }
    for(int i=0;i<2;i++) H(i,i) *= (1+0.001);
    h_inv = 1.0/H(0,0)*patch_size*patch_size;
    Matrix2f Hinv = H.inverse();
    float mean_diff = 0;  
    float u = cur_px_estimate.x();
    float v = cur_px_estimate.y();
    const float min_update_squared = 0.01*0.01;
    const int cur_step = cur_img.step.p[0]; 
    float chi2 = 0;
    Vector2f update; update.setZero();
    Vector2f Jres; Jres.setZero();
    for(int iter = 0; iter<n_iter; ++iter)
    {
        float* cur_patch_ptr = cur_patch;
        int u_r = floor(u);
        int v_r = floor(v);
        if(u_r < halfpatch_size_ || v_r < halfpatch_size_ || u_r >= cur_img.cols-halfpatch_size_ || v_r >= cur_img.rows-halfpatch_size_)
            break;

        if(isnan(u) || isnan(v)) 
            return false;

        float subpix_x = u-u_r;
        float subpix_y = v-v_r;
        float wTL = (1.0-subpix_x)*(1.0-subpix_y);
        float wTR = subpix_x * (1.0-subpix_y);
        float wBL = (1.0-subpix_x)*subpix_y;
        float wBR = subpix_x * subpix_y;
        float* it_ref = ref_patch;        
        float* it_ref_dv = ref_patch_dv;  
        float* it_weight = grad_weight;   
        float new_chi2 = 0.0;
        Jres.setZero();
        for(int y=0; y<patch_size; ++y)
        {
            uint8_t* it = (uint8_t*) cur_img.data + (v_r+y-halfpatch_size_)*cur_step + u_r-halfpatch_size_;
            for(int x=0; x<patch_size; ++x, ++it, ++it_ref, ++it_ref_dv, ++it_weight)
            {
                float search_pixel = wTL*it[0] + wTR*it[1] + wBL*it[cur_step] + wBR*it[cur_step+1]; 
                float res = search_pixel - *it_ref + mean_diff; 
                Jres[0] -= res*(*it_ref_dv)*(*it_weight);
                Jres[1] -= res*(*it_weight);
                new_chi2 += res*res*(*it_weight);
                if(cur_patch != NULL) {
                    *cur_patch_ptr = search_pixel;
                    ++cur_patch_ptr;
                }
            }
        }
        chi2 = new_chi2;
        update = Hinv * Jres;   
        u += update[0]*dir[0];  
        v += update[0]*dir[1];  
        mean_diff += update[1]; 

        if(update[0]*update[0] < min_update_squared)
        {
            converged=true;
            break;
        }
    }

    if(chi2 > 1000*patch_area) converged=false;

    cur_px_estimate << u, v;
    return converged;
}

bool align2D(
    const cv::Mat& cur_img,
    float* ref_patch_with_border,
    float* ref_patch,
    const int n_iter,
    Vector2d& cur_px_estimate,
    bool no_simd,
    float* cur_patch)
{ 
    const int halfpatch_size_ = 4;
    const int patch_size_ = 8;
    const int patch_area_ = 64;
    bool converged=false;
    float ref_patch_dx[patch_area_];
    float ref_patch_dy[patch_area_];
    Matrix3f H; H.setZero();
    float grad_weight[patch_area_];
    const int ref_step = patch_size_+2;
    float* it_dx = ref_patch_dx;
    float* it_dy = ref_patch_dy;
    float* it_weight = grad_weight;

    Vector3f J;
    for(int y=0; y<patch_size_; ++y)
    {
        float* it = ref_patch_with_border + (y+1)*ref_step + 1;
        for(int x=0; x<patch_size_; ++x, ++it, ++it_dx, ++it_dy, ++it_weight)
        {
            J[0] = 0.5 * (it[1] - it[-1]);
            J[1] = 0.5 * (it[ref_step] - it[-ref_step]);
            J[2] = 1.;
            *it_dx = J[0];
            *it_dy = J[1];
            *it_weight = sqrtf(250.0/(250.0+(J[0]*J[0]+J[1]*J[1])));
            H += J*J.transpose()*(*it_weight);
        }
    }

    for(int i=0;i<3;i++) H(i,i) *= (1+0.001);
    Matrix3f Hinv = H.inverse();
    float u = cur_px_estimate.x();
    float v = cur_px_estimate.y();
    const float min_update_squared = 0.03*0.03;
    const int cur_step = cur_img.step.p[0];
    float mean_diff = 0;
    float chi2 = 0;
    Vector3f update; update.setZero();
    Vector3f Jres; Jres.setZero();
  
    for(int iter = 0; iter<n_iter; ++iter)
    {
        float* cur_patch_ptr = cur_patch;

        int u_r = floor(u);
        int v_r = floor(v);
        if(u_r < halfpatch_size_ || v_r < halfpatch_size_ || u_r >= cur_img.cols-halfpatch_size_ || v_r >= cur_img.rows-halfpatch_size_)
            break;

        if(isnan(u) || isnan(v)) 
            return false;

        float subpix_x = u-u_r;
        float subpix_y = v-v_r;
        float wTL = (1.0-subpix_x)*(1.0-subpix_y);
        float wTR = subpix_x * (1.0-subpix_y);
        float wBL = (1.0-subpix_x)*subpix_y;
        float wBR = subpix_x * subpix_y;
        float* it_ref = ref_patch;
        float* it_ref_dx = ref_patch_dx;
        float* it_ref_dy = ref_patch_dy;
        float* it_weight = grad_weight;
        float new_chi2 = 0.0;
        Jres.setZero();
        for(int y=0; y<patch_size_; ++y)
        {
            uint8_t* it = (uint8_t*) cur_img.data + (v_r+y-halfpatch_size_)*cur_step + u_r-halfpatch_size_;
            for(int x=0; x<patch_size_; ++x, ++it, ++it_ref, ++it_ref_dx, ++it_ref_dy, ++it_weight)
            {
                float search_pixel = wTL*it[0] + wTR*it[1] + wBL*it[cur_step] + wBR*it[cur_step+1];
                float res = search_pixel - (*it_ref) + mean_diff;

                Jres[0] -= res*(*it_ref_dx)*(*it_weight);
                Jres[1] -= res*(*it_ref_dy)*(*it_weight);
                Jres[2] -= res*(*it_weight);

                new_chi2 += res*res*(*it_weight);

                if(cur_patch != NULL) {
                    *cur_patch_ptr = search_pixel;
                    ++cur_patch_ptr;
                }
            }
        }

        chi2 = new_chi2;
        update = Hinv * Jres;
        u += update[0];
        v += update[1];
        mean_diff += update[2];

        if(update[0]*update[0]+update[1]*update[1] < min_update_squared)
        {
            converged=true;
            break;
        }
    }

    if(chi2 > 1000*patch_area_) converged = false;

    cur_px_estimate << u, v;
    return converged;
}

} // namespace feature_alignment
} // namespace vihso
