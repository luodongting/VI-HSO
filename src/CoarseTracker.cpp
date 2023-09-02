#include "vihso/CoarseTracker.h"
#include "vihso/frame.h"
#include "vihso/feature.h"
#include "vihso/point.h"

#include "vihso/vikit/math_utils.h"

namespace vihso {

CoarseTracker::CoarseTracker(bool inverse_composition, int max_level, int min_level, int n_iter, bool verbose): 
	m_inverse_composition(inverse_composition), 
    m_max_level(max_level), 
    m_min_level(min_level), 
    m_n_iter(n_iter), 
    m_verbose(verbose),
	m_exposure_rat(0.0),
	m_b(0.0),
	m_level(max_level),
    m_iter(0), 
    m_total_terms(0), 
    m_saturated_terms(0),
	m_huber_thresh(0.0),
	m_outlier_thresh(0.0)
{
	imu_track_w.resize(8,0.0);
	imu_track_w[0] = 1e0;
	imu_track_w[1] = imu_track_w[0]/2.0;
	imu_track_w[2] = imu_track_w[1]/2;
	imu_track_w[3] = imu_track_w[2]/1;
	imu_track_w[4] = imu_track_w[3]/2;
	imu_track_w[5] = imu_track_w[4]/3.5;
	imu_track_w[6] = imu_track_w[5]/4;
	imu_track_w[7] = imu_track_w[6]/4.5;
}

CoarseTracker::~CoarseTracker(){}

size_t CoarseTracker::run(FramePtr ref_frame, FramePtr cur_frame)
{
	if(ref_frame->fts_.empty() && ref_frame->temp_fts_.empty())
		return 0;

	m_ref_frame = ref_frame;
	m_cur_frame = cur_frame;

	m_exposure_rat = m_cur_frame->integralImage_ / m_ref_frame->integralImage_; 
    m_b = 0;

	Sophus::SE3 T2w = m_cur_frame->GetPoseSE3();
	Sophus::SE3 Tw1 = m_ref_frame->GetPoseInverseSE3();
	m_T_cur_ref = T2w * Tw1; 
    
    makeDepthRef();

	for(m_level = m_max_level; m_level >= m_min_level; --m_level)
	{
		std::fill(m_visible_fts.begin(), m_visible_fts.end(), false);

        m_offset_all    = m_max_level-m_level+m_pattern_offset;
        HALF_PATCH_SIZE = staticPatternPadding[m_offset_all]; 
        PATCH_AREA      = staticPatternNum[m_offset_all];

        m_ref_patch_cache = cv::Mat(m_ref_frame->fts_.size()+m_ref_frame->temp_fts_.size(), PATCH_AREA, CV_32F);

        m_visible_fts.resize(m_ref_frame->fts_.size()+m_ref_frame->temp_fts_.size(), false);

        m_jacobian_cache_true.resize(Eigen::NoChange, m_ref_patch_cache.rows*PATCH_AREA);  
        m_jacobian_cache_raw.resize(Eigen::NoChange, m_ref_patch_cache.rows*PATCH_AREA);   
      
        precomputeReferencePatches();
     
        selectRobustFunctionLevel(m_T_cur_ref, m_exposure_rat);
        const double cutoff_error = m_outlier_thresh;
		Vector2d energy_old = computeResiduals(m_T_cur_ref, m_exposure_rat, cutoff_error);

		Matrix7d H; Vector7d b;
        computeGS(H,b); 

		Matrix6d H_imu = Matrix6d::Zero();
		Vector6d b_imu = Vector6d::Zero();
		Vector9d res_RVP = Vector9d::Zero();
		double energy_imu_old = 0;
		IMU::Preintegrated* IMU_preintegrator = NULL;
		if(cur_frame->mbImuPreintegrated && bIMUTrack)
		{
			IMU_preintegrator = cur_frame->mpImuPreintegratedFrame;
		    energy_imu_old = calcIMUResAndGS(H_imu, b_imu, m_T_cur_ref, IMU_preintegrator,res_RVP,energy_old[0],imu_track_w[m_level]);
		}
		if(m_verbose)	{std::cout <<" Eimu="<<energy_imu_old<<" E="<<energy_old[0] << endl;}
		

		float lambda = 0.1;
		for(m_iter=0; m_iter < m_n_iter; m_iter++)
		{
            Matrix7d Hl = H;

			if(bIMUTrack)
			{
			  Hl.block(1,1,6,6) = Hl.block(1,1,6,6) + H_imu;
			  b.block(1,0,6,1) = b.block(1,0,6,1) + b_imu.block(0,0,6,1);
			}

			for(int i=0;i<7;i++) Hl(i,i) *= (1+lambda);
			Vector7d step = Hl.ldlt().solve(b);

			float extrap_fac = 1;
			if(lambda < 0.001) extrap_fac = sqrt(sqrt(0.001 / lambda));
			step *= extrap_fac;

			if(!std::isfinite(step.sum()) || std::isnan(step[0])) step.setZero();

            float new_exposure_rat = m_exposure_rat + step[0];

			SE3 new_T_cur_ref;
			if(!m_inverse_composition)
            {
                new_T_cur_ref = Sophus::SE3::exp(-step.segment<6>(1))*m_T_cur_ref;
            }
			else
			{new_T_cur_ref = m_T_cur_ref*Sophus::SE3::exp(-step.segment<6>(1));}

			Vector2d energy_new = computeResiduals(new_T_cur_ref, new_exposure_rat, cutoff_error); 
			double energy_imu_new = 0.0;
			if(bIMUTrack)
			{
				energy_imu_new = calcIMUResAndGS(H_imu, b_imu, m_T_cur_ref, IMU_preintegrator,res_RVP,energy_new[0],imu_track_w[m_level]);
			}

			bool accept = (energy_new[0]/energy_new[1]) < (energy_old[0]/energy_old[1]);
			if(bIMUTrack)
			{
				accept = (energy_new[0]/energy_new[1] * energy_old[1] +energy_imu_new) < (energy_old[0]+energy_imu_old);
			}

			if(accept)
			{

				if(m_verbose)
		        {
		          cout << cur_frame->id_
                       << "\t level =  " << m_level
                       << "\t It. " << m_iter
		               << "\t Success"
		               << "\t n_meas = " << m_total_terms
		               << "\t rejected = " << m_saturated_terms
		               << "\t new_chi2 = " << energy_new[0]+energy_imu_new
		               << "\t exposure = " << new_exposure_rat
		               << "\t mu = " << lambda
                       << "\t step = " << step.norm()
		               << endl;
		        }

				computeGS(H,b);
				energy_old = energy_new;
				energy_imu_old = energy_imu_new;
				m_exposure_rat = new_exposure_rat; 
				m_T_cur_ref = new_T_cur_ref;       
                lambda *= 0.5;


                
			}
			else 
			{

				if(m_verbose)
		        {
                    cout << cur_frame->id_
                         << "\t level =  " << m_level
                         << "\t It. " << m_iter
                         << "\t Failure"
                         << "\t n_meas = " << m_total_terms
                         << "\t rejected = " << m_saturated_terms
                         << "\t new_chi2 = " << energy_new[0]+energy_imu_new
                         << "\t exposure = " << new_exposure_rat
                         << "\t mu = " << lambda
                         << "\t step = " << step.norm()
                         << endl;
		        }

                lambda *= 4;
				if(lambda < 0.001) 
                    lambda = 0.001;


			}


            if(!(step.norm() > 1e-4))
            {
                if(m_verbose)
				{
					printf("inc too small, break!\n");
				}
                break;
            }
		}
	}

	Sophus::SE3 T1w = m_ref_frame->GetPoseSE3();
	m_cur_frame->SetPose(m_T_cur_ref * T1w);

    while(!m_ref_frame->m_exposure_finish && m_cur_frame->m_pc != NULL){cv::waitKey(1);}
    m_cur_frame->m_exposure_time = m_exposure_rat*m_ref_frame->m_exposure_time;

    if(m_exposure_rat > 0.99 && m_exposure_rat < 1.01) m_cur_frame->m_exposure_time = m_ref_frame->m_exposure_time;

    return float(m_total_terms) / PATCH_AREA;
}

size_t CoarseTracker::runForRelocalization(FramePtr ref_frame, FramePtr cur_frame, Sophus::SE3& Tcw_curframe)
{
	if(ref_frame->fts_.empty() && m_ref_frame->temp_fts_.empty())
		return 0;

	m_ref_frame = ref_frame;
	m_cur_frame = cur_frame;


	m_exposure_rat = m_cur_frame->integralImage_ / m_ref_frame->integralImage_;
    m_b = 0;

	Sophus::SE3 T2w = m_cur_frame->GetPoseSE3();
	Sophus::SE3 Tw1 = m_ref_frame->GetPoseInverseSE3();
	m_T_cur_ref = T2w * Tw1;

    
    makeDepthRef();


	for(m_level = m_max_level; m_level >= m_min_level; --m_level)
	{
		std::fill(m_visible_fts.begin(), m_visible_fts.end(), false);

        m_offset_all    = m_max_level-m_level+m_pattern_offset;
        HALF_PATCH_SIZE = staticPatternPadding[m_offset_all];
        PATCH_AREA      = staticPatternNum[m_offset_all];

        m_ref_patch_cache = cv::Mat(m_ref_frame->nObs(), PATCH_AREA, CV_32F);
        m_visible_fts.resize(m_ref_frame->nObs(), false);
        m_jacobian_cache_true.resize(Eigen::NoChange, m_ref_patch_cache.rows*PATCH_AREA);
        m_jacobian_cache_raw.resize(Eigen::NoChange, m_ref_patch_cache.rows*PATCH_AREA);

        precomputeReferencePatches();
        selectRobustFunctionLevel(m_T_cur_ref, m_exposure_rat);
        const double cutoff_error = m_outlier_thresh;
		Vector2d energy_old = computeResiduals(m_T_cur_ref, m_exposure_rat, cutoff_error);
		Matrix7d H; Vector7d b;
        computeGS(H,b);

		float lambda = 0.1;
		for(m_iter=0; m_iter < m_n_iter; m_iter++)
		{
			Matrix7d Hl = H;
			for(int i=0;i<7;i++) Hl(i,i) *= (1+lambda);
			Vector7d step = Hl.ldlt().solve(b);
			float extrap_fac = 1;
			if(lambda < 0.001) extrap_fac = sqrt(sqrt(0.001 / lambda));
			step *= extrap_fac;

			if(!std::isfinite(step.sum()) || std::isnan(step[0])) step.setZero();

            float new_exposure_rat = m_exposure_rat + step[0];
			SE3 new_T_cur_ref;
			if(!m_inverse_composition)            
                new_T_cur_ref = Sophus::SE3::exp(-step.segment<6>(1))*m_T_cur_ref;				
			else
				new_T_cur_ref = m_T_cur_ref*Sophus::SE3::exp(-step.segment<6>(1));

			Vector2d energy_new = computeResiduals(new_T_cur_ref, new_exposure_rat, cutoff_error); 
			if((energy_new[0]/energy_new[1]) < (energy_old[0]/energy_old[1]))
			{
				if(m_verbose)
		        {
		          cout << cur_frame->id_
                       << "\t level =  " << m_level
                       << "\t It. " << m_iter
		               << "\t Success"
		               << "\t n_meas = " << m_total_terms
		               << "\t rejected = " << m_saturated_terms
		               << "\t new_chi2 = " << energy_new
		               << "\t exposure = " << new_exposure_rat
		               << "\t mu = " << lambda
                       << "\t step = " << step.norm()
		               << endl;
		        }

				computeGS(H,b);
				energy_old = energy_new;
				m_exposure_rat = new_exposure_rat;
				m_T_cur_ref = new_T_cur_ref;
                lambda *= 0.5;
			}
			else
			{
				if(m_verbose)
		        {
                    cout << cur_frame->id_
                         << "\t level =  " << m_level
                         << "\t It. " << m_iter
                         << "\t Failure"
                         << "\t n_meas = " << m_total_terms
                         << "\t rejected = " << m_saturated_terms
                         << "\t new_chi2 = " << energy_new
                         << "\t exposure = " << new_exposure_rat
                         << "\t mu = " << lambda
                         << "\t step = " << step.norm()
                         << endl;
		        }

                lambda *= 4;
				if(lambda < 0.001) 
                    lambda = 0.001;
			}
            if(!(step.norm() > 1e-4))
            {
                if(m_verbose)
                    printf("inc too small, break!\n");
                break;
            }
		}
	}

	Sophus::SE3 T1w = m_ref_frame->GetPoseSE3();
	Tcw_curframe = m_T_cur_ref * T1w;

    return float(m_total_terms) / PATCH_AREA;
}


void CoarseTracker::makeDepthRef()
{
    m_pt_ref.resize(m_ref_frame->nObs(), -1);

    size_t feature_counter = 0;
	list<Feature*> lfts = m_ref_frame->GetFeatures();
    for(auto it_ft=lfts.begin(); it_ft!=lfts.end(); ++it_ft, ++feature_counter)
    {
        if((*it_ft)->point == NULL) continue;

        Vector3d p_host = (*it_ft)->point->hostFeature_->f * (1.0/(*it_ft)->point->GetIdist());	
		SE3 Twh = (*it_ft)->point->hostFeature_->frame->GetPoseInverseSE3();
        SE3 T_r_h = m_ref_frame->GetPoseSE3() * Twh;
        Vector3d p_ref = T_r_h*p_host;                                                              
        if(p_ref[2] < 0.00001) continue;

        m_pt_ref[feature_counter] = p_ref.norm();
    }
}

Vector2d CoarseTracker::computeResiduals( const SE3& T_cur_ref,  
                                        float exposure_rat,     
                                        double cutoff_error,   
                                        float b)
{
	if(m_inverse_composition)
		m_jacobian_cache_true = exposure_rat*m_jacobian_cache_raw;

	const cv::Mat& cur_img = m_cur_frame->img_pyr_.at(m_level);
	const int stride = cur_img.cols;
    const int border = HALF_PATCH_SIZE+1;
    const float scale = 1.0f/(1<<m_level);
    const Vector3d ref_pos(m_ref_frame->GetCameraCenter());

    const double fxl = m_ref_frame->cam_->focal_length().x()*scale;
    const double fyl = m_ref_frame->cam_->focal_length().y()*scale;

    float setting_huberTH = m_huber_thresh;

    const float max_energy = 2*setting_huberTH*cutoff_error-setting_huberTH*setting_huberTH;

    const int pattern_offset = m_offset_all;

    m_buf_jacobian.clear();
    m_buf_weight.clear();
    m_buf_error.clear();
    m_total_terms = m_saturated_terms = 0;

    float E = 0;

    m_color_cur.clear(); m_color_ref.clear();

    size_t feature_counter = 0;
    std::vector<bool>::iterator visiblity_it = m_visible_fts.begin();
	list<Feature*> lfts = m_ref_frame->GetFeatures();
    for(list<Feature*>::iterator it_ft=lfts.begin(); it_ft!=lfts.end(); ++it_ft, ++feature_counter, ++visiblity_it)
    {
    	if(!*visiblity_it) continue; 

        double dist = m_pt_ref[feature_counter]; 
		if(dist < 0) continue;

        Vector3d xyz_ref((*it_ft)->f*dist);
        Vector3d xyz_cur(T_cur_ref * xyz_ref);
        if(xyz_cur[2] < 0) continue;

        Vector2f uv_cur_0(m_cur_frame->cam_->world2cam(xyz_cur).cast<float>());
        Vector2f uv_cur_pyr(uv_cur_0 * scale);
        float u_cur = uv_cur_pyr[0];
        float v_cur = uv_cur_pyr[1];
        int u_cur_i = floorf(u_cur);
        int v_cur_i = floorf(v_cur);

        if(u_cur_i-border < 0 || v_cur_i-border < 0 || u_cur_i+border >= cur_img.cols || v_cur_i+border >= cur_img.rows)
            continue;

        Matrix<double,2,6> frame_jac;
        if(!m_inverse_composition)
    		Frame::jacobian_xyz2uv(xyz_cur, frame_jac);

        float subpix_u_cur = u_cur-u_cur_i;
        float subpix_v_cur = v_cur-v_cur_i;
        float w_cur_tl = (1.0-subpix_u_cur) * (1.0-subpix_v_cur);
        float w_cur_tr = subpix_u_cur * (1.0-subpix_v_cur);
        float w_cur_bl = (1.0-subpix_u_cur) * subpix_v_cur;
        float w_cur_br = subpix_u_cur * subpix_v_cur;

        float* ref_patch_cache_ptr = reinterpret_cast<float*>(m_ref_patch_cache.data) + PATCH_AREA*feature_counter;
        size_t pixel_counter = 0;

        for(int n=0; n<PATCH_AREA; ++n, ++ref_patch_cache_ptr, ++pixel_counter)
    	{
            uint8_t* cur_img_ptr = (uint8_t*)cur_img.data + (v_cur_i + staticPattern[pattern_offset][n][1])*stride + u_cur_i + staticPattern[pattern_offset][n][0];//patch每个点坐标
    		float cur_color = w_cur_tl*cur_img_ptr[0] 
    						+ w_cur_tr*cur_img_ptr[1] 
    						+ w_cur_bl*cur_img_ptr[stride] 
    						+ w_cur_br*cur_img_ptr[stride+1];
    		if(!std::isfinite(cur_color)) continue;

    		float residual = cur_color - (exposure_rat*(*ref_patch_cache_ptr) + b); 

    		float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual); 

    		if(fabs(residual) > cutoff_error && m_level < m_max_level) 
    		{
    			E += max_energy;
    			m_total_terms++;
    			m_saturated_terms++;
    		}
    		else
    		{ 
                if(m_level == m_max_level)
                    E += hw *residual*residual;
                else

                    E += hw *residual*residual*(2-hw);  
				m_total_terms++;

				if(!m_inverse_composition)
				{
					float dx = 0.5f * ((w_cur_tl*cur_img_ptr[1]       + w_cur_tr*cur_img_ptr[2]        + w_cur_bl*cur_img_ptr[stride+1] + w_cur_br*cur_img_ptr[stride+2])
                                  	  -(w_cur_tl*cur_img_ptr[-1]      + w_cur_tr*cur_img_ptr[0]        + w_cur_bl*cur_img_ptr[stride-1] + w_cur_br*cur_img_ptr[stride]));
	        		float dy = 0.5f * ((w_cur_tl*cur_img_ptr[stride]  + w_cur_tr*cur_img_ptr[1+stride] + w_cur_bl*cur_img_ptr[stride*2] + w_cur_br*cur_img_ptr[stride*2+1])
	                                  -(w_cur_tl*cur_img_ptr[-stride] + w_cur_tr*cur_img_ptr[1-stride] + w_cur_bl*cur_img_ptr[0]        + w_cur_br*cur_img_ptr[1]));
        			Vector6d J_T = dx*frame_jac.row(0)*fxl + dy*frame_jac.row(1)*fyl;  

                    double J_e = -(*ref_patch_cache_ptr);
                    Vector7d J; J[0] = J_e;
                    J.segment<6>(1) = J_T;

        			m_buf_jacobian.push_back(J); 
        			m_buf_weight.push_back(hw);
        			m_buf_error.push_back(residual);
				}
				else
				{
					Vector6d J_T(m_jacobian_cache_true.col(feature_counter*PATCH_AREA + pixel_counter));

                    double J_e = -(*ref_patch_cache_ptr);

                    Vector7d J; J[0] = J_e;     
                    J.segment<6>(1) = J_T;

					m_buf_jacobian.push_back(J);  
        			m_buf_weight.push_back(hw);
        			m_buf_error.push_back(residual);
				}
    		}

            m_color_cur.push_back(cur_color);
            m_color_ref.push_back(*ref_patch_cache_ptr);
    	}
    }

	Vector2d rs;
	rs[0] = E;
	rs[1] = m_total_terms;
	return rs; 
}

double CoarseTracker::calcIMUResAndGS(	Matrix6d& H_out, Vector6d& b_out,	
										const SE3& T_cur_ref,				
										IMU::Preintegrated* IMU_preintegrator,	
										Vector9d &res_RVP, 		
										double PointEnergy, 	
										double imu_track_weight	
									)
{
	SE3 T_WCi = m_ref_frame->GetPoseInverseSE3();	
	Matrix4d M_WBi = m_ref_frame->GetImuPose();
	SE3 T_WBi(M_WBi.block<3,3>(0,0), M_WBi.block<3,1>(0,3));	
	Matrix3d R_WBi = T_WBi.rotation_matrix();	
    Vector3d t_WBi = T_WBi.translation();		

	SE3 T_ref_cur = T_cur_ref.inverse();		
	SE3 T_WCj = T_WCi*T_ref_cur;				
	SE3 T_cb = m_ref_frame->mImuCalib.SE3_Tcb;	
	SE3 T_bc = m_ref_frame->mImuCalib.SE3_Tbc;	
	SE3 T_WBj = T_WCj*T_cb;						
	Matrix3d R_WBj = T_WBj.rotation_matrix();	
    Vector3d t_WBj = T_WBj.translation();		
    
    double dt = IMU_preintegrator->dT;
    H_out = Matrix6d::Zero();
    b_out = Vector6d::Zero();
    if(dt>0.5)
	{
		return 0;
    }
    Vector3d Gz = Eigen::Vector3d(0,0,-IMU::GRAVITY_VALUE);

	IMU::Bias Bias1 = m_ref_frame->mImuBias;
    Matrix3d deltaR = IMU_preintegrator->GetDeltaRotation(Bias1);
    Vector3d deltaV = IMU_preintegrator->GetDeltaVelocity(Bias1);
    Vector3d deltaP = IMU_preintegrator->GetDeltaPosition(Bias1);


	Matrix3d res_R = deltaR.transpose() * R_WBi.transpose() * R_WBj;				
    Vector3d res_r = SO3(res_R).log();	
	
	Vector3d Vwbi = m_ref_frame->mVw;	
	Vector3d Vwbj = Vwbi + Gz*dt + R_WBi*deltaV;	
	Vector3d res_V = R_WBi.transpose()*(Vwbj-Vwbi-Gz*dt) - deltaV;					
	
	Vector3d res_P = R_WBi.transpose()*(t_WBj-t_WBi-Vwbi*dt-0.5*Gz*dt*dt) - deltaP;	

	res_RVP.block(0,0,3,1) = res_r;
    res_RVP.block(3,0,3,1) = Vector3d::Zero();
    res_RVP.block(6,0,3,1) = res_P;

	Matrix9d Cov = IMU_preintegrator->C.block<9,9>(0,0);
    double res = imu_track_weight*imu_track_weight * res_RVP.transpose()*Cov.inverse()*res_RVP;

    Matrix3d J_resR_Rj = IMU::InverseRightJacobianSO3(res_r);	
	Matrix3d J_resV_Vj = R_WBi.transpose();						
	Matrix3d J_resP_Pj = R_WBi.transpose()*R_WBj;				


    Matrix6d J_imu1 = Matrix6d::Zero();
    J_imu1.block(0,0,3,3) = J_resP_Pj;
    J_imu1.block(3,3,3,3) = J_resR_Rj;

    Matrix6d Weight = Matrix6d::Zero();
    Weight.block(0,0,3,3) = Cov.block(6,6,3,3);
    Weight.block(3,3,3,3) = Cov.block(0,0,3,3);
    Matrix6d Weight2 = Matrix6d::Zero();
    for(int i=0;i<6;++i)
	{
		Weight2(i,i) = Weight(i,i);
    }
    Weight = Weight2.inverse();
    Weight *=(imu_track_weight*imu_track_weight);
    
    Vector6d b_1 = Vector6d::Zero();
    b_1.block(0,0,3,1) = res_P;
    b_1.block(3,0,3,1) = res_r;
    
	SE3 T_temp = T_bc * T_WCj.inverse();
	Matrix6d J_rel = -1*T_temp.Adj();

    Matrix6d J_xi_tw_th = SE3(T_WCi).Adj();	
    Matrix6d J_xi_r_l = T_cur_ref.Adj().inverse();	

    Matrix6d J_2 = Matrix6d::Zero();
    J_2 = J_imu1*J_rel*J_xi_tw_th*J_xi_r_l;

    H_out = J_2.transpose()*Weight*J_2;
    b_out = J_2.transpose()*Weight*b_1;
    
	#define SCALE_XI_ROT 1.0f
	#define SCALE_XI_TRANS 1.0f
    H_out.block<6,3>(0,0) *= SCALE_XI_TRANS;
    H_out.block<6,3>(0,3) *= SCALE_XI_ROT;
    H_out.block<3,6>(0,0) *= SCALE_XI_TRANS;
    H_out.block<3,6>(3,0) *= SCALE_XI_ROT;
    
    b_out.segment<3>(0) *= SCALE_XI_TRANS;
    b_out.segment<3>(3) *= SCALE_XI_ROT;
  
    return res;
}

void CoarseTracker::addEdgePhotometricG2O(g2o::SparseOptimizer* optimizer,		
											vihso::VertexPhotometricPose* VT21,		
											vihso::VertexExposureRatio* VExpRatio, 	
											double cutoff_error)					
{
	if(m_inverse_composition)
		m_jacobian_cache_true = VExpRatio->estimate()*m_jacobian_cache_raw;

	const cv::Mat& cur_img = m_cur_frame->img_pyr_.at(m_level);
	const int stride = cur_img.cols;
    const int border = HALF_PATCH_SIZE+1;
    const float scale = 1.0f/(1<<m_level);

    const double fxl = m_ref_frame->cam_->focal_length().x()*scale; 
    const double fyl = m_ref_frame->cam_->focal_length().y()*scale;

    float setting_huberTH = m_huber_thresh;
    const float max_energy = 2*setting_huberTH*cutoff_error-setting_huberTH*setting_huberTH;	

    const int pattern_offset = m_offset_all;
    size_t feature_counter = 0;
    std::vector<bool>::iterator visiblity_it = m_visible_fts.begin();
	list<Feature*> lfts = m_ref_frame->GetFeatures();
    for(list<Feature*>::iterator it_ft=lfts.begin(); it_ft!=lfts.end(); ++it_ft, ++feature_counter, ++visiblity_it)	
    {	
    	if(!*visiblity_it)	continue;
		double dist = m_pt_ref[feature_counter]; 
		if(dist < 0)	continue;
		Vector3d xyz_ref((*it_ft)->f*dist);

		float* ref_patch_cache_ptr = reinterpret_cast<float*>(m_ref_patch_cache.data) + PATCH_AREA*feature_counter;	//ref_patch灰度
		size_t pixel_counter = 0;
    
        for(int n=0; n<PATCH_AREA; ++n, ++ref_patch_cache_ptr, ++pixel_counter)
    	{
			int patternRow = staticPattern[pattern_offset][n][1];
			int patternCol = staticPattern[pattern_offset][n][0];

			vihso::EdgePhotometric* edge = new vihso::EdgePhotometric();
			edge->setVertex(0,VT21);
    		edge->setVertex(1,VExpRatio);
			edge->setMeasurement(*ref_patch_cache_ptr);
			edge->setConfig(xyz_ref, cur_img, stride, border, m_level, scale, fxl, fyl, setting_huberTH, cutoff_error, max_energy, m_cur_frame, m_inverse_composition, patternRow, patternCol);

			if(edge->isDepthPositive())	
				continue;
			if(!edge->computeConfig())
				continue;;

			if(edge->error()(0) > cutoff_error*5)
				continue;
			
			if(m_inverse_composition)
			{
				Vector6d J_inv(m_jacobian_cache_true.col(feature_counter*PATCH_AREA + pixel_counter));
				edge->setJacobian_inv(J_inv);
			}
				
			g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
			edge->setRobustKernel(rk);
			rk->setDelta(setting_huberTH);
			
			optimizer->addEdge(edge);
    	}
    }
}


void CoarseTracker::precomputeReferencePatches()
{
	const int border = HALF_PATCH_SIZE+1;                       
    const cv::Mat& ref_img = m_ref_frame->img_pyr_[m_level];   
    const int stride = ref_img.cols;                            
    const float scale = 1.0f/(1<<m_level);                     
    const Vector3d ref_pos = m_ref_frame->GetCameraCenter();
    const double fxl = m_ref_frame->cam_->focal_length().x()*scale;
    const double fyl = m_ref_frame->cam_->focal_length().y()*scale;
    const int pattern_offset = m_offset_all;   

    std::vector<bool>::iterator visiblity_it = m_visible_fts.begin();  
    size_t feature_counter = 0;                                     

	list<Feature*> lfts = m_ref_frame->GetFeatures();
    for(list<Feature*>::iterator ft_it=lfts.begin(); ft_it!=lfts.end(); ++ft_it, ++visiblity_it, ++feature_counter)
    {
        if((*ft_it)->point == NULL)
            continue;

        float u_ref = (*ft_it)->px[0]*scale;
        float v_ref = (*ft_it)->px[1]*scale;
        int u_ref_i = floorf(u_ref);
        int v_ref_i = floorf(v_ref);
        if(u_ref_i-border < 0 || v_ref_i-border < 0 || u_ref_i+border >= ref_img.cols || v_ref_i+border >= ref_img.rows)
            continue;

        *visiblity_it = true;
        
        Matrix<double,2,6> frame_jac;
        if(m_inverse_composition)
        {
        	
            double dist = m_pt_ref[feature_counter];
            if(dist < 0) continue;
            Vector3d xyz_ref((*ft_it)->f*dist);

        	Frame::jacobian_xyz2uv(xyz_ref, frame_jac);
        }

  
        float subpix_u_ref = u_ref-u_ref_i;
        float subpix_v_ref = v_ref-v_ref_i;
        float w_ref_tl = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);
        float w_ref_tr = subpix_u_ref * (1.0-subpix_v_ref);
        float w_ref_bl = (1.0-subpix_u_ref) * subpix_v_ref;
        float w_ref_br = 1.0-(w_ref_tl+w_ref_tr+w_ref_bl);

        size_t pixel_counter = 0;   
        float* cache_ptr = reinterpret_cast<float*>(m_ref_patch_cache.data) + PATCH_AREA*feature_counter;   
        for(int n=0; n<PATCH_AREA; ++n, ++cache_ptr, ++pixel_counter)
    	{
         
            uint8_t* ref_img_ptr = (uint8_t*)ref_img.data + (v_ref_i + staticPattern[pattern_offset][n][1])*stride + u_ref_i + staticPattern[pattern_offset][n][0];

    		*cache_ptr = w_ref_tl*ref_img_ptr[0] 
    				   + w_ref_tr*ref_img_ptr[1] 
    				   + w_ref_bl*ref_img_ptr[stride] 
    				   + w_ref_br*ref_img_ptr[stride+1];

    		if(m_inverse_composition)
    		{   
    			float dx = 0.5f * ((w_ref_tl*ref_img_ptr[1]       + w_ref_tr*ref_img_ptr[2]        + w_ref_bl*ref_img_ptr[stride+1] + w_ref_br*ref_img_ptr[stride+2])
                                  -(w_ref_tl*ref_img_ptr[-1]      + w_ref_tr*ref_img_ptr[0]        + w_ref_bl*ref_img_ptr[stride-1] + w_ref_br*ref_img_ptr[stride]));
            	float dy = 0.5f * ((w_ref_tl*ref_img_ptr[stride]  + w_ref_tr*ref_img_ptr[1+stride] + w_ref_bl*ref_img_ptr[stride*2] + w_ref_br*ref_img_ptr[stride*2+1])
                              	  -(w_ref_tl*ref_img_ptr[-stride] + w_ref_tr*ref_img_ptr[1-stride] + w_ref_bl*ref_img_ptr[0]        + w_ref_br*ref_img_ptr[1]));
                
            	m_jacobian_cache_raw.col(feature_counter*PATCH_AREA + pixel_counter) = dx*frame_jac.row(0)*fxl + dy*frame_jac.row(1)*fyl; 
    		}
    	}
    }
}


void CoarseTracker::computeGS(Matrix7d& H_out, Vector7d& b_out)
{
	assert(m_buf_jacobian.size() == m_buf_weight.size());

    m_acc7.initialize();    

    b_out.setZero();
	for(size_t i=0; i<m_buf_jacobian.size(); ++i)
	{

        m_acc7.updateSingleWeighted(m_buf_jacobian[i][0],
                                    m_buf_jacobian[i][1],
                                    m_buf_jacobian[i][2],
                                    m_buf_jacobian[i][3],
                                    m_buf_jacobian[i][4],
                                    m_buf_jacobian[i][5],
                                    m_buf_jacobian[i][6],
                                    m_buf_weight[i], 0);

        b_out.noalias() -= m_buf_jacobian[i]*m_buf_error[i]*m_buf_weight[i];  
	}

    m_acc7.finish();
    H_out = m_acc7.H.cast<double>();
}


void CoarseTracker::selectRobustFunctionLevel(const SE3& T_cur_ref, float exposure_rat, float b)
{ 
    const cv::Mat& cur_img = m_cur_frame->img_pyr_.at(m_level); 
    const int stride = cur_img.cols;                            
    const int border = HALF_PATCH_SIZE+1;                       
    const float scale = 1.0f/(1<<m_level);                      
    const Vector3d ref_pos(m_ref_frame->GetCameraCenter());

    const int pattern_offset = m_offset_all;                    


    std::vector<float> errors;

    size_t feature_counter = 0; 
    std::vector<bool>::iterator visiblity_it = m_visible_fts.begin();
	list<Feature*> lfts = m_ref_frame->GetFeatures();
    for(list<Feature*>::iterator it_ft=lfts.begin(); it_ft!=lfts.end(); ++it_ft, ++feature_counter, ++visiblity_it)
    {
        if(!*visiblity_it) continue;


        double dist = m_pt_ref[feature_counter]; if(dist < 0) continue;
        
        Vector3d xyz_ref((*it_ft)->f*dist);     
        Vector3d xyz_cur(T_cur_ref * xyz_ref);  

        if(xyz_cur[2] < 0) continue;

        Vector2f uv_cur_pyr(m_cur_frame->cam_->world2cam(xyz_cur).cast<float>() * scale);
        float u_cur = uv_cur_pyr[0];
        float v_cur = uv_cur_pyr[1];
        int u_cur_i = floorf(u_cur);
        int v_cur_i = floorf(v_cur);

        if(u_cur_i-border < 0 || v_cur_i-border < 0 || u_cur_i+border >= cur_img.cols || v_cur_i+border >= cur_img.rows)
            continue;


        float subpix_u_cur = u_cur-u_cur_i;
        float subpix_v_cur = v_cur-v_cur_i;
        float w_cur_tl = (1.0-subpix_u_cur) * (1.0-subpix_v_cur);
        float w_cur_tr = subpix_u_cur * (1.0-subpix_v_cur);
        float w_cur_bl = (1.0-subpix_u_cur) * subpix_v_cur;
        float w_cur_br = subpix_u_cur * subpix_v_cur;

        float* ref_patch_cache_ptr = reinterpret_cast<float*>(m_ref_patch_cache.data) + PATCH_AREA*feature_counter; 

        for(int n=0; n<PATCH_AREA; ++n, ++ref_patch_cache_ptr)
        {

            uint8_t* cur_img_ptr = (uint8_t*)cur_img.data + (v_cur_i + staticPattern[pattern_offset][n][1])*stride + u_cur_i + staticPattern[pattern_offset][n][0];
            float cur_color = w_cur_tl*cur_img_ptr[0] 
                            + w_cur_tr*cur_img_ptr[1] 
                            + w_cur_bl*cur_img_ptr[stride] 
                            + w_cur_br*cur_img_ptr[stride+1];   
            float residual = cur_color - (exposure_rat*(*ref_patch_cache_ptr) + b); 
            errors.push_back(fabsf(residual));
        }
    }

    if(errors.size() < 30) 
    {
        m_huber_thresh = 5.2;
        m_outlier_thresh = 100;
        return;
    }

    float residual_median = hso::getMedian(errors);
    vector<float> absolute_deviation; 
    for(size_t i=0; i<errors.size(); ++i)
        absolute_deviation.push_back(fabs(errors[i]-residual_median));

    float standard_deviation = 1.4826*hso::getMedian(absolute_deviation);


    m_huber_thresh = residual_median + standard_deviation; 
    m_outlier_thresh = 3*m_huber_thresh; 
    if(m_outlier_thresh < 10) m_outlier_thresh = 10;
}

Vector2f CoarseTracker::lineFit(vector<float>& cur, vector<float>& ref, float a, float b)
{
    float sxx=0, syy=0, sxy=0, sx=0, sy=0, sw=0;
    for(size_t i = 0; i < cur.size(); ++i)
    {
        if(cur[i] < 5 || ref[i] < 5 || cur[i] > 250 || ref[i] > 250)
            continue;

        float res = cur[i] - a*ref[i] - b;

        const float cutoff_thresh = m_level == m_max_level? 80 : 25;
        const float weight_aff = fabsf(res) < cutoff_thresh? fabsf(res) < 8.0f? 1.0 : 8.0f / fabsf(res) : 0;  
        sxx += ref[i]*ref[i]*weight_aff; 
        syy += cur[i]*cur[i]*weight_aff;
        sx += ref[i]*weight_aff;
        sy += cur[i]*weight_aff;
        sw += weight_aff;
    }

    float aff_a = sqrtf((syy - sy*sy/sw) / (sxx - sx*sx/sw));

    return Vector2f(aff_a, (sy - aff_a*sx)/sw);
}
} //namespace vihso