#ifndef VIHSO_FEATURE_H_
#define VIHSO_FEATURE_H_

#include <vihso/frame.h>

namespace vihso {
struct Feature
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	enum FeatureType {CORNER, EDGELET, GRADIENT};

	FeatureType type;     
	Frame* frame;         
	Vector2d px;         
	Vector3d f;          
	int level;           
	Point* point;         
	Vector2d grad;      

	vector<double> outputs;
	vector<double> radiances;
	vector<double> outputs_grad;
	vector<double> rad_mean;
	Feature* m_prev_feature = NULL;  
	Feature* m_next_feature = NULL;
	bool m_added = false;  
	bool m_non_point = false;
	Matrix2d rotate_plane;
	bool imuBAOutlier = false;	

	Feature(Frame* _frame, const Vector2d& _px, int _level) :
	type(CORNER),
	frame(_frame),
	px(_px),
	f(frame->cam_->cam2world(px)),
	level(_level),
	point(NULL),
	grad(1.0,0.0)
	{
	}

	Feature(Frame* _frame, const Vector2d& _px, const Vector3d& _f, int _level) : 
	type(CORNER),
	frame(_frame),
	px(_px),
	f(_f),
	level(_level),
	point(NULL),
	grad(1.0,0.0)
	{
	}

	Feature(Frame* _frame, Point* _point, const Vector2d& _px, const Vector3d& _f, int _level) :  
	type(CORNER),
	frame(_frame),
	px(_px),
	f(_f),
	level(_level),
	point(_point),
	grad(1.0,0.0)
	{
	}

	Feature(Frame* _frame, const Vector2d& _px, const Vector2d& _grad, int _level) :
	type(EDGELET),
	frame(_frame),
	px(_px),
	f(frame->cam_->cam2world(px)),
	level(_level),
	point(NULL),
	grad(_grad)
	{
	}

	Feature(Frame* _frame, Point* _point, const Vector2d& _px, const Vector3d& _f, const Vector2d& _grad, int _level) : 
	type(EDGELET),
	frame(_frame),
	px(_px),
	f(_f),
	level(_level),
	point(_point),
	grad(_grad)
	{
	}

	Feature(Frame* _frame, const Vector2d& _px, int _level, FeatureType _type) :
	type(_type),
	frame(_frame),
	px(_px),
	f(frame->cam_->cam2world(px)),
	level(_level),
	point(NULL),
	grad(1.0,0.0)
	{
	}

	Feature(Frame* _frame, Point* _point, const Vector2d& _px, int _level, FeatureType _type) : 
	type(_type),
	frame(_frame),
	px(_px),
	f(frame->cam_->cam2world(px)),
	level(_level),
	point(_point),
	grad(1.0,0.0)
	{
	}

};

} // namespace vihso

#endif // VIHSO_FEATURE_H_
