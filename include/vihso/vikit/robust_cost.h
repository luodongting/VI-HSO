#pragma once
#include <vector>
#include <stdlib.h>
#include <memory>
namespace hso {
namespace robust_cost {
class ScaleEstimator
{
public:
	virtual ~ScaleEstimator() {};
	virtual float compute(std::vector<float>& errors) const = 0;
};
typedef std::shared_ptr<ScaleEstimator> ScaleEstimatorPtr;
class UnitScaleEstimator : public ScaleEstimator
{
public:
	UnitScaleEstimator() {}
	virtual ~UnitScaleEstimator() {}
	virtual float compute(std::vector<float>& errors) const { return 1.0f; };
};
class TDistributionScaleEstimator : public ScaleEstimator
{
public:
	TDistributionScaleEstimator(const float dof = DEFAULT_DOF);
	virtual ~TDistributionScaleEstimator() {};
	virtual float compute(std::vector<float>& errors) const;
	static const float DEFAULT_DOF;
	static const float INITIAL_SIGMA;
protected:
	float dof_;
	float initial_sigma_;
};
class MADScaleEstimator : public ScaleEstimator
{
public:
	MADScaleEstimator() {};
	virtual ~MADScaleEstimator() {};
	virtual float compute(std::vector<float>& errors) const;
private:
  	static const float NORMALIZER;;
};
class NormalDistributionScaleEstimator : public ScaleEstimator
{
public:
	NormalDistributionScaleEstimator() {};
	virtual ~NormalDistributionScaleEstimator() {};
	virtual float compute(std::vector<float>& errors) const;
private:
};
class WeightFunction
{
public:
	virtual ~WeightFunction() {};
	virtual float value(const float& x) const = 0;
	virtual void configure(const float& param) {};
};
typedef std::shared_ptr<WeightFunction> WeightFunctionPtr;
class UnitWeightFunction : public WeightFunction
{
public:
	UnitWeightFunction() {};
	virtual ~UnitWeightFunction() {};
	virtual float value(const float& x) const { return 1.0f; };
};
class TukeyWeightFunction : public WeightFunction
{
public:
	TukeyWeightFunction(const float b = DEFAULT_B);	
	virtual ~TukeyWeightFunction() {};
	virtual float value(const float& x) const;
	virtual void configure(const float& param);
  	static const float DEFAULT_B;
private:
  	float b_square;
};
class TDistributionWeightFunction : public WeightFunction
{
public:
	TDistributionWeightFunction(const float dof = DEFAULT_DOF);
	virtual ~TDistributionWeightFunction() {};
	virtual float value(const float& x) const;
	virtual void configure(const float& param);
	static const float DEFAULT_DOF;
private:
	float dof_;
	float normalizer_;
};
class HuberWeightFunction : public WeightFunction
{
public:
	HuberWeightFunction(const float k = DEFAULT_K);
	virtual ~HuberWeightFunction() {};
	virtual float value(const float& x) const;
	virtual void configure(const float& param);
	static const float DEFAULT_K;
private:
  	float k;
};
}
}
