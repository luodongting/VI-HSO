#include "vihso/PhotomatricCalibration.h"
#include "vihso/frame.h"
#include "vihso/feature.h"
#include "vihso/config.h"
#include "vihso/point.h"
#include "vihso/vikit/math_utils.h"
namespace vihso{
PhotomatricCalibration::PhotomatricCalibration(int patch_size, int width, int height)
{
    startThread();
}
void PhotomatricCalibration::startThread()
{
    m_thread = new thread(&vihso::PhotomatricCalibration::Run, this);
}
void PhotomatricCalibration::Run()
{

}

} //namespace vihso