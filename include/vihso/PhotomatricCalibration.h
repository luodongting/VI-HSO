#pragma once
#include "vihso/global.h"
#include <mutex>
#include <thread>
#include <condition_variable>
namespace vihso {
struct Feature;
class Frame;

class PhotomatricCalibration
{
public:

    PhotomatricCalibration(int patch_size, int width, int height);
   
private:
   
    void startThread();
    void Run();

    std::thread* m_thread;
    
};
} 
