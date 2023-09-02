#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include "vihso/frame_handler_mono.h"
#include "vihso/depth_filter.h"
#include "vihso/frame.h"
#include "vihso/feature.h"
#include "vihso/point.h"
#include "vihso/map.h"
#include <mutex>


namespace vihso
{

class SystemNode;
class FrameHandlerMono;
class DepthFilter;
class Frame;
class Feature;
class Seed;
class Point;
class Map;


class LocalMapping
{
public:
    LocalMapping(vihso::SystemNode* pSys, Map* pMap, bool bInertial);
};

} //namespace LocalMapping

#endif // LOCALMAPPING_H
