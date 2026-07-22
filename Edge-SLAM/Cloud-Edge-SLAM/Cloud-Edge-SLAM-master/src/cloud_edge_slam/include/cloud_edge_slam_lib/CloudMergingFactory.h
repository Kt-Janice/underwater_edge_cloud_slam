#ifndef CLOUD_MERGING_FACTORY_H
#define CLOUD_MERGING_FACTORY_H

#include "RuntimeEnvironment.h"
#include "ORBVocabulary.h"

namespace ORB_SLAM3 {

class Atlas;
class CloudMerging;
class FrameDrawer;
class KeyFrameDatabase;
class MapDrawer;
class System;

struct CloudMergingFactoryArguments {
    System *pSystem = nullptr;
    Atlas *pAtlas = nullptr;
    KeyFrameDatabase *pDatabase = nullptr;
    ORBVocabulary *pVocabulary = nullptr;
    bool fixScale = false;
    bool activeLoopClosing = false;
    bool cloudMergeEnabled = false;
    bool mergeAnyway = false;
    MapDrawer *pMapDrawer = nullptr;
    FrameDrawer *pFrameDrawer = nullptr;
    bool oldUdf = false;
    bool newUdf = false;
    RuntimeEnvironment runtimeEnvironment = RuntimeEnvironment::LAND;
};

RuntimeEnvironment GetFactoryDefaultRuntimeEnvironment();

CloudMerging *CreateCloudMerging(
    const CloudMergingFactoryArguments &arguments);

}  // namespace ORB_SLAM3

#endif
