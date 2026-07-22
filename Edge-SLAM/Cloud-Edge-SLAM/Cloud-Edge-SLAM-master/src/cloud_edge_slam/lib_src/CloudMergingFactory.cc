#include "CloudMergingFactory.h"

#include "CloudMerging.h"

#if defined(CLOUD_EDGE_MERGER_SLA)
#define CLOUD_EDGE_MERGER_MODE_COUNT_1 1
#else
#define CLOUD_EDGE_MERGER_MODE_COUNT_1 0
#endif

#if defined(CLOUD_EDGE_MERGER_SEA_LEGACY)
#define CLOUD_EDGE_MERGER_MODE_COUNT_2 1
#else
#define CLOUD_EDGE_MERGER_MODE_COUNT_2 0
#endif

#if defined(CLOUD_EDGE_MERGER_LAND_AIR_LEGACY)
#define CLOUD_EDGE_MERGER_MODE_COUNT_3 1
#else
#define CLOUD_EDGE_MERGER_MODE_COUNT_3 0
#endif

#if CLOUD_EDGE_MERGER_MODE_COUNT_1 + CLOUD_EDGE_MERGER_MODE_COUNT_2 + \
        CLOUD_EDGE_MERGER_MODE_COUNT_3 != 1
#error "Exactly one CloudMerging target mode must be selected"
#endif

namespace ORB_SLAM3 {

RuntimeEnvironment GetFactoryDefaultRuntimeEnvironment() {
#if defined(CLOUD_EDGE_MERGER_SEA_LEGACY)
    return RuntimeEnvironment::SEA;
#else
    return RuntimeEnvironment::LAND;
#endif
}

CloudMerging *CreateCloudMerging(
    const CloudMergingFactoryArguments &arguments) {
#if defined(CLOUD_EDGE_MERGER_SLA)
    return new CloudMerging(
        arguments.pSystem,
        arguments.pAtlas,
        arguments.pDatabase,
        arguments.pVocabulary,
        arguments.fixScale,
        arguments.activeLoopClosing,
        arguments.cloudMergeEnabled,
        arguments.mergeAnyway,
        arguments.pMapDrawer,
        arguments.pFrameDrawer,
        arguments.oldUdf,
        arguments.newUdf,
        arguments.runtimeEnvironment);
#elif defined(CLOUD_EDGE_MERGER_SEA_LEGACY)
    return new CloudMerging(
        arguments.pSystem,
        arguments.pAtlas,
        arguments.pDatabase,
        arguments.pVocabulary,
        arguments.fixScale,
        arguments.activeLoopClosing,
        arguments.cloudMergeEnabled,
        arguments.mergeAnyway,
        arguments.pMapDrawer,
        arguments.pFrameDrawer,
        arguments.oldUdf,
        arguments.newUdf);
#else
    return new CloudMerging(
        arguments.pAtlas,
        arguments.pDatabase,
        arguments.pVocabulary,
        arguments.fixScale,
        arguments.activeLoopClosing,
        arguments.cloudMergeEnabled,
        arguments.mergeAnyway,
        arguments.pMapDrawer,
        arguments.pFrameDrawer,
        arguments.oldUdf,
        arguments.newUdf);
#endif
}

}  // namespace ORB_SLAM3
