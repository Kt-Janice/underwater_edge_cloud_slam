#include "CloudMerging.h"

#include <unistd.h>

#include <utility>

namespace ORB_SLAM3 {

CloudMerging::CloudMerging(
    System *pSystem,
    Atlas *pAtlas, KeyFrameDatabase *pDB, ORBVocabulary *pVoc,
    const bool bFixScale, const bool bActiveLC,
    const bool bWork, const bool bMergeAnyway,
    MapDrawer *pMapDrawer, FrameDrawer *pFrameDrawer,
    const bool bOldUdf, const bool bNewUdf)
    : CloudMerging(
          pSystem,
          pAtlas,
          pDB,
          pVoc,
          bFixScale,
          bActiveLC,
          bWork,
          bMergeAnyway,
          pMapDrawer,
          pFrameDrawer,
          bOldUdf,
          bNewUdf,
          RuntimeEnvironment::LAND) {
}

CloudMerging::CloudMerging(
    System *pSystem,
    Atlas *pAtlas, KeyFrameDatabase *pDB, ORBVocabulary *pVoc,
    const bool bFixScale, const bool bActiveLC,
    const bool bWork, const bool bMergeAnyway,
    MapDrawer *pMapDrawer, FrameDrawer *pFrameDrawer,
    const bool bOldUdf, const bool bNewUdf,
    RuntimeEnvironment runtimeEnvironment)
    : mbResetRequested(false),
      mbResetActiveMapRequested(false),
      mpMapToReset(nullptr),
      mbFinishRequested(false),
      mbFinished(true),
      mbRunning(false),
      mpAtlas(pAtlas),
      mpTracker(nullptr),
      mpSystem(pSystem),
      mpKeyFrameDB(pDB),
      mpORBVocabulary(pVoc),
      mpLocalMapper(nullptr),
      mpLoopClosing(nullptr),
      mpMapDrawer(pMapDrawer),
      mpFrameDrawer(pFrameDrawer),
      mpCurrentCloudMap(nullptr),
      mpCurrentEdgeFrontMap(nullptr),
      mpCurrentEdgeBackMap(nullptr),
      mpCurrentKF(nullptr),
      mpLastCurrentKF(nullptr),
      mpMatchedKF(nullptr),
      mbWork(bWork),
      mbMergeAnyway(bMergeAnyway),
      mbOldUdf(bOldUdf),
      mbNewUdf(bNewUdf),
      mbMergeDetected(false),
      mnMergeNumCoincidences(0),
      mnMergeNumNotFound(0),
      mpMergeLastCurrentKF(nullptr),
      mpMergeMatchedKF(nullptr),
      mLastLoopKFid(0),
      mbRunningGBA(false),
      mbFinishedGBA(true),
      mbStopGBA(false),
      mpThreadGBA(nullptr),
      mbFixScale(bFixScale),
      mnFullBAIdx(0),
      mnCloudMergeDebugOutputIndex(0),
      mbActiveCM(bActiveLC),
      mRuntimeEnvironment(runtimeEnvironment) {
    mnCovisibilityConsistencyTh = 3;
    mstrFolderSubTraj = "SubTrajectories/";
    mnNumCorrection = 0;
    mnCorrectionGBA = 0;
}

void CloudMerging::SetTracker(Tracking *pTracker) {
    mpTracker = pTracker;
}

void CloudMerging::SetLocalMapper(LocalMapping *pLocalMapper) {
    mpLocalMapper = pLocalMapper;
}

void CloudMerging::SetLoopClosing(LoopClosing *pLoopClosing) {
    mpLoopClosing = pLoopClosing;
}

void CloudMerging::SetCloudMergeDebugOutputDir(
    const std::string &debugOutputDir) {
    std::lock_guard<std::mutex> lock(mMutexCloudMergeDebugOutputDir);
    mCloudMergeDebugOutputDir = debugOutputDir;
}

void CloudMerging::Run(const bool bOnline) {
    static_cast<void>(bOnline);
    {
        std::lock_guard<std::mutex> lock(mMutexFinish);
        mbFinished = false;
        mbRunning = false;
    }

    while (!CheckFinish()) {
        PendingCloudMap pending;
        bool hasPendingCloudMap = false;

        {
            std::unique_lock<std::mutex> lock(mMutexCloudQueue);
            if (!mlPendingCloudMapQueue.empty()) {
                pending = std::move(mlPendingCloudMapQueue.front());
                mlPendingCloudMapQueue.pop_front();
                hasPendingCloudMap = true;
            }
        }

        if (!hasPendingCloudMap) {
            usleep(5000);
            continue;
        }

        mbRunning = true;
        CloudMergeExecutionResult executionResult;
        try {
            if (IsSeaEnvironment(mRuntimeEnvironment)) {
                executionResult = RunSeaCloudMerge(pending);
            } else {
                executionResult = RunLandAirCloudMerge(pending);
            }
        } catch (const std::exception &exception) {
            executionResult.outcome = CloudMergeOutcome::FAILED_EXCEPTION;
            executionResult.detail = exception.what();
            executionResult.cleanupAction = CloudMapCleanupAction::NONE;
        } catch (...) {
            executionResult.outcome = CloudMergeOutcome::FAILED_EXCEPTION;
            executionResult.detail = "unknown CloudMerging exception";
            executionResult.cleanupAction = CloudMapCleanupAction::NONE;
        }

        ApplyCloudMapCleanup(pending, executionResult.cleanupAction);
        CompleteCloudMap(
            pending.pTicket,
            executionResult.outcome,
            executionResult.detail);
        mbRunning = false;
    }

    CancelPendingCloudMaps(
        CloudMergeOutcome::CANCELLED_SHUTDOWN,
        "cloud merger is shutting down");
    SetFinish();
}

void CloudMerging::InsertCloudMap(Map *pMap) {
    static_cast<void>(InsertCloudMapWithTicket(pMap));
}

CloudMergeTicketPtr CloudMerging::InsertCloudMapWithTicket(Map *pMap) {
    CloudMergeTicketPtr ticket = std::make_shared<CloudMergeTicket>(0);
    bool cancelTicket = false;

    {
        std::unique_lock<std::mutex> lock(mMutexCloudQueue);
        ticket = std::make_shared<CloudMergeTicket>(++mnNextCloudMergeSequence);
        if (mbSlaShutdownRequested) {
            cancelTicket = true;
        } else {
            PendingCloudMap pending;
            pending.pMap = pMap;
            pending.pTicket = ticket;
            mlPendingCloudMapQueue.push_back(std::move(pending));
        }
    }

    if (cancelTicket) {
        CompleteCloudMap(
            ticket,
            CloudMergeOutcome::CANCELLED_SHUTDOWN,
            "cloud merger is shutting down");
    }

    return ticket;
}

bool CloudMerging::WaitForCompletion(
    const CloudMergeTicketPtr &ticket,
    CloudMergeResult &result) {
    if (!ticket) {
        return false;
    }

    return ticket->Wait(result);
}

void CloudMerging::RequestFinish() {
    {
        std::lock_guard<std::mutex> lock(mMutexFinish);
        mbFinishRequested = true;
    }
    {
        std::lock_guard<std::mutex> lock(mMutexCloudQueue);
        mbSlaShutdownRequested = true;
    }
    CancelPendingCloudMaps(
        CloudMergeOutcome::CANCELLED_SHUTDOWN,
        "cloud merger finish requested");
}

bool CloudMerging::CheckFinish() {
    std::lock_guard<std::mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void CloudMerging::SetFinish() {
    std::lock_guard<std::mutex> lock(mMutexFinish);
    mbFinished = true;
    mbRunning = false;
}

bool CloudMerging::isFinished() {
    std::lock_guard<std::mutex> lock(mMutexFinish);
    return mbFinished;
}

bool CloudMerging::isRunning() {
    std::lock_guard<std::mutex> lock(mMutexFinish);
    return mbRunning;
}

Map *CloudMerging::GetEdgeFrontMap() {
    return mpCurrentEdgeFrontMap;
}

Map *CloudMerging::GetEdgeBackMap() {
    return mpCurrentEdgeBackMap;
}

Map *CloudMerging::GetCloudMap() {
    return mpCurrentCloudMap;
}

void CloudMerging::CompleteCloudMap(
    const CloudMergeTicketPtr &ticket,
    CloudMergeOutcome outcome,
    const std::string &detail) {
    if (!ticket) {
        return;
    }

    CloudMergeResult result;
    result.sequence = ticket->GetSequence();
    result.outcome = outcome;
    result.detail = detail;
    ticket->TryComplete(result);
}

void CloudMerging::CancelPendingCloudMaps(
    CloudMergeOutcome outcome,
    const std::string &detail) {
    std::list<PendingCloudMap> pendingCloudMaps;

    {
        std::unique_lock<std::mutex> lock(mMutexCloudQueue);
        pendingCloudMaps.swap(mlPendingCloudMapQueue);
    }

    for (const PendingCloudMap &pending : pendingCloudMaps) {
        CompleteCloudMap(pending.pTicket, outcome, detail);
    }
}

CloudMergeExecutionResult CloudMerging::RunSeaCloudMerge(
    PendingCloudMap &pending) {
    CloudMergeExecutionResult result;
    if (pending.pMap == nullptr) {
        result.outcome = CloudMergeOutcome::MERGE_SKIPPED_INVALID_MAP;
        result.detail = "null CloudMap";
        return result;
    }

    result.outcome = CloudMergeOutcome::FAILED_EXCEPTION;
    result.detail = "sea strategy has not been migrated";
    return result;
}

CloudMergeExecutionResult CloudMerging::RunLandAirCloudMerge(
    PendingCloudMap &pending) {
    CloudMergeExecutionResult result;
    if (pending.pMap == nullptr) {
        result.outcome = CloudMergeOutcome::MERGE_SKIPPED_INVALID_MAP;
        result.detail = "null CloudMap";
        return result;
    }

    result.outcome = CloudMergeOutcome::FAILED_EXCEPTION;
    result.detail = "land/air strategy has not been migrated";
    return result;
}

void CloudMerging::ApplyCloudMapCleanup(
    PendingCloudMap &pending,
    const CloudMapCleanupAction cleanupAction) {
    static_cast<void>(pending);
    static_cast<void>(cleanupAction);
}

}  // namespace ORB_SLAM3
