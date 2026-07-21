#include "CloudMerging.h"
#include "Map.h"
#include "MapDrawer.h"
#include "SVIn2ORBWrapper.h"
#include "System.h"

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <utility>

extern SVIn2ORBWrapper *pSVIn2ORBWrapper;

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

    if (mpSystem == nullptr) {
        result.outcome = CloudMergeOutcome::FAILED_EXCEPTION;
        result.detail = "System is unavailable for sea merge";
        return result;
    }

    BackendWriteController *backendController =
        mpSystem->GetBackendWriteController();
    if (backendController == nullptr ||
        !backendController->EnterMergingAndWait()) {
        result.outcome = CloudMergeOutcome::MERGE_SKIPPED_BACKEND_BLOCK;
        result.detail = "BackendWriteController rejected sea merge";
        return result;
    }

    result = RunLandAirCloudMerge(pending);
    if (result.outcome != CloudMergeOutcome::MERGE_COMPLETED_LAND_AIR) {
        mpSystem->ExitBackendMerging();
        return result;
    }

    const MergeFinishResult finishResult =
        mpSystem->FinishBackendMergingAndMaybeEnterReplaying();
    if (finishResult == MergeFinishResult::EXITED_TO_IDLE_LOST) {
        result.outcome = CloudMergeOutcome::MERGED_REPLAY_SKIPPED_LOST;
        result.detail = "sea merge completed while LOST backend block was active";
        return result;
    }
    if (finishResult == MergeFinishResult::EXITED_TO_IDLE_SHUTDOWN) {
        result.outcome = CloudMergeOutcome::MERGED_REPLAY_SKIPPED_SHUTDOWN;
        result.detail = "sea merge completed during backend shutdown";
        return result;
    }
    if (finishResult != MergeFinishResult::ENTERED_REPLAYING) {
        result.outcome = CloudMergeOutcome::FAILED_EXCEPTION;
        result.detail = "BackendWriteController did not enter REPLAYING";
        return result;
    }

    if (::pSVIn2ORBWrapper == nullptr) {
        mpSystem->ExitBackendReplayingAndWaitForDrain();
        result.outcome = CloudMergeOutcome::MERGED_REPLAY_ABORTED_SHUTDOWN;
        result.detail = "SVIn2ORBWrapper is unavailable for replay";
        return result;
    }

    const SVIn2ORBWrapper::ReplayMergeDeferredOutcome replayOutcome =
        ::pSVIn2ORBWrapper->ReplayMergeDeferredBufferWithOutcome();
    if (mpSystem->GetBackendWriteState() == BackendWriteState::REPLAYING) {
        mpSystem->ExitBackendReplayingAndWaitForDrain();
    }
    if (replayOutcome ==
        SVIn2ORBWrapper::ReplayMergeDeferredOutcome::COMPLETED) {
        result.outcome = CloudMergeOutcome::MERGED_REPLAY_COMPLETED;
        result.detail = "sea merge and deferred replay completed";
        return result;
    }
    if (replayOutcome ==
        SVIn2ORBWrapper::ReplayMergeDeferredOutcome::SKIPPED_EMPTY) {
        result.outcome = CloudMergeOutcome::MERGED_REPLAY_COMPLETED;
        result.detail = "sea merge completed with no deferred replay frames";
        return result;
    }
    if (replayOutcome ==
        SVIn2ORBWrapper::ReplayMergeDeferredOutcome::SKIPPED_LOST) {
        result.outcome = CloudMergeOutcome::MERGED_REPLAY_SKIPPED_LOST;
        result.detail = "deferred replay skipped by LOST";
        return result;
    }
    if (replayOutcome ==
        SVIn2ORBWrapper::ReplayMergeDeferredOutcome::SKIPPED_SHUTDOWN) {
        result.outcome = CloudMergeOutcome::MERGED_REPLAY_SKIPPED_SHUTDOWN;
        result.detail = "deferred replay skipped by shutdown";
        return result;
    }
    if (replayOutcome ==
        SVIn2ORBWrapper::ReplayMergeDeferredOutcome::ABORTED_WARNING) {
        result.outcome = CloudMergeOutcome::MERGED_REPLAY_ABORTED_WARNING;
        result.detail = "deferred replay aborted by WARNING";
        return result;
    }
    if (replayOutcome ==
        SVIn2ORBWrapper::ReplayMergeDeferredOutcome::ABORTED_LOST) {
        result.outcome = CloudMergeOutcome::MERGED_REPLAY_ABORTED_LOST;
        result.detail = "deferred replay aborted by LOST";
        return result;
    }
    if (replayOutcome ==
        SVIn2ORBWrapper::ReplayMergeDeferredOutcome::ABORTED_SHUTDOWN) {
        result.outcome = CloudMergeOutcome::MERGED_REPLAY_ABORTED_SHUTDOWN;
        result.detail = "deferred replay aborted by shutdown";
        return result;
    }

    result.outcome = CloudMergeOutcome::FAILED_EXCEPTION;
    result.detail = "deferred replay failed";
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

    mpCurrentCloudMap = pending.pMap;
    mpCurrentEdgeFrontMap = mpAtlas->GetSpecifyMap(
        mpCurrentCloudMap->edgeFrontMapMnId);
    mpCurrentEdgeBackMap = mpAtlas->GetSpecifyMap(
        mpCurrentCloudMap->edgeBackMapMnId);
    if (mpCurrentEdgeFrontMap == nullptr || mpCurrentEdgeBackMap == nullptr) {
        result.outcome = CloudMergeOutcome::MERGE_SKIPPED_EDGE_MAP_MISSING;
        result.detail = "referenced edge map is unavailable";
        result.cleanupAction = CloudMapCleanupAction::DELETE_CLOUD_MAP;
        return result;
    }

    const double timestampTolerance = 0.05;
    const std::vector<KeyFrame *> &edgeFrontKeyFrames =
        mpCurrentEdgeFrontMap->GetAllKeyFrames();
    const std::vector<KeyFrame *> &cloudKeyFrames =
        mpCurrentCloudMap->GetAllKeyFrames();
    std::vector<bool> cloudKeyFrameMatched(cloudKeyFrames.size(), false);
    mpEdgeFrontCloudKeyFrameMatch.clear();

    for (size_t edgeFrontIndex = 0;
         edgeFrontIndex < edgeFrontKeyFrames.size();
         edgeFrontIndex++) {
        KeyFrame *edgeFrontKeyFrame = edgeFrontKeyFrames[edgeFrontIndex];
        if (edgeFrontKeyFrame == nullptr) {
            continue;
        }

        double bestTimestampDelta = timestampTolerance;
        int bestCloudIndex = -1;
        for (size_t cloudIndex = 0;
             cloudIndex < cloudKeyFrames.size();
             cloudIndex++) {
            if (cloudKeyFrameMatched[cloudIndex]) {
                continue;
            }
            KeyFrame *cloudKeyFrame = cloudKeyFrames[cloudIndex];
            if (cloudKeyFrame == nullptr) {
                continue;
            }

            const double timestampDelta = std::abs(
                edgeFrontKeyFrame->mTimeStamp - cloudKeyFrame->mTimeStamp);
            if (timestampDelta < bestTimestampDelta) {
                bestTimestampDelta = timestampDelta;
                bestCloudIndex = static_cast<int>(cloudIndex);
            }
        }

        if (bestCloudIndex >= 0) {
            mpEdgeFrontCloudKeyFrameMatch[static_cast<int>(edgeFrontIndex)] =
                bestCloudIndex;
            cloudKeyFrameMatched[bestCloudIndex] = true;
        }
    }

    if (mpEdgeFrontCloudKeyFrameMatch.empty()) {
        result.outcome = CloudMergeOutcome::MERGE_SKIPPED_MATCH_FAILURE;
        result.detail = "no edge-front to CloudMap timestamp matches";
        result.cleanupAction = CloudMapCleanupAction::DELETE_CLOUD_MAP;
        return result;
    }

    std::vector<int> firstMergeIndexes;
    firstMergeIndexes.reserve(mpEdgeFrontCloudKeyFrameMatch.size());
    for (const auto &match : mpEdgeFrontCloudKeyFrameMatch) {
        firstMergeIndexes.push_back(match.first);
    }
    std::sort(
        firstMergeIndexes.begin(),
        firstMergeIndexes.end(),
        [&edgeFrontKeyFrames](const int left, const int right) {
            return edgeFrontKeyFrames[left]->mTimeStamp <
                   edgeFrontKeyFrames[right]->mTimeStamp;
        });

    std::map<int, int> selectedFirstMergeMatches;
    const size_t maxMatchCount = std::min<size_t>(50, firstMergeIndexes.size());
    for (size_t index = 0; index < maxMatchCount; index++) {
        const int edgeFrontIndex = firstMergeIndexes[index];
        selectedFirstMergeMatches[edgeFrontIndex] =
            mpEdgeFrontCloudKeyFrameMatch[edgeFrontIndex];
    }

    const bool firstSim3Succeeded = ComputeSubmapSim3(
        mpCurrentEdgeFrontMap,
        mpCurrentCloudMap,
        mpEdgeFrontCloudKeyFrameMatch,
        selectedFirstMergeMatches,
        false,
        mgSwEdgeFrontCloud,
        mvpEdgeFrontCloudMatchedKeyPoints,
        mpMapDrawer,
        mbOldUdf,
        mbNewUdf,
        false);
    if (!firstSim3Succeeded && !mbMergeAnyway) {
        result.outcome = CloudMergeOutcome::MERGE_SKIPPED_SIM3_FAILURE;
        result.detail = "edge-front to CloudMap Sim3 failed";
        result.cleanupAction = CloudMapCleanupAction::DELETE_CLOUD_MAP;
        return result;
    }

    CloudMergeMap(
        mpCurrentEdgeFrontMap,
        mpCurrentCloudMap,
        mgSwEdgeFrontCloud,
        mpEdgeFrontCloudKeyFrameMatch,
        mvpEdgeFrontCloudMatchedKeyPoints,
        mpLocalMapper,
        true,
        mbOldUdf,
        mbNewUdf);
    mpAtlas->RemoveBadMaps();

    const std::vector<KeyFrame *> &newEdgeFrontKeyFrames =
        mpCurrentEdgeFrontMap->GetAllKeyFrames();
    const std::vector<KeyFrame *> &edgeBackKeyFrames =
        mpCurrentEdgeBackMap->GetAllKeyFrames();
    std::vector<bool> edgeBackKeyFrameMatched(edgeBackKeyFrames.size(), false);
    mpNewEdgeFrontEdgeBackKeyFrameMatch.clear();

    for (size_t edgeFrontIndex = 0;
         edgeFrontIndex < newEdgeFrontKeyFrames.size();
         edgeFrontIndex++) {
        KeyFrame *edgeFrontKeyFrame = newEdgeFrontKeyFrames[edgeFrontIndex];
        if (edgeFrontKeyFrame == nullptr) {
            continue;
        }

        double bestTimestampDelta = timestampTolerance;
        int bestEdgeBackIndex = -1;
        for (size_t edgeBackIndex = 0;
             edgeBackIndex < edgeBackKeyFrames.size();
             edgeBackIndex++) {
            if (edgeBackKeyFrameMatched[edgeBackIndex]) {
                continue;
            }
            KeyFrame *edgeBackKeyFrame = edgeBackKeyFrames[edgeBackIndex];
            if (edgeBackKeyFrame == nullptr) {
                continue;
            }

            const double timestampDelta = std::abs(
                edgeFrontKeyFrame->mTimeStamp - edgeBackKeyFrame->mTimeStamp);
            if (timestampDelta < bestTimestampDelta) {
                bestTimestampDelta = timestampDelta;
                bestEdgeBackIndex = static_cast<int>(edgeBackIndex);
            }
        }

        if (bestEdgeBackIndex >= 0) {
            mpNewEdgeFrontEdgeBackKeyFrameMatch[
                static_cast<int>(edgeFrontIndex)] = bestEdgeBackIndex;
            edgeBackKeyFrameMatched[bestEdgeBackIndex] = true;
        }
    }

    if (mpNewEdgeFrontEdgeBackKeyFrameMatch.empty()) {
        mpCurrentEdgeBackMap->ResetHaveMerged();
        result.outcome = CloudMergeOutcome::MERGE_SKIPPED_MATCH_FAILURE;
        result.detail = "no edge-front to edge-back timestamp matches";
        result.cleanupAction = CloudMapCleanupAction::DELETE_CLOUD_MAP;
        return result;
    }

    std::vector<int> secondMergeIndexes;
    secondMergeIndexes.reserve(mpNewEdgeFrontEdgeBackKeyFrameMatch.size());
    for (const auto &match : mpNewEdgeFrontEdgeBackKeyFrameMatch) {
        secondMergeIndexes.push_back(match.first);
    }
    std::sort(
        secondMergeIndexes.begin(),
        secondMergeIndexes.end(),
        [&newEdgeFrontKeyFrames](const int left, const int right) {
            return newEdgeFrontKeyFrames[left]->mTimeStamp <
                   newEdgeFrontKeyFrames[right]->mTimeStamp;
        });

    std::map<int, int> selectedSecondMergeMatches;
    size_t secondStart = 0;
    if (secondMergeIndexes.size() > 50) {
        secondStart = secondMergeIndexes.size() - 50;
    }
    for (size_t index = secondStart;
         index < secondMergeIndexes.size();
         index++) {
        const int edgeFrontIndex = secondMergeIndexes[index];
        selectedSecondMergeMatches[edgeFrontIndex] =
            mpNewEdgeFrontEdgeBackKeyFrameMatch[edgeFrontIndex];
    }

    bool secondSim3Succeeded = false;
    if (mbOldUdf || mbNewUdf) {
        mgSwNewCloudEdgeBack = g2o::Sim3(
            Eigen::Matrix3d::Identity(),
            Eigen::Vector3d::Zero(),
            1.0);
        secondSim3Succeeded = true;
    } else {
        secondSim3Succeeded = ComputeSubmapSim3(
            mpCurrentEdgeFrontMap,
            mpCurrentEdgeBackMap,
            mpNewEdgeFrontEdgeBackKeyFrameMatch,
            selectedSecondMergeMatches,
            false,
            mgSwNewCloudEdgeBack,
            mvpNewEdgeFrontEdgeBackMatchedKeyPoints,
            mpMapDrawer,
            false,
            false,
            false);
    }

    if (!secondSim3Succeeded && !mbMergeAnyway) {
        mpCurrentEdgeBackMap->ResetHaveMerged();
        result.outcome = CloudMergeOutcome::MERGE_SKIPPED_SIM3_FAILURE;
        result.detail = "edge-front to edge-back Sim3 failed";
        result.cleanupAction = CloudMapCleanupAction::DELETE_CLOUD_MAP;
        return result;
    }

    CloudMergeMap(
        mpCurrentEdgeFrontMap,
        mpCurrentEdgeBackMap,
        mgSwNewCloudEdgeBack,
        mpNewEdgeFrontEdgeBackKeyFrameMatch,
        mvpNewEdgeFrontEdgeBackMatchedKeyPoints,
        mpLocalMapper,
        true,
        mbOldUdf,
        mbNewUdf);
    if (mpAtlas->GetCurrentMap() == mpCurrentEdgeBackMap) {
        mpAtlas->ChangeMap(mpCurrentEdgeFrontMap);
    }
    mpCurrentEdgeFrontMap->ChangeId(mpCurrentEdgeBackMap->GetId());
    mpCurrentEdgeFrontMap->ResetHaveMerged();
    mpAtlas->SetMapBad(mpCurrentEdgeBackMap);
    mpAtlas->RemoveBadMaps();

    result.outcome = CloudMergeOutcome::MERGE_COMPLETED_LAND_AIR;
    result.detail = "land/air two-stage merge completed";
    result.cleanupAction = CloudMapCleanupAction::DELETE_CLOUD_MAP;
    return result;
}

void CloudMerging::ApplyCloudMapCleanup(
    PendingCloudMap &pending,
    const CloudMapCleanupAction cleanupAction) {
    if (cleanupAction == CloudMapCleanupAction::DELETE_CLOUD_MAP &&
        pending.pMap != nullptr) {
        delete pending.pMap;
        pending.pMap = nullptr;
    }
}

}  // namespace ORB_SLAM3
