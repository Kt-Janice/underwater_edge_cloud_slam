#include "CloudMerging.h"

#include <utility>

namespace ORB_SLAM3 {

namespace {

enum class CloudMapCleanupAction {
    NONE,
    DELETE_CLOUD_MAP,
    PRESERVE_FOR_ATLAS,
    PRESERVE_AFTER_PARTIAL_CORRECTION
};

struct CloudMergeExecutionResult {
    CloudMergeOutcome outcome = CloudMergeOutcome::FAILED_EXCEPTION;
    std::string detail;
    CloudMapCleanupAction cleanupAction = CloudMapCleanupAction::NONE;
};

CloudMergeExecutionResult MakeUnsupportedResult(
    RuntimeEnvironment runtimeEnvironment) {
    CloudMergeExecutionResult result;

    if (runtimeEnvironment == RuntimeEnvironment::SEA) {
        result.detail = "sea strategy has not been selected";
    } else {
        result.detail = "land/air strategy has not been selected";
    }

    return result;
}

}  // namespace

CloudMerging::CloudMerging(
    System *pSystem,
    Atlas *pAtlas, KeyFrameDatabase *pDB, ORBVocabulary *pVoc,
    const bool bFixScale, const bool bActiveLC,
    const bool bWork, const bool bMergeAnyway,
    MapDrawer *pMapDrawer, FrameDrawer *pFrameDrawer,
    const bool bOldUdf, const bool bNewUdf,
    RuntimeEnvironment runtimeEnvironment)
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
          bNewUdf) {
    mRuntimeEnvironment = runtimeEnvironment;
}

CloudMergeTicketPtr CloudMerging::InsertCloudMapWithTicket(Map *pMap) {
    std::unique_lock<std::mutex> lock(mMutexCloudQueue);

    if (mbSlaShutdownRequested) {
        CloudMergeTicketPtr ticket = std::make_shared<CloudMergeTicket>(
            ++mnNextCloudMergeSequence);
        CompleteCloudMap(
            ticket,
            CloudMergeOutcome::CANCELLED_SHUTDOWN,
            "cloud merger is shutting down");
        return ticket;
    }

    CloudMergeTicketPtr ticket = std::make_shared<CloudMergeTicket>(
        ++mnNextCloudMergeSequence);
    PendingCloudMap pending;
    pending.pMap = pMap;
    pending.pTicket = ticket;
    mlPendingCloudMapQueue.push_back(std::move(pending));
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

}  // namespace ORB_SLAM3
