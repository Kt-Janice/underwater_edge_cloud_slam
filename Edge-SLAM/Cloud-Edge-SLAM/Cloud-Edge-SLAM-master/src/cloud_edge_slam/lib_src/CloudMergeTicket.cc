#include "CloudMergeTicket.h"

namespace ORB_SLAM3 {

const char *CloudMergeOutcomeToString(const CloudMergeOutcome outcome) {
    switch (outcome) {
        case CloudMergeOutcome::PENDING: {
            return "PENDING";
        }
        case CloudMergeOutcome::MERGED_REPLAY_COMPLETED: {
            return "MERGED_REPLAY_COMPLETED";
        }
        case CloudMergeOutcome::MERGED_REPLAY_SKIPPED_LOST: {
            return "MERGED_REPLAY_SKIPPED_LOST";
        }
        case CloudMergeOutcome::MERGED_REPLAY_SKIPPED_SHUTDOWN: {
            return "MERGED_REPLAY_SKIPPED_SHUTDOWN";
        }
        case CloudMergeOutcome::MERGED_REPLAY_ABORTED_WARNING: {
            return "MERGED_REPLAY_ABORTED_WARNING";
        }
        case CloudMergeOutcome::MERGED_REPLAY_ABORTED_LOST: {
            return "MERGED_REPLAY_ABORTED_LOST";
        }
        case CloudMergeOutcome::MERGED_REPLAY_ABORTED_SHUTDOWN: {
            return "MERGED_REPLAY_ABORTED_SHUTDOWN";
        }
        case CloudMergeOutcome::MERGE_COMPLETED_LAND_AIR: {
            return "MERGE_COMPLETED_LAND_AIR";
        }
        case CloudMergeOutcome::MERGE_NOT_IMPLEMENTED_AIR: {
            return "MERGE_NOT_IMPLEMENTED_AIR";
        }
        case CloudMergeOutcome::MERGE_SKIPPED_BACKEND_BLOCK: {
            return "MERGE_SKIPPED_BACKEND_BLOCK";
        }
        case CloudMergeOutcome::MERGE_SKIPPED_SIM3_FAILURE: {
            return "MERGE_SKIPPED_SIM3_FAILURE";
        }
        case CloudMergeOutcome::MERGE_SKIPPED_EDGE_MAP_MISSING: {
            return "MERGE_SKIPPED_EDGE_MAP_MISSING";
        }
        case CloudMergeOutcome::MERGE_SKIPPED_MATCH_FAILURE: {
            return "MERGE_SKIPPED_MATCH_FAILURE";
        }
        case CloudMergeOutcome::MERGE_SKIPPED_INVALID_MAP: {
            return "MERGE_SKIPPED_INVALID_MAP";
        }
        case CloudMergeOutcome::PARTIAL_FAILURE_AFTER_CORRECTION: {
            return "PARTIAL_FAILURE_AFTER_CORRECTION";
        }
        case CloudMergeOutcome::CANCELLED_SHUTDOWN: {
            return "CANCELLED_SHUTDOWN";
        }
        case CloudMergeOutcome::FAILED_EXCEPTION: {
            return "FAILED_EXCEPTION";
        }
    }
    return "UNKNOWN_OUTCOME";
}

CloudMergeTicket::CloudMergeTicket(const std::uint64_t sequence)
    : mSequence(sequence) {
    mResult.sequence = sequence;
}

std::uint64_t CloudMergeTicket::GetSequence() const {
    return mSequence;
}

bool CloudMergeTicket::TryComplete(const CloudMergeResult &result) {
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (mbCompleted) {
            return false;
        }

        mResult = result;
        mResult.sequence = mSequence;
        mbCompleted = true;
    }

    mCondition.notify_all();
    return true;
}

bool CloudMergeTicket::Wait(CloudMergeResult &result) const {
    std::unique_lock<std::mutex> lock(mMutex);
    mCondition.wait(lock, [this]() {
        return mbCompleted;
    });
    result = mResult;
    return true;
}

}  // namespace ORB_SLAM3
