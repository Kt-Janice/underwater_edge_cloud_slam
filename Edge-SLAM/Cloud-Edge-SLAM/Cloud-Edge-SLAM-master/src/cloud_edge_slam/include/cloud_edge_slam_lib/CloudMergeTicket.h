#ifndef CLOUD_MERGE_TICKET_H
#define CLOUD_MERGE_TICKET_H

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace ORB_SLAM3 {

class CloudMerging;
class CloudMergeTicketTestAccess;

enum class CloudMergeOutcome {
    PENDING,
    MERGED_REPLAY_COMPLETED,
    MERGED_REPLAY_SKIPPED_LOST,
    MERGED_REPLAY_SKIPPED_SHUTDOWN,
    MERGED_REPLAY_ABORTED_WARNING,
    MERGED_REPLAY_ABORTED_LOST,
    MERGED_REPLAY_ABORTED_SHUTDOWN,
    MERGE_COMPLETED_LAND_AIR,
    MERGE_SKIPPED_BACKEND_BLOCK,
    MERGE_SKIPPED_SIM3_FAILURE,
    MERGE_SKIPPED_EDGE_MAP_MISSING,
    MERGE_SKIPPED_MATCH_FAILURE,
    MERGE_SKIPPED_INVALID_MAP,
    PARTIAL_FAILURE_AFTER_CORRECTION,
    CANCELLED_SHUTDOWN,
    FAILED_EXCEPTION
};

struct CloudMergeResult {
    std::uint64_t sequence = 0;
    CloudMergeOutcome outcome = CloudMergeOutcome::PENDING;
    std::string detail;
};

class CloudMergeTicket {
public:
    explicit CloudMergeTicket(std::uint64_t sequence);

    std::uint64_t GetSequence() const;

private:
    friend class CloudMerging;
    friend class CloudMergeTicketTestAccess;

    bool TryComplete(const CloudMergeResult &result);
    bool Wait(CloudMergeResult &result) const;

    const std::uint64_t mSequence;
    bool mbCompleted = false;
    CloudMergeResult mResult;
    mutable std::mutex mMutex;
    mutable std::condition_variable mCondition;
};

using CloudMergeTicketPtr = std::shared_ptr<CloudMergeTicket>;

}  // namespace ORB_SLAM3

#endif
