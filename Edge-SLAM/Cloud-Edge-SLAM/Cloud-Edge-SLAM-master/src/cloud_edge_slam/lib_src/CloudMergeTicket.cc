#include "CloudMergeTicket.h"

namespace ORB_SLAM3 {

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
