#include "CloudUploadTransactionGate.h"

namespace ORB_SLAM3 {

CloudUploadTransactionGate::Lease::Lease() = default;

CloudUploadTransactionGate::Lease::Lease(std::mutex &mutex)
    : mLock(mutex) {
}

CloudUploadTransactionGate::Lease CloudUploadTransactionGate::Acquire() {
    return Lease(mMutex);
}

}  // namespace ORB_SLAM3
