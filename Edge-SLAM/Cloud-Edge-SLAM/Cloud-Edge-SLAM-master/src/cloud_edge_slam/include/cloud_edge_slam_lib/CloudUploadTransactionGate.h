#ifndef CLOUD_UPLOAD_TRANSACTION_GATE_H
#define CLOUD_UPLOAD_TRANSACTION_GATE_H

#include <mutex>

namespace ORB_SLAM3 {

// Holds the sole committed cloud-upload transaction from sendGoal through the
// completion of every CloudMap ticket created by that Action result.
class CloudUploadTransactionGate {
public:
    class Lease {
    public:
        Lease();
        explicit Lease(std::mutex &mutex);

        Lease(const Lease &) = delete;
        Lease &operator=(const Lease &) = delete;
        Lease(Lease &&) noexcept = default;
        Lease &operator=(Lease &&) noexcept = default;

    private:
        std::unique_lock<std::mutex> mLock;
    };

    Lease Acquire();

private:
    std::mutex mMutex;
};

}  // namespace ORB_SLAM3

#endif  // CLOUD_UPLOAD_TRANSACTION_GATE_H
