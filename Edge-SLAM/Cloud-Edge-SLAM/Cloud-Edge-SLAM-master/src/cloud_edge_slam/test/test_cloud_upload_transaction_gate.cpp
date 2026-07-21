#include "CloudUploadTransactionGate.h"

#include <atomic>
#include <chrono>
#include <future>
#include <thread>

#include <gtest/gtest.h>

namespace ORB_SLAM3 {
namespace {

TEST(CloudUploadTransactionGateTest, BlocksAnotherTransactionUntilLeaseEnds) {
    CloudUploadTransactionGate gate;
    CloudUploadTransactionGate::Lease firstLease = gate.Acquire();
    std::atomic<bool> secondAcquired(false);

    std::future<void> secondTransaction = std::async(
        std::launch::async,
        [&gate, &secondAcquired]() {
            CloudUploadTransactionGate::Lease secondLease = gate.Acquire();
            secondAcquired.store(true, std::memory_order_release);
        });

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_FALSE(secondAcquired.load(std::memory_order_acquire));

    firstLease = CloudUploadTransactionGate::Lease();
    EXPECT_EQ(
        std::future_status::ready,
        secondTransaction.wait_for(std::chrono::seconds(1)));
    EXPECT_TRUE(secondAcquired.load(std::memory_order_acquire));
}

}  // namespace
}  // namespace ORB_SLAM3
