#define CLOUD_EDGE_UPLOAD_LIFECYCLE_ONLY
#include "../src/cloud_edge.cpp"

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace cloud_edge_slam_sla {
namespace {

struct FakeTicketResult {
    bool accepted = false;
};

TEST(CloudUploadLifecycleTest, ShutdownWaitsForActiveUploadToDrain) {
    std::shared_ptr<UploadLifecycleState> state =
        std::make_shared<UploadLifecycleState>();
    std::atomic<bool> drainFinished(false);
    std::thread drainThread;

    {
        ActiveUpload activeUpload = TryStartUpload(state);
        ASSERT_TRUE(activeUpload);
        EXPECT_EQ(1U, GetActiveUploadCountForTesting(state));

        StopAcceptingUploads(state);
        EXPECT_FALSE(IsAcceptingUploadsForTesting(state));

        ActiveUpload rejectedUpload = TryStartUpload(state);
        EXPECT_FALSE(rejectedUpload);
        EXPECT_EQ(1U, GetActiveUploadCountForTesting(state));

        drainThread = std::thread([state, &drainFinished]() {
            WaitForActiveUploads(state);
            drainFinished.store(true, std::memory_order_release);
        });

        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        EXPECT_FALSE(drainFinished.load(std::memory_order_acquire));
    }

    drainThread.join();
    EXPECT_TRUE(drainFinished.load(std::memory_order_acquire));
    EXPECT_EQ(0U, GetActiveUploadCountForTesting(state));
}

TEST(CloudUploadLifecycleTest, RegistrationAfterShutdownWakesImmediately) {
    std::shared_ptr<UploadLifecycleState> state =
        std::make_shared<UploadLifecycleState>();
    StopAcceptingUploads(state);

    std::atomic<int> wakeCount(0);
    const std::size_t callbackId = RegisterUploadWakeCallback(
        state,
        [&wakeCount]() {
            wakeCount.fetch_add(1, std::memory_order_acq_rel);
        });

    EXPECT_EQ(0U, callbackId);
    EXPECT_EQ(1, wakeCount.load(std::memory_order_acquire));
}

TEST(CloudUploadLifecycleTest, CancellationWakeReleasesRegisteredWaiter) {
    std::shared_ptr<UploadLifecycleState> state =
        std::make_shared<UploadLifecycleState>();
    std::mutex mutex;
    std::condition_variable condition;
    bool cancelled = false;
    const std::size_t callbackId = RegisterUploadWakeCallback(
        state,
        [&mutex, &condition, &cancelled]() {
            {
                std::lock_guard<std::mutex> lock(mutex);
                cancelled = true;
            }
            condition.notify_all();
        });
    ASSERT_NE(0U, callbackId);

    std::future<void> waiter = std::async(
        std::launch::async,
        [&mutex, &condition, &cancelled]() {
            std::unique_lock<std::mutex> lock(mutex);
            condition.wait(lock, [&cancelled]() {
                return cancelled;
            });
        });

    WakeUploadWaiters(state);
    EXPECT_EQ(
        std::future_status::ready,
        waiter.wait_for(std::chrono::seconds(1)));
    UnregisterUploadWakeCallback(state, callbackId);
}

TEST(CloudUploadLifecycleTest, DrainCancelsGoalSentAfterInitialCancellation) {
    std::shared_ptr<UploadLifecycleState> state =
        std::make_shared<UploadLifecycleState>();
    std::shared_ptr<ActiveUpload> activeUpload =
        std::make_shared<ActiveUpload>(TryStartUpload(state));
    ASSERT_TRUE(*activeUpload);

    std::promise<void> initialCancellationFinished;
    const std::shared_future<void> allowLateSend =
        initialCancellationFinished.get_future().share();
    std::atomic<bool> remoteGoalActive(false);
    std::atomic<int> cancellationCount(0);
    std::thread lateSender(
        [activeUpload, allowLateSend, &remoteGoalActive]() {
            allowLateSend.wait();
            remoteGoalActive.store(true, std::memory_order_release);
        });
    activeUpload.reset();

    const std::function<void()> cancelGoals =
        [&remoteGoalActive, &cancellationCount]() {
            cancellationCount.fetch_add(1, std::memory_order_acq_rel);
            remoteGoalActive.store(false, std::memory_order_release);
        };
    StopAcceptingUploads(state);
    cancelGoals();
    initialCancellationFinished.set_value();

    DrainUploadsAndCancelLateGoals(state, cancelGoals);
    lateSender.join();

    EXPECT_EQ(2, cancellationCount.load(std::memory_order_acquire));
    EXPECT_FALSE(remoteGoalActive.load(std::memory_order_acquire));
}

TEST(CloudUploadLifecycleTest, ShutdownWaitsForGuardedCallbackToReturn) {
    std::shared_ptr<UploadLifecycleState> state =
        std::make_shared<UploadLifecycleState>();
    std::promise<void> callbackEntered;
    std::future<void> callbackEnteredFuture = callbackEntered.get_future();
    std::promise<void> releasePromise;
    const std::shared_future<void> releaseCallback =
        releasePromise.get_future().share();

    std::future<bool> callback = std::async(
        std::launch::async,
        [state, &callbackEntered, releaseCallback]() {
            return RunWithActiveUpload(
                state,
                [&callbackEntered, releaseCallback]() {
                    callbackEntered.set_value();
                    releaseCallback.wait();
                });
        });
    const std::future_status callbackEnteredStatus =
        callbackEnteredFuture.wait_for(std::chrono::seconds(1));
    if (callbackEnteredStatus != std::future_status::ready) {
        releasePromise.set_value();
        callback.wait();
        FAIL() << "guarded callback did not start within one second";
        return;
    }

    StopAcceptingUploads(state);
    std::future<void> drain = std::async(
        std::launch::async,
        [state]() {
            WaitForActiveUploads(state);
        });
    EXPECT_EQ(
        std::future_status::timeout,
        drain.wait_for(std::chrono::milliseconds(20)));

    releasePromise.set_value();
    EXPECT_EQ(
        std::future_status::ready,
        drain.wait_for(std::chrono::seconds(1)));
    EXPECT_TRUE(callback.get());
}

TEST(CloudUploadLifecycleTest, GuardedCallbackDoesNotRunAfterShutdown) {
    std::shared_ptr<UploadLifecycleState> state =
        std::make_shared<UploadLifecycleState>();
    StopAcceptingUploads(state);
    bool callbackRan = false;

    const bool started = RunWithActiveUpload(
        state,
        [&callbackRan]() {
            callbackRan = true;
        });

    EXPECT_FALSE(started);
    EXPECT_FALSE(callbackRan);
}

TEST(CloudUploadLifecycleTest, FailedLostDispatchReturnsUploadToAvailable) {
    EXPECT_EQ(
        LostRecoveryUploadState::AVAILABLE,
        FinalizedLostRecoveryUploadState(false));
}

TEST(CloudUploadLifecycleTest, StartedLostDispatchConsumesUpload) {
    EXPECT_EQ(
        LostRecoveryUploadState::CONSUMED,
        FinalizedLostRecoveryUploadState(true));
}

TEST(CloudUploadLifecycleTest, DispatchStartHandshakeReportsFailure) {
    UploadDispatchStartHandshake handshake;
    std::shared_future<bool> started = handshake.GetFuture();

    handshake.Report(false);

    ASSERT_EQ(
        std::future_status::ready,
        started.wait_for(std::chrono::seconds(1)));
    EXPECT_FALSE(started.get());
}

TEST(CloudUploadLifecycleTest, DispatchStartHandshakeKeepsFirstReport) {
    UploadDispatchStartHandshake handshake;
    std::shared_future<bool> started = handshake.GetFuture();

    handshake.Report(true);
    handshake.Report(false);

    ASSERT_EQ(
        std::future_status::ready,
        started.wait_for(std::chrono::seconds(1)));
    EXPECT_TRUE(started.get());
}

TEST(CloudUploadLifecycleTest, RejectedRegistrationDoesNotSendUpload) {
    UploadDispatchStartHandshake handshake;
    std::shared_future<bool> started = handshake.GetFuture();
    bool sendCalled = false;

    const bool dispatched = StartUploadDispatchAndReport(
        handshake,
        []() {
            return false;
        },
        [&sendCalled]() {
            sendCalled = true;
        });

    EXPECT_FALSE(dispatched);
    EXPECT_FALSE(sendCalled);
    EXPECT_FALSE(started.get());
}

TEST(CloudUploadLifecycleTest, SendExceptionReportsDispatchFailure) {
    UploadDispatchStartHandshake handshake;
    std::shared_future<bool> started = handshake.GetFuture();

    EXPECT_THROW(
        StartUploadDispatchAndReport(
            handshake,
            []() {
                return true;
            },
            []() {
                throw std::runtime_error("send failed");
            }),
        std::runtime_error);
    EXPECT_FALSE(started.get());
}

TEST(CloudUploadLifecycleTest, SuccessfulSendReportsDispatchStarted) {
    UploadDispatchStartHandshake handshake;
    std::shared_future<bool> started = handshake.GetFuture();

    const bool dispatched = StartUploadDispatchAndReport(
        handshake,
        []() {
            return true;
        },
        []() {
        });

    EXPECT_TRUE(dispatched);
    EXPECT_TRUE(started.get());
}

TEST(CloudUploadLifecycleTest, MultiTicketBatchWaitsForEveryTicket) {
    std::vector<int> tickets;
    tickets.push_back(11);
    tickets.push_back(12);
    tickets.push_back(13);
    int waitedTicketCount = 0;

    const TicketBatchResult<FakeTicketResult> batch =
        WaitForTicketBatch<int, FakeTicketResult>(
            tickets,
            [&waitedTicketCount](
                const int ticket,
                FakeTicketResult &result) {
                waitedTicketCount++;
                result.accepted = ticket != 12;
                return true;
            },
            [](const FakeTicketResult &result) {
                return result.accepted;
            });

    EXPECT_EQ(3, waitedTicketCount);
    EXPECT_EQ(3U, batch.results.size());
    EXPECT_FALSE(batch.succeeded);
    EXPECT_FALSE(batch.waitFailed);
}

TEST(CloudUploadLifecycleTest, ZeroTicketBatchIsNotSuccessful) {
    const std::vector<int> tickets;
    const TicketBatchResult<FakeTicketResult> batch =
        WaitForTicketBatch<int, FakeTicketResult>(
            tickets,
            [](const int, FakeTicketResult &) {
                return true;
            },
            [](const FakeTicketResult &) {
                return true;
            });

    EXPECT_FALSE(batch.succeeded);
    EXPECT_FALSE(batch.waitFailed);
    EXPECT_TRUE(batch.results.empty());
}

}  // namespace
}  // namespace cloud_edge_slam_sla
