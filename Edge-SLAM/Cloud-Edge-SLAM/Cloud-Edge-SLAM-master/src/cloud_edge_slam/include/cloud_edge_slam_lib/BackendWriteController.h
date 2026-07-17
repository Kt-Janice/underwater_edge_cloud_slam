#ifndef BACKEND_WRITE_CONTROLLER_H
#define BACKEND_WRITE_CONTROLLER_H

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>

namespace ORB_SLAM3 {

// [BackendWriteGate] 后端写入状态。阶段 1 只启用 IDLE / MERGING，REPLAYING 仅保留枚举。
enum class BackendWriteState {
    IDLE = 0,
    MERGING = 1,
    REPLAYING = 2
};

// [BackendWriteGate] 后端注入来源。LIVE 只允许 IDLE，REPLAY 只允许 REPLAYING。
enum class BackendInjectionSource {
    LIVE = 0,
    REPLAY = 1
};

enum class LostTopologyPhase {
    NONE = 0,
    REQUESTED = 1,
    IN_PROGRESS = 2,
    COMPLETED_BLOCKED = 3,
    FAILED_BLOCKED = 4
};

enum class LostTopologyRequestResult { REQUESTED, ALREADY_PENDING, SHUTDOWN_BLOCKED };
enum class LostTopologyWaitResult { OWNER_ACQUIRED, TIMEOUT, CANCELLED, SHUTDOWN, STALE_GENERATION };
enum class LostTopologyCompleteResult { COMPLETED_BLOCKED, RELEASED, SHUTDOWN_CLOSED, NOT_OWNER, STALE_GENERATION };
enum class LostTopologyReleaseResult { RELEASED, DEFERRED, FAILED_BLOCKED, NO_BLOCK, STALE_GENERATION, SHUTDOWN_BLOCKED };
enum class LostTopologyReleaseWaitResult { RELEASED, FAILED_BLOCKED, SHUTDOWN, TIMEOUT, STALE_GENERATION };
enum class LostTopologyCancelResult { CANCELLED, OWNER_RUNNING, NO_REQUEST, STALE_GENERATION };
enum class MergeFinishResult { ENTERED_REPLAYING, EXITED_TO_IDLE_LOST, EXITED_TO_IDLE_SHUTDOWN, NOT_MERGING };

struct LostTopologyRequest {
    LostTopologyRequestResult result;
    std::uint64_t generation;
};

class BackendWriteController {
public:
    BackendWriteController();

    BackendWriteState GetState() const;
    bool CanInjectBackendKeyFrame() const;

    bool TryBeginBackendInjection();
    bool TryBeginBackendInjection(BackendInjectionSource source);
    void EndBackendInjection();

    bool EnterMergingAndWait();
    std::uint64_t ExitMerging();
    MergeFinishResult FinishMergingAndMaybeEnterReplaying();
    std::uint64_t ExitReplayingAndWaitForDrain();

    LostTopologyRequest RequestLostTopologyTransition();
    LostTopologyWaitResult WaitAndBeginLostTopologyTransition(std::uint64_t generation);
    LostTopologyCompleteResult CompleteLostTopologyTransition(std::uint64_t generation);
    LostTopologyCompleteResult FailLostTopologyTransition(std::uint64_t generation, const char *reason);
    LostTopologyReleaseResult ReleaseLostBackendBlock(std::uint64_t generation);
    LostTopologyReleaseWaitResult WaitForLostBackendBlockRelease(std::uint64_t generation);
    LostTopologyCancelResult CancelPendingLostTopologyTransition(std::uint64_t generation);
    void BeginBackendWriteShutdown();
    bool IsLostBackendBlockRequested() const;

    std::uint64_t RecordSkippedNormalInjection();
    std::uint64_t GetSkippedNormalInjections() const;
    void ResetSkippedNormalInjections();
    std::uint64_t RecordSkippedLiveNormalDuringReplaying();
    std::uint64_t GetSkippedLiveNormalDuringReplaying() const;
    void ResetSkippedLiveNormalDuringReplaying();

private:
    std::atomic<int> mState;
    std::atomic<std::uint64_t> mSkippedNormalInjections;
    std::atomic<std::uint64_t> mSkippedLiveNormalDuringReplaying;

    mutable std::mutex mMutex;
    std::condition_variable mCondition;
    int mActiveBackendInjections;
    std::uint64_t mNextLostTopologyGeneration;
    std::uint64_t mActiveLostTopologyGeneration;
    std::uint64_t mLastReleasedLostTopologyGeneration;
    std::uint64_t mLastFailedLostTopologyGeneration;
    LostTopologyPhase mLostTopologyPhase;
    bool mbLostReleasePending;
    bool mbShutdownGate;
    bool mbReplayExitPending;
    std::thread::id mLostTopologyOwner;
};

class BackendInjectionScope {
public:
    explicit BackendInjectionScope(BackendWriteController *pController);
    BackendInjectionScope(BackendWriteController *pController, BackendInjectionSource source);
    ~BackendInjectionScope();

    BackendInjectionScope(const BackendInjectionScope &) = delete;
    BackendInjectionScope &operator=(const BackendInjectionScope &) = delete;
    BackendInjectionScope(BackendInjectionScope &&) = delete;
    BackendInjectionScope &operator=(BackendInjectionScope &&) = delete;

    bool IsActive() const;

private:
    BackendWriteController *mpController;
    bool mbActive;
};

} // namespace ORB_SLAM3

#endif // BACKEND_WRITE_CONTROLLER_H
