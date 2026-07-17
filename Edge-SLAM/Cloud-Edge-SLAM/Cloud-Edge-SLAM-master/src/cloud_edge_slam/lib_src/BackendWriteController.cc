#include "BackendWriteController.h"

#include <chrono>
#include <iostream>

namespace ORB_SLAM3 {

BackendWriteController::BackendWriteController()
    : mState(static_cast<int>(BackendWriteState::IDLE)),
      mSkippedNormalInjections(0),
      mSkippedLiveNormalDuringReplaying(0),
      mActiveBackendInjections(0),
      mNextLostTopologyGeneration(0),
      mActiveLostTopologyGeneration(0),
      mLastReleasedLostTopologyGeneration(0),
      mLastFailedLostTopologyGeneration(0),
      mLostTopologyPhase(LostTopologyPhase::NONE),
      mbLostReleasePending(false),
      mbShutdownGate(false),
      mbReplayExitPending(false),
      mLostTopologyOwner() {
}

BackendWriteState BackendWriteController::GetState() const {
    const int state = mState.load(std::memory_order_acquire);
    if (state == static_cast<int>(BackendWriteState::MERGING)) {
        return BackendWriteState::MERGING;
    }

    if (state == static_cast<int>(BackendWriteState::REPLAYING)) {
        return BackendWriteState::REPLAYING;
    }

    return BackendWriteState::IDLE;
}

bool BackendWriteController::CanInjectBackendKeyFrame() const {
    std::lock_guard<std::mutex> lock(mMutex);
    return !mbShutdownGate && mLostTopologyPhase == LostTopologyPhase::NONE &&
           mState.load(std::memory_order_acquire) == static_cast<int>(BackendWriteState::IDLE);
}

bool BackendWriteController::TryBeginBackendInjection() {
    return TryBeginBackendInjection(BackendInjectionSource::LIVE);
}

bool BackendWriteController::TryBeginBackendInjection(BackendInjectionSource source) {
    std::unique_lock<std::mutex> lock(mMutex);
    if (mbShutdownGate || mLostTopologyPhase != LostTopologyPhase::NONE || mbReplayExitPending) {
        return false;
    }
    const int currentState = mState.load(std::memory_order_acquire);

    bool bAllowed = false;
    if (source == BackendInjectionSource::LIVE) {
        if (currentState == static_cast<int>(BackendWriteState::IDLE)) {
            bAllowed = true;
        }
    } else if (source == BackendInjectionSource::REPLAY) {
        if (currentState == static_cast<int>(BackendWriteState::REPLAYING)) {
            bAllowed = true;
        }
    }

    if (!bAllowed) {
        return false;
    }

    mActiveBackendInjections++;
    return true;
}

void BackendWriteController::EndBackendInjection() {
    bool bNotify = false;

    {
        std::unique_lock<std::mutex> lock(mMutex);
        if (mActiveBackendInjections <= 0) {
            std::cout << "[BackendWriteGate][ERROR] EndBackendInjection underflow detected." << std::endl;
            return;
        }

        mActiveBackendInjections--;
        if (mActiveBackendInjections == 0) {
            bNotify = true;
        }
    }

    if (bNotify) {
        mCondition.notify_all();
    }
}

bool BackendWriteController::EnterMergingAndWait() {
    const auto waitStart = std::chrono::steady_clock::now();

    std::cout << "[BackendWriteGate] Enter MERGING requested" << std::endl;

    std::unique_lock<std::mutex> lock(mMutex);
    if (mbShutdownGate || mLostTopologyPhase != LostTopologyPhase::NONE) {
        return false;
    }
    mSkippedNormalInjections.store(0, std::memory_order_release);
    mState.store(static_cast<int>(BackendWriteState::MERGING), std::memory_order_release);

    while (mActiveBackendInjections > 0 && !mbShutdownGate && mLostTopologyPhase == LostTopologyPhase::NONE) {
        mCondition.wait(lock);
    }

    if (mbShutdownGate || mLostTopologyPhase != LostTopologyPhase::NONE) {
        mState.store(static_cast<int>(BackendWriteState::IDLE), std::memory_order_release);
        lock.unlock();
        mCondition.notify_all();
        return false;
    }

    const auto waitEnd = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> waitMs = waitEnd - waitStart;

    std::cout << "[BackendWriteGate] Enter MERGING ready, active_backend_injections=0, drain_wait_ms="
              << waitMs.count()
              << std::endl;
    return true;
}

std::uint64_t BackendWriteController::ExitMerging() {
    {
        std::unique_lock<std::mutex> lock(mMutex);
        if (mState.load(std::memory_order_acquire) != static_cast<int>(BackendWriteState::MERGING)) {
            std::cout << "[BackendWriteGate][WARNING] ExitMerging called while state is not MERGING." << std::endl;
        }
        mState.store(static_cast<int>(BackendWriteState::IDLE), std::memory_order_release);
    }

    mCondition.notify_all();

    return mSkippedNormalInjections.exchange(0, std::memory_order_acq_rel);
}

MergeFinishResult BackendWriteController::FinishMergingAndMaybeEnterReplaying() {
    std::unique_lock<std::mutex> lock(mMutex);

    if (mState.load(std::memory_order_acquire) != static_cast<int>(BackendWriteState::MERGING)) {
        std::cout << "[BackendWriteGate][WARNING] FinishMergingAndMaybeEnterReplaying called while state is not MERGING." << std::endl;
        return MergeFinishResult::NOT_MERGING;
    }

    if (mbShutdownGate) {
        mState.store(static_cast<int>(BackendWriteState::IDLE), std::memory_order_release);
        lock.unlock();
        mCondition.notify_all();
        return MergeFinishResult::EXITED_TO_IDLE_SHUTDOWN;
    }

    if (mLostTopologyPhase != LostTopologyPhase::NONE) {
        mState.store(static_cast<int>(BackendWriteState::IDLE), std::memory_order_release);
        lock.unlock();
        mCondition.notify_all();
        return MergeFinishResult::EXITED_TO_IDLE_LOST;
    }

    mSkippedLiveNormalDuringReplaying.store(0, std::memory_order_release);
    mState.store(static_cast<int>(BackendWriteState::REPLAYING), std::memory_order_release);
    return MergeFinishResult::ENTERED_REPLAYING;
}

std::uint64_t BackendWriteController::ExitReplayingAndWaitForDrain() {
    std::unique_lock<std::mutex> lock(mMutex);
    if (mState.load(std::memory_order_acquire) != static_cast<int>(BackendWriteState::REPLAYING)) {
        std::cout << "[BackendWriteGate][WARNING] ExitReplaying called while state is not REPLAYING." << std::endl;
        return mSkippedLiveNormalDuringReplaying.exchange(0, std::memory_order_acq_rel);
    }
    mbReplayExitPending = true;
    while (mActiveBackendInjections > 0) {
        mCondition.wait(lock);
    }
    mbReplayExitPending = false;
    mState.store(static_cast<int>(BackendWriteState::IDLE), std::memory_order_release);
    lock.unlock();
    mCondition.notify_all();
    return mSkippedLiveNormalDuringReplaying.exchange(0, std::memory_order_acq_rel);
}

LostTopologyRequest BackendWriteController::RequestLostTopologyTransition() {
    std::lock_guard<std::mutex> lock(mMutex);
    if (mbShutdownGate) {
        return {LostTopologyRequestResult::SHUTDOWN_BLOCKED, 0};
    }
    if (mLostTopologyPhase != LostTopologyPhase::NONE) {
        return {LostTopologyRequestResult::ALREADY_PENDING, mActiveLostTopologyGeneration};
    }
    mActiveLostTopologyGeneration = ++mNextLostTopologyGeneration;
    mLostTopologyPhase = LostTopologyPhase::REQUESTED;
    mbLostReleasePending = false;
    mCondition.notify_all();
    return {LostTopologyRequestResult::REQUESTED, mActiveLostTopologyGeneration};
}

LostTopologyWaitResult BackendWriteController::WaitAndBeginLostTopologyTransition(std::uint64_t generation) {
    std::unique_lock<std::mutex> lock(mMutex);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (true) {
        if (mbShutdownGate) {
            return LostTopologyWaitResult::SHUTDOWN;
        }
        if (generation != mActiveLostTopologyGeneration) {
            return LostTopologyWaitResult::STALE_GENERATION;
        }
        if (mLostTopologyPhase == LostTopologyPhase::NONE) {
            return LostTopologyWaitResult::CANCELLED;
        }
        if (mLostTopologyPhase != LostTopologyPhase::REQUESTED) {
            return LostTopologyWaitResult::CANCELLED;
        }
        if (mState.load(std::memory_order_acquire) == static_cast<int>(BackendWriteState::IDLE) &&
            mActiveBackendInjections == 0) {
            mLostTopologyPhase = LostTopologyPhase::IN_PROGRESS;
            mLostTopologyOwner = std::this_thread::get_id();
            return LostTopologyWaitResult::OWNER_ACQUIRED;
        }
        if (mCondition.wait_until(lock, deadline) == std::cv_status::timeout) {
            return LostTopologyWaitResult::TIMEOUT;
        }
    }
}

LostTopologyCompleteResult BackendWriteController::CompleteLostTopologyTransition(std::uint64_t generation) {
    std::unique_lock<std::mutex> lock(mMutex);
    if (generation != mActiveLostTopologyGeneration) {
        return LostTopologyCompleteResult::STALE_GENERATION;
    }
    if (mLostTopologyPhase != LostTopologyPhase::IN_PROGRESS || mLostTopologyOwner != std::this_thread::get_id()) {
        return LostTopologyCompleteResult::NOT_OWNER;
    }
    mLostTopologyOwner = std::thread::id();
    if (mbShutdownGate) {
        mLostTopologyPhase = LostTopologyPhase::NONE;
        lock.unlock();
        mCondition.notify_all();
        return LostTopologyCompleteResult::SHUTDOWN_CLOSED;
    }
    if (mbLostReleasePending) {
        mLostTopologyPhase = LostTopologyPhase::NONE;
        mLastReleasedLostTopologyGeneration = generation;
        mbLostReleasePending = false;
        lock.unlock();
        mCondition.notify_all();
        return LostTopologyCompleteResult::RELEASED;
    }
    mLostTopologyPhase = LostTopologyPhase::COMPLETED_BLOCKED;
    lock.unlock();
    mCondition.notify_all();
    return LostTopologyCompleteResult::COMPLETED_BLOCKED;
}

LostTopologyCompleteResult BackendWriteController::FailLostTopologyTransition(std::uint64_t generation, const char *reason) {
    std::unique_lock<std::mutex> lock(mMutex);
    if (generation != mActiveLostTopologyGeneration) {
        return LostTopologyCompleteResult::STALE_GENERATION;
    }
    if (mLostTopologyPhase != LostTopologyPhase::IN_PROGRESS || mLostTopologyOwner != std::this_thread::get_id()) {
        return LostTopologyCompleteResult::NOT_OWNER;
    }
    std::cout << "[BackendWriteGate][ERROR] LOST topology transition failed, generation=" << generation
              << ", reason=" << reason << std::endl;
    mLostTopologyOwner = std::thread::id();
    if (mbShutdownGate) {
        mLostTopologyPhase = LostTopologyPhase::NONE;
        lock.unlock();
        mCondition.notify_all();
        return LostTopologyCompleteResult::SHUTDOWN_CLOSED;
    }
    mLostTopologyPhase = LostTopologyPhase::FAILED_BLOCKED;
    mLastFailedLostTopologyGeneration = generation;
    lock.unlock();
    mCondition.notify_all();
    return LostTopologyCompleteResult::COMPLETED_BLOCKED;
}

LostTopologyReleaseResult BackendWriteController::ReleaseLostBackendBlock(std::uint64_t generation) {
    std::unique_lock<std::mutex> lock(mMutex);
    if (mbShutdownGate) {
        return LostTopologyReleaseResult::SHUTDOWN_BLOCKED;
    }
    if (generation != mActiveLostTopologyGeneration) {
        return LostTopologyReleaseResult::STALE_GENERATION;
    }
    if (mLostTopologyPhase == LostTopologyPhase::FAILED_BLOCKED) {
        return LostTopologyReleaseResult::FAILED_BLOCKED;
    }
    if (mLostTopologyPhase == LostTopologyPhase::REQUESTED || mLostTopologyPhase == LostTopologyPhase::IN_PROGRESS) {
        mbLostReleasePending = true;
        return LostTopologyReleaseResult::DEFERRED;
    }
    if (mLostTopologyPhase == LostTopologyPhase::COMPLETED_BLOCKED) {
        mLostTopologyPhase = LostTopologyPhase::NONE;
        mLastReleasedLostTopologyGeneration = generation;
        lock.unlock();
        mCondition.notify_all();
        return LostTopologyReleaseResult::RELEASED;
    }
    return LostTopologyReleaseResult::NO_BLOCK;
}

LostTopologyReleaseWaitResult BackendWriteController::WaitForLostBackendBlockRelease(std::uint64_t generation) {
    std::unique_lock<std::mutex> lock(mMutex);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (true) {
        if (mbShutdownGate) {
            return LostTopologyReleaseWaitResult::SHUTDOWN;
        }
        if (mLastReleasedLostTopologyGeneration >= generation) {
            return LostTopologyReleaseWaitResult::RELEASED;
        }
        if (mLastFailedLostTopologyGeneration == generation) {
            return LostTopologyReleaseWaitResult::FAILED_BLOCKED;
        }
        if (generation != mActiveLostTopologyGeneration) {
            return LostTopologyReleaseWaitResult::STALE_GENERATION;
        }
        if (mCondition.wait_until(lock, deadline) == std::cv_status::timeout) {
            return LostTopologyReleaseWaitResult::TIMEOUT;
        }
    }
}

LostTopologyCancelResult BackendWriteController::CancelPendingLostTopologyTransition(std::uint64_t generation) {
    std::unique_lock<std::mutex> lock(mMutex);
    if (generation != mActiveLostTopologyGeneration) {
        return LostTopologyCancelResult::STALE_GENERATION;
    }
    if (mLostTopologyPhase == LostTopologyPhase::REQUESTED) {
        mLostTopologyPhase = LostTopologyPhase::NONE;
        mbLostReleasePending = false;
        lock.unlock();
        mCondition.notify_all();
        return LostTopologyCancelResult::CANCELLED;
    }
    if (mLostTopologyPhase == LostTopologyPhase::IN_PROGRESS) {
        return LostTopologyCancelResult::OWNER_RUNNING;
    }
    return LostTopologyCancelResult::NO_REQUEST;
}

void BackendWriteController::BeginBackendWriteShutdown() {
    std::unique_lock<std::mutex> lock(mMutex);
    if (mbShutdownGate) {
        return;
    }
    mbShutdownGate = true;
    if (mLostTopologyPhase == LostTopologyPhase::REQUESTED) {
        mLostTopologyPhase = LostTopologyPhase::NONE;
        mbLostReleasePending = false;
    }
    lock.unlock();
    mCondition.notify_all();
}

bool BackendWriteController::IsLostBackendBlockRequested() const {
    std::lock_guard<std::mutex> lock(mMutex);
    return mLostTopologyPhase != LostTopologyPhase::NONE;
}

std::uint64_t BackendWriteController::RecordSkippedNormalInjection() {
    return mSkippedNormalInjections.fetch_add(1, std::memory_order_acq_rel) + 1;
}

std::uint64_t BackendWriteController::GetSkippedNormalInjections() const {
    return mSkippedNormalInjections.load(std::memory_order_acquire);
}

void BackendWriteController::ResetSkippedNormalInjections() {
    mSkippedNormalInjections.store(0, std::memory_order_release);
}

std::uint64_t BackendWriteController::RecordSkippedLiveNormalDuringReplaying() {
    return mSkippedLiveNormalDuringReplaying.fetch_add(1, std::memory_order_acq_rel) + 1;
}

std::uint64_t BackendWriteController::GetSkippedLiveNormalDuringReplaying() const {
    return mSkippedLiveNormalDuringReplaying.load(std::memory_order_acquire);
}

void BackendWriteController::ResetSkippedLiveNormalDuringReplaying() {
    mSkippedLiveNormalDuringReplaying.store(0, std::memory_order_release);
}

BackendInjectionScope::BackendInjectionScope(BackendWriteController *pController)
    : BackendInjectionScope(pController, BackendInjectionSource::LIVE) {
}

BackendInjectionScope::BackendInjectionScope(BackendWriteController *pController, BackendInjectionSource source)
    : mpController(pController),
      mbActive(false) {
    if (mpController != nullptr) {
        mbActive = mpController->TryBeginBackendInjection(source);
    }
}

BackendInjectionScope::~BackendInjectionScope() {
    if (mbActive) {
        mpController->EndBackendInjection();
    }
}

bool BackendInjectionScope::IsActive() const {
    return mbActive;
}

} // namespace ORB_SLAM3
