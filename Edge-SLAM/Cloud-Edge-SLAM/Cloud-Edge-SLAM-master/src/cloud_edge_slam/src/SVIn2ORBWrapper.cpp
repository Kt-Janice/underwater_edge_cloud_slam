#include "SVIn2ORBWrapper.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "System.h"
#include "Atlas.h"
#include "LocalMapping.h"
#include "BackendWriteController.h"
#include <ros/ros.h>
#include <algorithm>
#include <chrono>
#include <iostream> 
#include <fstream>
#include <iomanip>
#include <set>
#include <mutex>
#include <thread>
#include <limits> // [阶段2B修改] 前端位姿最近邻查询
#include <cmath>  // [阶段2B修改] std::abs / std::isfinite

// 声明全局指针，供 SVIn2(OKVIS) 的 Estimator.cpp 访问
SVIn2ORBWrapper* pSVIn2ORBWrapper = nullptr;

namespace {

constexpr double kReplayWaitNormalTimeoutMs = 4000.0;
constexpr double kReplayWaitNormalSleepMs = 10.0;
constexpr float kNormalCompressionTranslationMeters = 0.05f;
constexpr float kNormalCompressionRotationDeg = 2.0f;
constexpr double kNormalCompressionForceKeepIntervalSec = 0.10;
constexpr double kNormalDeferredDurationWarnSec = 2.0;
constexpr float kNormalRotationOrthonormalTolerance = 1e-3f;
constexpr float kNormalRotationDeterminantTolerance = 1e-3f;

enum class WaitNormalResult {
    NORMAL = 0,
    WARNING_TIMEOUT = 1,
    LOST = 2,
    ROS_SHUTDOWN = 3
};

std::mutex gWarningBufferMutex;

struct NormalCompressionStats {
    size_t original_count = 0;
    size_t compressed_count = 0;
    size_t dropped_count = 0;
    size_t pose_parse_failed_count = 0;
    size_t timestamp_invalid_count = 0;
    bool duration_available = false;
    double duration_sec = 0.0;
};

bool IsCachedFrameTimestampLess(const CachedMarginalizedData &a, const CachedMarginalizedData &b) {
    const bool aTimestampFinite = std::isfinite(a.timestamp);
    const bool bTimestampFinite = std::isfinite(b.timestamp);

    if (aTimestampFinite && bTimestampFinite) {
        if (a.timestamp < b.timestamp) {
            return true;
        }

        if (a.timestamp > b.timestamp) {
            return false;
        }

        return a.frame_id < b.frame_id;
    }

    if (aTimestampFinite && !bTimestampFinite) {
        return true;
    }

    if (!aTimestampFinite && bTimestampFinite) {
        return false;
    }

    return a.frame_id < b.frame_id;
}

bool ShouldLogNormalCompressionWarning(const size_t failureCount) {
    if (failureCount <= 5) {
        return true;
    }

    if (failureCount % 30 == 0) {
        return true;
    }

    return false;
}

void LogNormalCompressionKeepWarning(
    const CachedMarginalizedData &data,
    const size_t failureCount,
    const std::string &reason) {
    if (ShouldLogNormalCompressionWarning(failureCount)) {
        std::cout << "[BackendWriteGate][WARNING] Keep NormalDeferredBuffer frame because "
                  << reason
                  << ", frame_id="
                  << data.frame_id
                  << ", timestamp="
                  << data.timestamp
                  << ", failure_count="
                  << failureCount
                  << std::endl;
    }
}

bool ExtractCachedTwc(const CachedMarginalizedData &data, Sophus::SE3f &Twc) {
    Eigen::Map<const Eigen::Matrix4f> TcwMatMapped(data.Tcw_mat);
    if (!TcwMatMapped.allFinite()) {
        return false;
    }

    const Eigen::Matrix3f Rcw = TcwMatMapped.block<3, 3>(0, 0);
    const Eigen::Vector3f tcw = TcwMatMapped.block<3, 1>(0, 3);

    if (!Rcw.allFinite()) {
        return false;
    }

    if (!tcw.allFinite()) {
        return false;
    }

    const Eigen::Matrix3f shouldBeIdentity = Rcw.transpose() * Rcw;
    if (!shouldBeIdentity.allFinite()) {
        return false;
    }

    const float orthonormalError = (shouldBeIdentity - Eigen::Matrix3f::Identity()).norm();
    if (!std::isfinite(orthonormalError)) {
        return false;
    }

    if (orthonormalError > kNormalRotationOrthonormalTolerance) {
        return false;
    }

    const float determinant = Rcw.determinant();
    if (!std::isfinite(determinant)) {
        return false;
    }

    if (std::abs(determinant - 1.0f) > kNormalRotationDeterminantTolerance) {
        return false;
    }

    Eigen::Quaternionf qcw(Rcw);
    qcw.normalize();
    if (!qcw.coeffs().allFinite()) {
        return false;
    }

    const Sophus::SE3f Tcw(qcw, tcw);
    Twc = Tcw.inverse();

    if (!Twc.translation().allFinite()) {
        return false;
    }

    if (!Twc.rotationMatrix().allFinite()) {
        return false;
    }

    const Eigen::Vector3f rotationLog = Twc.so3().log();
    if (!rotationLog.allFinite()) {
        return false;
    }

    return true;
}

void UpdateNormalSnapshotDuration(
    const std::vector<CachedMarginalizedData> &snapshot,
    NormalCompressionStats &stats) {
    bool hasMinTimestamp = false;
    bool hasMaxTimestamp = false;
    double minTimestamp = 0.0;
    double maxTimestamp = 0.0;
    size_t finiteTimestampCount = 0;

    for (size_t i = 0; i < snapshot.size(); ++i) {
        const double timestamp = snapshot[i].timestamp;
        if (!std::isfinite(timestamp)) {
            continue;
        }

        finiteTimestampCount++;

        if (!hasMinTimestamp) {
            minTimestamp = timestamp;
            hasMinTimestamp = true;
        } else if (timestamp < minTimestamp) {
            minTimestamp = timestamp;
        }

        if (!hasMaxTimestamp) {
            maxTimestamp = timestamp;
            hasMaxTimestamp = true;
        } else if (timestamp > maxTimestamp) {
            maxTimestamp = timestamp;
        }
    }

    if (finiteTimestampCount >= 2) {
        const double duration = maxTimestamp - minTimestamp;
        if (std::isfinite(duration) && duration >= 0.0) {
            stats.duration_sec = duration;
            stats.duration_available = true;
        }
    }
}

void RecordNormalTimestampInvalid(
    const CachedMarginalizedData &data,
    NormalCompressionStats &stats) {
    stats.timestamp_invalid_count++;
    LogNormalCompressionKeepWarning(
        data,
        stats.timestamp_invalid_count,
        "timestamp is invalid or out of order");
}

void RecordNormalPoseParseFailed(
    const CachedMarginalizedData &data,
    NormalCompressionStats &stats) {
    stats.pose_parse_failed_count++;
    LogNormalCompressionKeepWarning(
        data,
        stats.pose_parse_failed_count,
        "cached pose parsing failed");
}

void KeepNormalFrame(
    const std::vector<CachedMarginalizedData> &snapshot,
    const size_t index,
    const bool poseValid,
    const Sophus::SE3f &Twc,
    std::vector<CachedMarginalizedData> &compressed,
    size_t &lastKeptIndex,
    double &lastKeptTimestamp,
    Sophus::SE3f &lastKeptTwc,
    bool &lastKeptPoseValid) {
    if (compressed.empty() || lastKeptIndex != index) {
        compressed.push_back(snapshot[index]);
    }

    lastKeptIndex = index;
    lastKeptTimestamp = snapshot[index].timestamp;
    lastKeptPoseValid = poseValid;
    if (poseValid) {
        lastKeptTwc = Twc;
    }
}

std::vector<CachedMarginalizedData> CompressNormalDeferredSnapshot(
    std::vector<CachedMarginalizedData> snapshot,
    NormalCompressionStats &stats) {
    stats.original_count = snapshot.size();
    if (snapshot.empty()) {
        stats.compressed_count = 0;
        stats.dropped_count = 0;
        return std::vector<CachedMarginalizedData>();
    }

    std::sort(snapshot.begin(), snapshot.end(), IsCachedFrameTimestampLess);
    UpdateNormalSnapshotDuration(snapshot, stats);

    std::vector<CachedMarginalizedData> compressed;
    compressed.reserve(snapshot.size());

    size_t lastKeptIndex = 0;
    double lastKeptTimestamp = snapshot[0].timestamp;
    Sophus::SE3f lastKeptTwc;
    Sophus::SE3f currentTwc;
    bool lastKeptPoseValid = ExtractCachedTwc(snapshot[0], lastKeptTwc);
    if (!std::isfinite(snapshot[0].timestamp)) {
        RecordNormalTimestampInvalid(snapshot[0], stats);
    }

    if (!lastKeptPoseValid) {
        RecordNormalPoseParseFailed(snapshot[0], stats);
    }

    compressed.push_back(snapshot[0]);

    if (snapshot.size() >= 3) {
        for (size_t i = 1; i + 1 < snapshot.size(); ++i) {
            const CachedMarginalizedData &candidate = snapshot[i];
            const bool candidateTimestampFinite = std::isfinite(candidate.timestamp);
            const bool lastTimestampFinite = std::isfinite(lastKeptTimestamp);
            double dt = 0.0;
            bool timestampValid = false;

            if (candidateTimestampFinite && lastTimestampFinite) {
                dt = candidate.timestamp - lastKeptTimestamp;
                if (std::isfinite(dt) && dt >= 0.0) {
                    timestampValid = true;
                }
            }

            bool candidatePoseValid = ExtractCachedTwc(candidate, currentTwc);

            if (!timestampValid) {
                RecordNormalTimestampInvalid(candidate, stats);
                if (!candidatePoseValid) {
                    RecordNormalPoseParseFailed(candidate, stats);
                }

                KeepNormalFrame(
                    snapshot,
                    i,
                    candidatePoseValid,
                    currentTwc,
                    compressed,
                    lastKeptIndex,
                    lastKeptTimestamp,
                    lastKeptTwc,
                    lastKeptPoseValid);
                continue;
            }

            if (!lastKeptPoseValid) {
                if (!candidatePoseValid) {
                    RecordNormalPoseParseFailed(candidate, stats);
                }

                KeepNormalFrame(
                    snapshot,
                    i,
                    candidatePoseValid,
                    currentTwc,
                    compressed,
                    lastKeptIndex,
                    lastKeptTimestamp,
                    lastKeptTwc,
                    lastKeptPoseValid);
                continue;
            }

            if (!candidatePoseValid) {
                RecordNormalPoseParseFailed(candidate, stats);
                KeepNormalFrame(
                    snapshot,
                    i,
                    false,
                    currentTwc,
                    compressed,
                    lastKeptIndex,
                    lastKeptTimestamp,
                    lastKeptTwc,
                    lastKeptPoseValid);
                continue;
            }

            const float deltaTrans = (currentTwc.translation() - lastKeptTwc.translation()).norm();
            const Sophus::SO3f deltaR = lastKeptTwc.so3().inverse() * currentTwc.so3();
            const Eigen::Vector3f deltaLog = deltaR.log();
            if (!deltaLog.allFinite()) {
                RecordNormalPoseParseFailed(candidate, stats);
                KeepNormalFrame(
                    snapshot,
                    i,
                    candidatePoseValid,
                    currentTwc,
                    compressed,
                    lastKeptIndex,
                    lastKeptTimestamp,
                    lastKeptTwc,
                    lastKeptPoseValid);
                continue;
            }

            const float deltaRotDeg = deltaLog.norm() * 180.0f / static_cast<float>(M_PI);
            if (!std::isfinite(deltaTrans) || !std::isfinite(deltaRotDeg)) {
                RecordNormalPoseParseFailed(candidate, stats);
                KeepNormalFrame(
                    snapshot,
                    i,
                    candidatePoseValid,
                    currentTwc,
                    compressed,
                    lastKeptIndex,
                    lastKeptTimestamp,
                    lastKeptTwc,
                    lastKeptPoseValid);
                continue;
            }

            if (dt >= kNormalCompressionForceKeepIntervalSec ||
                deltaTrans >= kNormalCompressionTranslationMeters ||
                deltaRotDeg >= kNormalCompressionRotationDeg) {
                KeepNormalFrame(
                    snapshot,
                    i,
                    candidatePoseValid,
                    currentTwc,
                    compressed,
                    lastKeptIndex,
                    lastKeptTimestamp,
                    lastKeptTwc,
                    lastKeptPoseValid);
            }
        }
    }

    const size_t lastIndex = snapshot.size() - 1;
    if (lastKeptIndex != lastIndex) {
        Sophus::SE3f lastTwc;
        const bool lastPoseValid = ExtractCachedTwc(snapshot[lastIndex], lastTwc);
        if (!std::isfinite(snapshot[lastIndex].timestamp)) {
            RecordNormalTimestampInvalid(snapshot[lastIndex], stats);
        }

        if (!lastPoseValid) {
            RecordNormalPoseParseFailed(snapshot[lastIndex], stats);
        }

        KeepNormalFrame(
            snapshot,
            lastIndex,
            lastPoseValid,
            lastTwc,
            compressed,
            lastKeptIndex,
            lastKeptTimestamp,
            lastKeptTwc,
            lastKeptPoseValid);
    }

    stats.compressed_count = compressed.size();
    stats.dropped_count = stats.original_count - stats.compressed_count;
    return compressed;
}

const char *TrackingStateToString(const TrackingState state) {
    if (state == TrackingState::NORMAL) {
        return "NORMAL";
    }

    if (state == TrackingState::WARNING) {
        return "WARNING";
    }

    if (state == TrackingState::LOST) {
        return "LOST";
    }

    return "UNKNOWN";
}

class LostTopologyTransitionGuard {
public:
    LostTopologyTransitionGuard(ORB_SLAM3::System *pSystem, const std::uint64_t generation)
        : mpSystem(pSystem), mGeneration(generation), mbCompleted(false) {
    }

    ~LostTopologyTransitionGuard() {
        if (!mbCompleted && mpSystem != nullptr) {
            mpSystem->FailLostTopologyTransition(mGeneration, "owner_scope_exit_without_complete");
        }
    }

    void Complete() {
        if (!mbCompleted && mpSystem != nullptr) {
            mpSystem->CompleteLostTopologyTransition(mGeneration);
            mbCompleted = true;
        }
    }

    LostTopologyTransitionGuard(const LostTopologyTransitionGuard &) = delete;
    LostTopologyTransitionGuard &operator=(const LostTopologyTransitionGuard &) = delete;

private:
    ORB_SLAM3::System *mpSystem;
    std::uint64_t mGeneration;
    bool mbCompleted;
};

} // namespace

SVIn2ORBWrapper::SVIn2ORBWrapper(ORB_SLAM3::System* pSLAM) {
    mpSLAM = pSLAM;
    mbMergeDeferredBufferInvalid = false;
    mnMergeDeferredTotalPushed = 0;
    mbNormalDeferredBufferInvalid = false;
    mnNormalDeferredTotalPushed = 0;
    mnNormalDeferredTotalReplayedAttempted = 0;
    mnNormalDeferredDroppedAfterSingleRound = 0;
    mbWarningFlushBlockedLogged = false;
    mbLostEventLatched.store(false, std::memory_order_release);
    mbLostTransitionTimeoutLogged.store(false, std::memory_order_release);
    mbLostRecoveryUploadPending.store(false, std::memory_order_release);
    mnLostEventGeneration.store(0, std::memory_order_release);
    mbLastAcceptedTwcValid = false;
    mbCommittedBackendTimeValid = false;
    mLastCommittedBackendTime = 0.0;
    mnCommittedUpdateCount = 0;
    mnCommittedInvalidTimestampCount = 0;
    mLostUploadBoundary.valid = false;
    mLostUploadBoundary.frontend_timestamp = 0.0;
    mLostUploadBoundary.generation = 0;
    mLostUploadBoundaryDispatchState = LostUploadBoundaryDispatchState::EMPTY;
}

SVIn2ORBWrapper::~SVIn2ORBWrapper() {
}

// [阶段2B修改] 缓存 OKVIS / SVIn2 前端连续 Twc 位姿。
// 注意：该函数应由 fullStateCallback 调用，而不是只在 ExecuteInjection() 中调用。
// 原因是 ORB-SLAM3 后端 LOST 时 ExecuteInjection() 可能不执行，但 OKVIS 前端仍可能连续输出位姿。
void SVIn2ORBWrapper::CacheFrontendPose(const double timestamp, const Sophus::SE3f &Twc) {
    if (!std::isfinite(timestamp)) {
        return;
    }

    const Eigen::Vector3f translation = Twc.translation();
    const Eigen::Matrix3f rotation = Twc.rotationMatrix();

    if (!translation.allFinite()) {
        return;
    }

    if (!rotation.allFinite()) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mFrontendPoseMutex);

        FrontendPoseData data;
        data.timestamp = timestamp;
        data.Twc = Twc;
        mFrontendPoseBuffer.push_back(data);

        while (mFrontendPoseBuffer.size() > mnMaxFrontendPoseBufferSize) {
            mFrontendPoseBuffer.pop_front();
        }
    }

    {
        std::lock_guard<std::mutex> lock(mOkvisFullTrajFileMutex);

        if (mOkvisFullTrajFile.is_open()) {
            Eigen::Quaternionf q_wc(rotation);
            q_wc.normalize();

            mOkvisFullTrajFile << std::fixed << std::setprecision(9)
                               << timestamp << " "
                               << translation.x() << " "
                               << translation.y() << " "
                               << translation.z() << " "
                               << q_wc.x() << " "
                               << q_wc.y() << " "
                               << q_wc.z() << " "
                               << q_wc.w() << std::endl;
        }
    }
}

// [阶段2B修改] 根据时间戳查询最近的 OKVIS / SVIn2 前端 Twc 位姿。
// 如果缓存为空或最近时间差超过 tolerance，则返回 false，CloudMerging 会自动退回阶段2A运动弧长权重。
bool SVIn2ORBWrapper::GetNearestFrontendPose(const double timestamp, const double tolerance, Sophus::SE3f &Twc) {
    double timeGap = 0.0;
    return GetNearestFrontendPoseWithTimeGap(timestamp, tolerance, Twc, timeGap);
}

// [CloudMap校正诊断] 根据时间戳查询最近的前端 Twc 位姿，并返回实际时间差。
bool SVIn2ORBWrapper::GetNearestFrontendPoseWithTimeGap(
    const double timestamp,
    const double tolerance,
    Sophus::SE3f &Twc,
    double &timeGap) {
    std::lock_guard<std::mutex> lock(mFrontendPoseMutex);

    if (mFrontendPoseBuffer.empty()) {
        return false;
    }

    double bestDeltaTime = std::numeric_limits<double>::max();
    int bestIndex = -1;

    for (size_t i = 0; i < mFrontendPoseBuffer.size(); i++) {
        const double deltaTime = std::abs(mFrontendPoseBuffer[i].timestamp - timestamp);
        if (deltaTime < bestDeltaTime) {
            bestDeltaTime = deltaTime;
            bestIndex = static_cast<int>(i);
        }
    }

    if (bestIndex < 0) {
        return false;
    }

    if (bestDeltaTime > tolerance) {
        return false;
    }

    Twc = mFrontendPoseBuffer[bestIndex].Twc;
    timeGap = bestDeltaTime;
    return true;
}

// 回调绑定实现
void SVIn2ORBWrapper::RegisterStateCallbacks(std::function<void(const MarginalizedData&)> watchdog_cb, std::function<TrackingState()> get_state_cb) {
    mWatchdogCb = watchdog_cb;
    mGetStateCb = get_state_cb;
}

void SVIn2ORBWrapper::RegisterLostTopologyCallbacks(
    std::function<void(std::uint64_t)> lost_requested_cb,
    std::function<bool(std::uint64_t, LostUploadBoundarySnapshot)> prepare_recovery_upload_cb,
    std::function<bool(std::uint64_t, bool)> finalize_recovery_upload_cb) {
    mLostTopologyRequestedCb = lost_requested_cb;
    mPrepareRecoveryUploadCb = prepare_recovery_upload_cb;
    mFinalizeRecoveryUploadCb = finalize_recovery_upload_cb;
}

void SVIn2ORBWrapper::UpdateLastCommittedBackendTime(
    const double timestamp,
    const CommittedTimeSource source) {
    bool shouldLogUpdate = false;
    bool shouldLogInvalid = false;
    bool previousValid = false;
    double previousTimestamp = 0.0;
    double currentTimestamp = 0.0;
    std::uint64_t updateCount = 0;
    std::uint64_t invalidCount = 0;

    {
        std::lock_guard<std::mutex> lock(mCommittedBackendTimeMutex);
        if (!std::isfinite(timestamp)) {
            mnCommittedInvalidTimestampCount++;
            invalidCount = mnCommittedInvalidTimestampCount;
            if (invalidCount <= 5 || invalidCount % 30 == 0) {
                shouldLogInvalid = true;
            }
        } else {
            previousValid = mbCommittedBackendTimeValid;
            if (previousValid) {
                previousTimestamp = mLastCommittedBackendTime;
            }

            if (!mbCommittedBackendTimeValid || timestamp > mLastCommittedBackendTime) {
                mLastCommittedBackendTime = timestamp;
                mbCommittedBackendTimeValid = true;
                mnCommittedUpdateCount++;
                updateCount = mnCommittedUpdateCount;
                currentTimestamp = mLastCommittedBackendTime;
                if (updateCount == 1 || updateCount % 50 == 0) {
                    shouldLogUpdate = true;
                }
            }
        }
    }

    std::string sourceName = "LIVE";
    if (source == CommittedTimeSource::REPLAY) {
        sourceName = "REPLAY";
    }

    if (shouldLogInvalid) {
        std::cout << "[BackendCommitTime][WARNING] Ignore non-finite committed timestamp, source="
                  << sourceName
                  << ", invalid_count="
                  << invalidCount
                  << std::endl;
    }

    if (shouldLogUpdate) {
        std::cout << "[BackendCommitTime] Update, source="
                  << sourceName
                  << ", frame_timestamp="
                  << timestamp
                  << ", previous=";
        if (previousValid) {
            std::cout << previousTimestamp;
        } else {
            std::cout << "not_available";
        }
        std::cout << ", current="
                  << currentTimestamp
                  << ", update_count="
                  << updateCount
                  << std::endl;
    }
}

BackendCommittedTimeSnapshot SVIn2ORBWrapper::GetLastCommittedBackendTimeSnapshot() const {
    BackendCommittedTimeSnapshot snapshot;
    std::lock_guard<std::mutex> lock(mCommittedBackendTimeMutex);
    snapshot.valid = mbCommittedBackendTimeValid;
    if (snapshot.valid) {
        snapshot.timestamp = mLastCommittedBackendTime;
    }
    return snapshot;
}

bool SVIn2ORBWrapper::SnapshotLostUploadBoundary(const std::uint64_t generation) {
    if (!mbLostEventLatched.load(std::memory_order_acquire)) {
        return false;
    }

    if (mnLostEventGeneration.load(std::memory_order_acquire) != generation) {
        return false;
    }

    const BackendCommittedTimeSnapshot committedSnapshot = GetLastCommittedBackendTimeSnapshot();
    {
        std::lock_guard<std::mutex> lock(mLostUploadBoundaryMutex);
        if (!mbLostEventLatched.load(std::memory_order_acquire) ||
            mnLostEventGeneration.load(std::memory_order_acquire) != generation) {
            return false;
        }
        mLostUploadBoundary.valid = committedSnapshot.valid;
        mLostUploadBoundary.frontend_timestamp = 0.0;
        if (committedSnapshot.valid) {
            mLostUploadBoundary.frontend_timestamp = committedSnapshot.timestamp;
        }
        mLostUploadBoundary.generation = generation;
        mLostUploadBoundaryDispatchState = LostUploadBoundaryDispatchState::AVAILABLE;
    }

    std::cout << "[LostUpload] Snapshot committed boundary, generation="
              << generation
              << ", valid=";
    if (committedSnapshot.valid) {
        std::cout << "true, frontend_timestamp=" << committedSnapshot.timestamp;
    } else {
        std::cout << "false, frontend_timestamp=not_available";
    }
    std::cout << std::endl;
    return true;
}

LostUploadBoundaryDispatchResult SVIn2ORBWrapper::BeginLostUploadBoundaryDispatch(
    const std::uint64_t generation,
    LostUploadBoundarySnapshot &snapshot) {
    snapshot.valid = false;
    snapshot.frontend_timestamp = 0.0;
    snapshot.generation = generation;

    std::lock_guard<std::mutex> lock(mLostUploadBoundaryMutex);
    if (mLostUploadBoundary.generation != generation) {
        return LostUploadBoundaryDispatchResult::STALE_GENERATION;
    }

    if (mLostUploadBoundaryDispatchState == LostUploadBoundaryDispatchState::DISPATCHING) {
        return LostUploadBoundaryDispatchResult::ALREADY_DISPATCHING;
    }

    if (mLostUploadBoundaryDispatchState == LostUploadBoundaryDispatchState::CONSUMED) {
        return LostUploadBoundaryDispatchResult::ALREADY_CONSUMED;
    }

    if (mLostUploadBoundaryDispatchState != LostUploadBoundaryDispatchState::AVAILABLE) {
        return LostUploadBoundaryDispatchResult::NOT_AVAILABLE;
    }

    snapshot = mLostUploadBoundary;
    mLostUploadBoundaryDispatchState = LostUploadBoundaryDispatchState::DISPATCHING;
    if (snapshot.valid) {
        return LostUploadBoundaryDispatchResult::ACQUIRED_COMMITTED;
    }
    return LostUploadBoundaryDispatchResult::ACQUIRED_LEGACY;
}

bool SVIn2ORBWrapper::CompleteLostUploadBoundaryDispatch(const std::uint64_t generation) {
    std::lock_guard<std::mutex> lock(mLostUploadBoundaryMutex);
    if (mLostUploadBoundary.generation != generation ||
        mLostUploadBoundaryDispatchState != LostUploadBoundaryDispatchState::DISPATCHING) {
        return false;
    }
    mLostUploadBoundaryDispatchState = LostUploadBoundaryDispatchState::CONSUMED;
    return true;
}

bool SVIn2ORBWrapper::RollbackLostUploadBoundaryDispatch(const std::uint64_t generation) {
    std::lock_guard<std::mutex> lock(mLostUploadBoundaryMutex);
    if (mLostUploadBoundary.generation != generation ||
        mLostUploadBoundaryDispatchState != LostUploadBoundaryDispatchState::DISPATCHING) {
        return false;
    }
    mLostUploadBoundaryDispatchState = LostUploadBoundaryDispatchState::AVAILABLE;
    return true;
}

bool SVIn2ORBWrapper::RestoreConsumedLostUploadBoundary(const std::uint64_t generation) {
    std::lock_guard<std::mutex> lock(mLostUploadBoundaryMutex);
    if (mLostUploadBoundary.generation != generation ||
        mLostUploadBoundaryDispatchState != LostUploadBoundaryDispatchState::CONSUMED) {
        return false;
    }
    mLostUploadBoundaryDispatchState = LostUploadBoundaryDispatchState::AVAILABLE;
    return true;
}

CachedMarginalizedData SVIn2ORBWrapper::MakeCachedMarginalizedData(const MarginalizedData& data) {
    CachedMarginalizedData safe_data;
    safe_data.timestamp = data.timestamp;
    safe_data.frame_id = data.frame_id;

    for (int i = 0; i < 16; ++i) {
        safe_data.Tcw_mat[i] = data.Tcw_mat[i];
    }

    if (data.landmarks != nullptr) {
        safe_data.landmarks_vec.reserve(data.num_landmarks);
        for (size_t i = 0; i < data.num_landmarks; ++i) {
            safe_data.landmarks_vec.push_back(data.landmarks[i]);
        }
    }

    return safe_data;
}

TrackingState SVIn2ORBWrapper::GetCurrentFrontendHealthState() {
    if (mGetStateCb) {
        return mGetStateCb();
    }

    return TrackingState::NORMAL;
}

bool SVIn2ORBWrapper::CanReplayDeferredBuffers() const {
    if (mpSLAM == nullptr) {
        return false;
    }

    return true;
}

size_t SVIn2ORBWrapper::PushMergeDeferredFrame(const MarginalizedData& data) {
    CachedMarginalizedData safe_data = MakeCachedMarginalizedData(data);

    std::lock_guard<std::mutex> lock(mMergeDeferredBufferMutex);
    if (mbMergeDeferredBufferInvalid) {
        return mMergeDeferredBuffer.size();
    }

    mMergeDeferredBuffer.push_back(safe_data);
    mnMergeDeferredTotalPushed++;
    return mMergeDeferredBuffer.size();
}

std::vector<CachedMarginalizedData> SVIn2ORBWrapper::SnapshotAndClearMergeDeferredBuffer() {
    std::lock_guard<std::mutex> lock(mMergeDeferredBufferMutex);

    std::vector<CachedMarginalizedData> snapshot = mMergeDeferredBuffer;
    mMergeDeferredBuffer.clear();
    mnMergeDeferredTotalPushed = 0;
    return snapshot;
}

void SVIn2ORBWrapper::ClearMergeDeferredBuffer(const std::string &reason) {
    std::lock_guard<std::mutex> lock(mMergeDeferredBufferMutex);

    const size_t cleared = mMergeDeferredBuffer.size();
    mMergeDeferredBuffer.clear();
    mbMergeDeferredBufferInvalid = false;
    mnMergeDeferredTotalPushed = 0;

    std::cout << "[BackendWriteGate] Clear MergeDeferredBuffer, reason="
              << reason
              << ", cleared="
              << cleared
              << std::endl;
}

bool SVIn2ORBWrapper::HasMergeDeferredFrames() {
    std::lock_guard<std::mutex> lock(mMergeDeferredBufferMutex);
    return !mMergeDeferredBuffer.empty();
}

size_t SVIn2ORBWrapper::GetMergeDeferredFrameCount() {
    std::lock_guard<std::mutex> lock(mMergeDeferredBufferMutex);
    return mMergeDeferredBuffer.size();
}

std::vector<double> SVIn2ORBWrapper::GetMergeDeferredTimestampsSnapshot() {
    std::lock_guard<std::mutex> lock(mMergeDeferredBufferMutex);

    std::vector<double> timestamps;
    timestamps.reserve(mMergeDeferredBuffer.size());
    for (const CachedMarginalizedData &data : mMergeDeferredBuffer) {
        timestamps.push_back(data.timestamp);
    }

    return timestamps;
}

bool SVIn2ORBWrapper::IsMergeDeferredBufferInvalid() {
    std::lock_guard<std::mutex> lock(mMergeDeferredBufferMutex);
    return mbMergeDeferredBufferInvalid;
}

void SVIn2ORBWrapper::MarkMergeDeferredBufferInvalid(const std::string &reason) {
    std::lock_guard<std::mutex> lock(mMergeDeferredBufferMutex);

    const size_t cleared = mMergeDeferredBuffer.size();
    mMergeDeferredBuffer.clear();
    mbMergeDeferredBufferInvalid = true;
    mnMergeDeferredTotalPushed = 0;

    std::cout << "[BackendWriteGate] Mark MergeDeferredBuffer invalid, reason="
              << reason
              << ", cleared="
              << cleared
              << std::endl;
}

size_t SVIn2ORBWrapper::PushNormalDeferredFrame(const MarginalizedData& data) {
    CachedMarginalizedData safe_data = MakeCachedMarginalizedData(data);

    std::lock_guard<std::mutex> lock(mNormalDeferredBufferMutex);
    if (mbNormalDeferredBufferInvalid) {
        return mNormalDeferredBuffer.size();
    }

    mNormalDeferredBuffer.push_back(safe_data);
    mnNormalDeferredTotalPushed++;

    const size_t deferred = mNormalDeferredBuffer.size();
    if (deferred > 0 && (deferred <= 5 || deferred % 30 == 0)) {
        std::cout << "[BackendWriteGate] Deferred live NORMAL during REPLAYING, t="
                  << data.timestamp
                  << ", normal_deferred="
                  << deferred
                  << std::endl;
    }

    return deferred;
}

std::vector<CachedMarginalizedData> SVIn2ORBWrapper::SnapshotAndClearNormalDeferredBuffer() {
    std::lock_guard<std::mutex> lock(mNormalDeferredBufferMutex);

    std::vector<CachedMarginalizedData> snapshot = mNormalDeferredBuffer;
    mNormalDeferredBuffer.clear();
    return snapshot;
}

void SVIn2ORBWrapper::ClearNormalDeferredBuffer(const std::string &reason) {
    std::lock_guard<std::mutex> lock(mNormalDeferredBufferMutex);

    const size_t cleared = mNormalDeferredBuffer.size();
    mNormalDeferredBuffer.clear();
    mbNormalDeferredBufferInvalid = false;

    std::cout << "[BackendWriteGate] Clear NormalDeferredBuffer, reason="
              << reason
              << ", cleared="
              << cleared
              << std::endl;
}

bool SVIn2ORBWrapper::HasNormalDeferredFrames() {
    std::lock_guard<std::mutex> lock(mNormalDeferredBufferMutex);
    return !mNormalDeferredBuffer.empty();
}

size_t SVIn2ORBWrapper::GetNormalDeferredFrameCount() {
    std::lock_guard<std::mutex> lock(mNormalDeferredBufferMutex);
    return mNormalDeferredBuffer.size();
}

bool SVIn2ORBWrapper::IsNormalDeferredBufferInvalid() {
    std::lock_guard<std::mutex> lock(mNormalDeferredBufferMutex);
    return mbNormalDeferredBufferInvalid;
}

void SVIn2ORBWrapper::MarkNormalDeferredBufferInvalid(const std::string &reason) {
    std::lock_guard<std::mutex> lock(mNormalDeferredBufferMutex);

    const size_t cleared = mNormalDeferredBuffer.size();
    mNormalDeferredBuffer.clear();
    mbNormalDeferredBufferInvalid = true;

    std::cout << "[BackendWriteGate] Mark NormalDeferredBuffer invalid, reason="
              << reason
              << ", cleared="
              << cleared
              << std::endl;
}

size_t SVIn2ORBWrapper::ReplayCachedDeferredFrames(
    std::vector<CachedMarginalizedData> cachedFrames,
    bool &bAborted,
    std::string &abortReason) {
    bAborted = false;
    abortReason.clear();

    std::sort(cachedFrames.begin(), cachedFrames.end(), IsCachedFrameTimestampLess);

    size_t attempted = 0;
    for (size_t i = 0; i < cachedFrames.size(); ++i) {
        if (mpSLAM == nullptr) {
            bAborted = true;
            abortReason = "system_null";
            break;
        }

        if (mpSLAM->GetBackendWriteState() != ORB_SLAM3::BackendWriteState::REPLAYING) {
            bAborted = true;
            abortReason = "backend_state_not_replaying";
            break;
        }

        const TrackingState currentFrontendState = GetCurrentFrontendHealthState();
        if (currentFrontendState != TrackingState::NORMAL) {
            bAborted = true;
            abortReason = std::string("frontend_state_") + TrackingStateToString(currentFrontendState);
            break;
        }

        if (IsMergeDeferredBufferInvalid()) {
            bAborted = true;
            abortReason = "merge_deferred_buffer_invalid";
            break;
        }

        if (IsNormalDeferredBufferInvalid()) {
            bAborted = true;
            abortReason = "normal_deferred_buffer_invalid";
            break;
        }

        ORB_SLAM3::BackendWriteController *pBackendWriteController = mpSLAM->GetBackendWriteController();
        ORB_SLAM3::BackendInjectionScope backendInjectionScope(
            pBackendWriteController,
            ORB_SLAM3::BackendInjectionSource::REPLAY);

        if (!backendInjectionScope.IsActive()) {
            bAborted = true;
            abortReason = "replay_scope_inactive";
            break;
        }

        CachedMarginalizedData cachedData = cachedFrames[i];
        MarginalizedData rawData = cachedData.toRawData();
        attempted++;
        const InjectionResult injectionResult = ExecuteInjection(rawData);
        if (injectionResult == InjectionResult::COMMITTED) {
            UpdateLastCommittedBackendTime(
                rawData.timestamp,
                CommittedTimeSource::REPLAY);
        }
    }

    return attempted;
}

void SVIn2ORBWrapper::ReplayMergeDeferredBuffer() {
    if (mpSLAM == nullptr) {
        std::cout << "[BackendWriteGate][ERROR] Cannot replay MergeDeferredBuffer because System is null." << std::endl;
        return;
    }

    const auto replayStart = std::chrono::steady_clock::now();
    size_t mergeAttempted = 0;
    size_t warningSnapshotCount = 0;
    size_t mergeSnapshotCount = 0;
    size_t combinedAttempted = 0;
    size_t normalAttempted = 0;
    size_t normalSnapshotCount = 0;
    size_t normalCompressedCount = 0;
    size_t normalCompressionDropped = 0;
    size_t normalPoseParseFailed = 0;
    size_t normalTimestampInvalid = 0;
    size_t normalArrivedAfterSnapshot = 0;
    bool bAborted = false;
    std::string abortReason;
    double waitNormalMs = 0.0;
    std::string waitNormalResult = "NONE";
    std::string warningReplayResult = "not_started";
    std::string normalCompressionResult = "not_started";

    auto snapshotAndClearWarningBufferForReplay = [&]() {
        std::lock_guard<std::mutex> warningLock(gWarningBufferMutex);

        std::vector<CachedMarginalizedData> snapshot = mWarningBuffer;
        mWarningBuffer.clear();
        return snapshot;
    };

    auto getWarningBufferCountForLog = [&]() {
        std::lock_guard<std::mutex> warningLock(gWarningBufferMutex);
        return mWarningBuffer.size();
    };

    auto waitForFrontendNormalBeforeReplay = [&](
        double &waitedMs,
        TrackingState &finalState) -> WaitNormalResult {
        const auto waitStart = std::chrono::steady_clock::now();
        waitedMs = 0.0;
        finalState = GetCurrentFrontendHealthState();

        while (true) {
            if (!ros::ok()) {
                const auto waitEnd = std::chrono::steady_clock::now();
                const std::chrono::duration<double, std::milli> durationMs = waitEnd - waitStart;
                waitedMs = durationMs.count();
                finalState = GetCurrentFrontendHealthState();
                return WaitNormalResult::ROS_SHUTDOWN;
            }

            finalState = GetCurrentFrontendHealthState();
            if (finalState == TrackingState::NORMAL) {
                const auto waitEnd = std::chrono::steady_clock::now();
                const std::chrono::duration<double, std::milli> durationMs = waitEnd - waitStart;
                waitedMs = durationMs.count();
                return WaitNormalResult::NORMAL;
            }

            if (finalState == TrackingState::LOST) {
                const auto waitEnd = std::chrono::steady_clock::now();
                const std::chrono::duration<double, std::milli> durationMs = waitEnd - waitStart;
                waitedMs = durationMs.count();
                return WaitNormalResult::LOST;
            }

            const auto waitNow = std::chrono::steady_clock::now();
            const std::chrono::duration<double, std::milli> durationMs = waitNow - waitStart;
            waitedMs = durationMs.count();
            if (waitedMs >= kReplayWaitNormalTimeoutMs) {
                finalState = TrackingState::WARNING;
                return WaitNormalResult::WARNING_TIMEOUT;
            }

            const std::chrono::duration<double, std::milli> sleepDuration(kReplayWaitNormalSleepMs);
            std::this_thread::sleep_for(sleepDuration);
        }
    };

    do {
        if (IsMergeDeferredBufferInvalid()) {
            bAborted = true;
            abortReason = "merge_deferred_buffer_invalid_before_replay";
            std::cout << "[BackendWriteGate] Abort REPLAYING because MergeDeferredBuffer is invalid" << std::endl;
            break;
        }

        if (IsNormalDeferredBufferInvalid()) {
            bAborted = true;
            abortReason = "normal_deferred_buffer_invalid_before_replay";
            std::cout << "[BackendWriteGate] Abort REPLAYING because NormalDeferredBuffer is invalid" << std::endl;
            break;
        }

        const TrackingState currentFrontendState = GetCurrentFrontendHealthState();
        if (currentFrontendState == TrackingState::WARNING) {
            std::cout << "[BackendWriteGate] REPLAYING waits for frontend NORMAL, current=WARNING, timeout_ms="
                      << kReplayWaitNormalTimeoutMs
                      << std::endl;

            TrackingState finalWaitState = TrackingState::WARNING;
            const WaitNormalResult currentWaitNormalResult = waitForFrontendNormalBeforeReplay(
                waitNormalMs,
                finalWaitState);

            if (currentWaitNormalResult == WaitNormalResult::NORMAL) {
                waitNormalResult = "NORMAL";
                std::cout << "[BackendWriteGate] REPLAYING resumes because frontend recovered to NORMAL, wait_ms="
                          << waitNormalMs
                          << std::endl;

                const size_t warningBufferSize = getWarningBufferCountForLog();
                if (warningBufferSize > 0) {
                    std::cout << "[BackendWriteGate][WARNING] Warning buffer has "
                              << warningBufferSize
                              << " frames after frontend recovered to NORMAL; ordered warning-buffer flush is deferred to a later stage."
                              << std::endl;
                }

                if (mpSLAM->GetBackendWriteState() != ORB_SLAM3::BackendWriteState::REPLAYING) {
                    bAborted = true;
                    abortReason = "backend_state_not_REPLAYING_after_wait_NORMAL";
                    std::cout << "[BackendWriteGate] Abort REPLAYING because backend state is not REPLAYING after waiting NORMAL." << std::endl;
                    break;
                }

                const TrackingState frontendStateAfterWait = GetCurrentFrontendHealthState();
                if (frontendStateAfterWait != TrackingState::NORMAL) {
                    bAborted = true;
                    abortReason = std::string("frontend_state_") + TrackingStateToString(frontendStateAfterWait) + "_after_wait_NORMAL";
                    std::cout << "[BackendWriteGate] Abort REPLAYING because frontend state is "
                              << TrackingStateToString(frontendStateAfterWait)
                              << " after waiting NORMAL."
                              << std::endl;
                    break;
                }

                if (IsMergeDeferredBufferInvalid()) {
                    bAborted = true;
                    abortReason = "merge_deferred_buffer_invalid_after_wait_NORMAL";
                    std::cout << "[BackendWriteGate] Abort REPLAYING because MergeDeferredBuffer is invalid after waiting NORMAL." << std::endl;
                    break;
                }

                if (IsNormalDeferredBufferInvalid()) {
                    bAborted = true;
                    abortReason = "normal_deferred_buffer_invalid_after_wait_NORMAL";
                    std::cout << "[BackendWriteGate] Abort REPLAYING because NormalDeferredBuffer is invalid after waiting NORMAL." << std::endl;
                    break;
                }
            } else if (currentWaitNormalResult == WaitNormalResult::LOST) {
                waitNormalResult = "LOST";
                bAborted = true;
                abortReason = "frontend_state_LOST_while_waiting_NORMAL";
                std::cout << "[BackendWriteGate] Abort REPLAYING because frontend became LOST while waiting NORMAL" << std::endl;
                break;
            } else if (currentWaitNormalResult == WaitNormalResult::ROS_SHUTDOWN) {
                waitNormalResult = "ROS_SHUTDOWN";
                bAborted = true;
                abortReason = "ros_shutdown_while_waiting_NORMAL";
                std::cout << "[BackendWriteGate] Abort REPLAYING because ros shutdown while waiting NORMAL" << std::endl;
                break;
            } else if (currentWaitNormalResult == WaitNormalResult::WARNING_TIMEOUT) {
                waitNormalResult = "WARNING_TIMEOUT";
                bAborted = true;
                abortReason = "frontend_state_WARNING_wait_NORMAL_timeout";
                std::cout << "[BackendWriteGate] Abort REPLAYING because frontend stayed WARNING until timeout, wait_ms="
                          << waitNormalMs
                          << std::endl;
                break;
            }
        } else if (currentFrontendState != TrackingState::NORMAL) {
            bAborted = true;
            abortReason = std::string("frontend_state_") + TrackingStateToString(currentFrontendState);
            std::cout << "[BackendWriteGate] Abort REPLAYING because frontend state is "
                      << TrackingStateToString(currentFrontendState)
                      << std::endl;
            break;
        }

        std::vector<CachedMarginalizedData> warningSnapshot = snapshotAndClearWarningBufferForReplay();
        warningSnapshotCount = warningSnapshot.size();

        if (mpSLAM->GetBackendWriteState() != ORB_SLAM3::BackendWriteState::REPLAYING) {
            bAborted = true;
            abortReason = "backend_state_not_REPLAYING_after_warning_snapshot";
            warningReplayResult = "aborted_dropped_snapshot";
            std::cout << "[BackendWriteGate] Abort REPLAYING after WarningBuffer snapshot because backend state is not REPLAYING, warning_replay_result="
                      << warningReplayResult
                      << std::endl;
            break;
        }

        const TrackingState frontendStateAfterWarningSnapshot = GetCurrentFrontendHealthState();
        if (frontendStateAfterWarningSnapshot != TrackingState::NORMAL) {
            bAborted = true;
            abortReason = std::string("frontend_state_") + TrackingStateToString(frontendStateAfterWarningSnapshot) + "_after_warning_snapshot";
            warningReplayResult = "aborted_dropped_snapshot";
            std::cout << "[BackendWriteGate] Abort REPLAYING after WarningBuffer snapshot because frontend state is "
                      << TrackingStateToString(frontendStateAfterWarningSnapshot)
                      << ", warning_replay_result="
                      << warningReplayResult
                      << std::endl;
            break;
        }

        if (IsMergeDeferredBufferInvalid()) {
            bAborted = true;
            abortReason = "merge_deferred_buffer_invalid_after_warning_snapshot";
            warningReplayResult = "aborted_dropped_snapshot";
            std::cout << "[BackendWriteGate] Abort REPLAYING after WarningBuffer snapshot because MergeDeferredBuffer is invalid, warning_replay_result="
                      << warningReplayResult
                      << std::endl;
            break;
        }

        if (IsNormalDeferredBufferInvalid()) {
            bAborted = true;
            abortReason = "normal_deferred_buffer_invalid_after_warning_snapshot";
            warningReplayResult = "aborted_dropped_snapshot";
            std::cout << "[BackendWriteGate] Abort REPLAYING after WarningBuffer snapshot because NormalDeferredBuffer is invalid, warning_replay_result="
                      << warningReplayResult
                      << std::endl;
            break;
        }

        std::vector<CachedMarginalizedData> mergeSnapshot = SnapshotAndClearMergeDeferredBuffer();
        mergeSnapshotCount = mergeSnapshot.size();

        std::vector<CachedMarginalizedData> warningAndMergeSnapshot;
        warningAndMergeSnapshot.reserve(warningSnapshot.size() + mergeSnapshot.size());
        warningAndMergeSnapshot.insert(
            warningAndMergeSnapshot.end(),
            warningSnapshot.begin(),
            warningSnapshot.end());
        warningAndMergeSnapshot.insert(
            warningAndMergeSnapshot.end(),
            mergeSnapshot.begin(),
            mergeSnapshot.end());

        std::cout << "[BackendWriteGate] Replay WarningBuffer with MergeDeferredBuffer, warning_snapshot="
                  << warningSnapshotCount
                  << ", merge_snapshot="
                  << mergeSnapshotCount
                  << ", combined="
                  << warningAndMergeSnapshot.size()
                  << std::endl;

        if (warningAndMergeSnapshot.empty()) {
            warningReplayResult = "skipped_empty";
            std::cout << "[BackendWriteGate] Warning/Merge replay skipped because combined snapshot is empty, warning_replay_result="
                      << warningReplayResult
                      << std::endl;
        } else {
            const auto combinedReplayStart = std::chrono::steady_clock::now();
            combinedAttempted = ReplayCachedDeferredFrames(warningAndMergeSnapshot, bAborted, abortReason);
            const auto combinedReplayEnd = std::chrono::steady_clock::now();
            const std::chrono::duration<double, std::milli> combinedDurationMs = combinedReplayEnd - combinedReplayStart;

            if (bAborted) {
                warningReplayResult = "aborted_dropped_snapshot";
                std::cout << "[BackendWriteGate] Abort REPLAYING during Warning/Merge combined replay, reason="
                      << abortReason
                      << ", warning_replay_result="
                      << warningReplayResult
                      << std::endl;
                break;
            }

            warningReplayResult = "success";
            std::cout << "[BackendWriteGate] Warning/Merge replay finished, combined_attempted="
                      << combinedAttempted
                      << ", warning_snapshot="
                      << warningSnapshotCount
                      << ", merge_snapshot="
                      << mergeSnapshotCount
                      << ", duration_ms="
                      << combinedDurationMs.count()
                      << std::endl;
        }

        if (IsMergeDeferredBufferInvalid()) {
            bAborted = true;
            abortReason = "merge_deferred_buffer_invalid_before_normal_replay";
            std::cout << "[BackendWriteGate] Abort REPLAYING because MergeDeferredBuffer is invalid before normal replay." << std::endl;
            break;
        }

        if (IsNormalDeferredBufferInvalid()) {
            bAborted = true;
            abortReason = "normal_deferred_buffer_invalid_before_normal_replay";
            std::cout << "[BackendWriteGate] Abort REPLAYING because NormalDeferredBuffer is invalid before normal replay." << std::endl;
            break;
        }

        const TrackingState frontendStateBeforeNormalReplay = GetCurrentFrontendHealthState();
        if (frontendStateBeforeNormalReplay != TrackingState::NORMAL) {
            bAborted = true;
            abortReason = std::string("frontend_state_") + TrackingStateToString(frontendStateBeforeNormalReplay);
            std::cout << "[BackendWriteGate] Abort REPLAYING before NormalDeferredBuffer because frontend state is "
                      << TrackingStateToString(frontendStateBeforeNormalReplay)
                      << std::endl;
            break;
        }

        std::vector<CachedMarginalizedData> normalSnapshot = SnapshotAndClearNormalDeferredBuffer();
        normalSnapshotCount = normalSnapshot.size();
        if (normalSnapshot.empty()) {
            normalCompressionResult = "skipped_empty";
            std::cout << "[BackendWriteGate] NormalDeferredBuffer compressed replay skipped because snapshot is empty." << std::endl;
            break;
        }

        NormalCompressionStats normalCompressionStats;
        std::vector<CachedMarginalizedData> compressedNormalSnapshot =
            CompressNormalDeferredSnapshot(normalSnapshot, normalCompressionStats);
        normalCompressedCount = normalCompressionStats.compressed_count;
        normalCompressionDropped = normalCompressionStats.dropped_count;
        normalPoseParseFailed = normalCompressionStats.pose_parse_failed_count;
        normalTimestampInvalid = normalCompressionStats.timestamp_invalid_count;

        if (normalCompressionStats.duration_available) {
            std::cout << "[BackendWriteGate] NormalDeferredBuffer snapshot, original="
                      << normalSnapshotCount
                      << ", duration_sec="
                      << normalCompressionStats.duration_sec
                      << std::endl;
        } else {
            std::cout << "[BackendWriteGate] NormalDeferredBuffer snapshot, original="
                      << normalSnapshotCount
                      << ", duration_sec=not_available"
                      << std::endl;
        }

        if (normalCompressionStats.duration_available &&
            normalCompressionStats.duration_sec > kNormalDeferredDurationWarnSec) {
            std::cout << "[BackendWriteGate][WARNING] NormalDeferredBuffer snapshot duration exceeds warning threshold, duration_sec="
                      << normalCompressionStats.duration_sec
                      << ", threshold_sec="
                      << kNormalDeferredDurationWarnSec
                      << std::endl;
        }

        if (normalCompressionStats.duration_available) {
            std::cout << "[BackendWriteGate] Compress NormalDeferredBuffer, original="
                      << normalCompressionStats.original_count
                      << ", compressed="
                      << normalCompressionStats.compressed_count
                      << ", dropped="
                      << normalCompressionStats.dropped_count
                      << ", duration_sec="
                      << normalCompressionStats.duration_sec
                      << ", pose_parse_failed="
                      << normalCompressionStats.pose_parse_failed_count
                      << ", timestamp_invalid="
                      << normalCompressionStats.timestamp_invalid_count
                      << ", policy=motion_time, translation_m="
                      << kNormalCompressionTranslationMeters
                      << ", rotation_deg="
                      << kNormalCompressionRotationDeg
                      << ", force_keep_interval_sec="
                      << kNormalCompressionForceKeepIntervalSec
                      << std::endl;
        } else {
            std::cout << "[BackendWriteGate] Compress NormalDeferredBuffer, original="
                      << normalCompressionStats.original_count
                      << ", compressed="
                      << normalCompressionStats.compressed_count
                      << ", dropped="
                      << normalCompressionStats.dropped_count
                      << ", duration_sec=not_available"
                      << ", pose_parse_failed="
                      << normalCompressionStats.pose_parse_failed_count
                      << ", timestamp_invalid="
                      << normalCompressionStats.timestamp_invalid_count
                      << ", policy=motion_time, translation_m="
                      << kNormalCompressionTranslationMeters
                      << ", rotation_deg="
                      << kNormalCompressionRotationDeg
                      << ", force_keep_interval_sec="
                      << kNormalCompressionForceKeepIntervalSec
                      << std::endl;
        }

        if (!ros::ok()) {
            bAborted = true;
            abortReason = "ros_shutdown_before_normal_compressed_replay";
            normalCompressionResult = "aborted_before_replay";
            std::cout << "[BackendWriteGate] Abort REPLAYING before NormalDeferredBuffer compressed replay because ros shutdown." << std::endl;
            break;
        }

        if (mpSLAM->GetBackendWriteState() != ORB_SLAM3::BackendWriteState::REPLAYING) {
            bAborted = true;
            abortReason = "backend_state_not_REPLAYING_before_normal_compressed_replay";
            normalCompressionResult = "aborted_before_replay";
            std::cout << "[BackendWriteGate] Abort REPLAYING before NormalDeferredBuffer compressed replay because backend state is not REPLAYING." << std::endl;
            break;
        }

        const TrackingState frontendStateBeforeNormalCompressedReplay = GetCurrentFrontendHealthState();
        if (frontendStateBeforeNormalCompressedReplay != TrackingState::NORMAL) {
            bAborted = true;
            abortReason = std::string("frontend_state_") + TrackingStateToString(frontendStateBeforeNormalCompressedReplay) + "_before_normal_compressed_replay";
            normalCompressionResult = "aborted_before_replay";
            std::cout << "[BackendWriteGate] Abort REPLAYING before NormalDeferredBuffer compressed replay because frontend state is "
                      << TrackingStateToString(frontendStateBeforeNormalCompressedReplay)
                      << std::endl;
            break;
        }

        if (IsMergeDeferredBufferInvalid()) {
            bAborted = true;
            abortReason = "merge_deferred_buffer_invalid_before_normal_compressed_replay";
            normalCompressionResult = "aborted_before_replay";
            std::cout << "[BackendWriteGate] Abort REPLAYING before NormalDeferredBuffer compressed replay because MergeDeferredBuffer is invalid." << std::endl;
            break;
        }

        if (IsNormalDeferredBufferInvalid()) {
            bAborted = true;
            abortReason = "normal_deferred_buffer_invalid_before_normal_compressed_replay";
            normalCompressionResult = "aborted_before_replay";
            std::cout << "[BackendWriteGate] Abort REPLAYING before NormalDeferredBuffer compressed replay because NormalDeferredBuffer is invalid." << std::endl;
            break;
        }

        const auto normalReplayStart = std::chrono::steady_clock::now();
        normalAttempted = ReplayCachedDeferredFrames(compressedNormalSnapshot, bAborted, abortReason);
        const auto normalReplayEnd = std::chrono::steady_clock::now();
        const std::chrono::duration<double, std::milli> normalDurationMs = normalReplayEnd - normalReplayStart;

        mnNormalDeferredTotalReplayedAttempted += normalAttempted;

        std::cout << "[BackendWriteGate] NormalDeferredBuffer compressed replay finished, attempted="
                  << normalAttempted
                  << ", compressed="
                  << normalCompressedCount
                  << ", original="
                  << normalSnapshotCount
                  << ", duration_ms="
                  << normalDurationMs.count()
                  << std::endl;

        if (bAborted) {
            normalCompressionResult = "aborted_during_replay";
            std::cout << "[BackendWriteGate] Abort REPLAYING, reason="
                      << abortReason
                      << ", normal_attempted="
                      << normalAttempted
                      << std::endl;
            break;
        }

        normalCompressionResult = "success";
    } while (false);

    if (bAborted) {
        ClearMergeDeferredBuffer(std::string("abort_replaying_") + abortReason);
        ClearNormalDeferredBuffer(std::string("abort_replaying_") + abortReason);
    } else {
        normalArrivedAfterSnapshot = GetNormalDeferredFrameCount();
        if (normalArrivedAfterSnapshot > 0) {
            mnNormalDeferredDroppedAfterSingleRound += normalArrivedAfterSnapshot;
            std::cout << "[BackendWriteGate][WARNING] Drop NormalDeferredBuffer frames arrived after normal snapshot, left="
                      << normalArrivedAfterSnapshot
                      << ", reason=stage4_single_round_boundary"
                      << std::endl;
            ClearNormalDeferredBuffer("stage4_single_round_boundary");
        }
    }

    if (mpSLAM->GetBackendWriteState() == ORB_SLAM3::BackendWriteState::REPLAYING) {
        mpSLAM->ExitBackendReplayingAndWaitForDrain();
    } else {
        std::cout << "[BackendWriteGate][WARNING] Skip ExitBackendReplaying because backend state is not REPLAYING." << std::endl;
    }

    const auto replayEnd = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> durationMs = replayEnd - replayStart;

    std::cout << "[BackendWriteGate] Exit REPLAYING, merge_attempted="
              << mergeAttempted
              << ", merge_attempted_scope=not_separately_tracked_after_stage4_combined_replay"
              << ", normal_replay_attempted="
              << normalAttempted
              << ", normal_snapshot="
              << normalSnapshotCount
              << ", normal_compressed="
              << normalCompressedCount
              << ", normal_compression_dropped="
              << normalCompressionDropped
              << ", normal_pose_parse_failed="
              << normalPoseParseFailed
              << ", normal_timestamp_invalid="
              << normalTimestampInvalid
              << ", normal_arrived_after_snapshot="
              << normalArrivedAfterSnapshot
              << ", normal_compression_result="
              << normalCompressionResult
              << ", normal_compression_policy=motion_time"
              << ", accepted_estimate_not_precise=not_available"
              << ", warning_snapshot="
              << warningSnapshotCount
              << ", merge_snapshot="
              << mergeSnapshotCount
              << ", combined_attempted="
              << combinedAttempted
              << ", warning_replay_result="
              << warningReplayResult
              << ", wait_normal_ms="
              << waitNormalMs
              << ", wait_normal_result="
              << waitNormalResult
              << ", duration_ms="
              << durationMs.count()
              << std::endl;
}

void SVIn2ORBWrapper::InjectSVIn2MarginalizedData(const MarginalizedData& data) {
    // 异常数据拦截
    if (data.num_landmarks == 0 || data.num_landmarks > 50000) {
        std::cout << "\033[1;33m[SVIn2Wrapper] Warning: 边缘化数据异常 (Size: " << data.num_landmarks << ")，已触发防御机制，丢弃该帧。\033[0m" << std::endl;
        return;
    }
    for (int i = 0; i < 16; i++) {
        if (std::isnan(data.Tcw_mat[i]) || std::isinf(data.Tcw_mat[i]) || std::abs(data.Tcw_mat[i]) > 1e6) {
            std::cout << "\033[1;31m[SVIn2Wrapper] 警告: 边缘化位姿发散或出现 NaN，抛弃该帧！\033[0m" << std::endl;
            return;
        }
    }

    if (mpSLAM == nullptr) {
        std::cout << "[BackendWriteGate][ERROR] SVIn2ORBWrapper has null System pointer; backend injection skipped." << std::endl;
        return;
    }

    if (mWatchdogCb) {
        mWatchdogCb(data);
    }

    TrackingState state = TrackingState::NORMAL;
    if (mGetStateCb) {
        state = mGetStateCb();
    }

    // 记录上一帧的状态，用于判断状态跃变
    static TrackingState last_state = TrackingState::NORMAL; 

    if (state == TrackingState::NORMAL) {
        ORB_SLAM3::BackendWriteController *pBackendWriteController = nullptr;
        if (mpSLAM != nullptr) {
            pBackendWriteController = mpSLAM->GetBackendWriteController();
        }

        if (pBackendWriteController != nullptr && mbLostEventLatched.load(std::memory_order_acquire)) {
            const std::uint64_t generation = mnLostEventGeneration.load(std::memory_order_acquire);
            bool backendBlockReleased = mbLostRecoveryUploadPending.load(std::memory_order_acquire);
            ORB_SLAM3::LostTopologyReleaseResult releaseResult = ORB_SLAM3::LostTopologyReleaseResult::RELEASED;
            if (!backendBlockReleased) {
                releaseResult = mpSLAM->ReleaseLostBackendBlock(generation);
            }

            while (!backendBlockReleased &&
                   releaseResult == ORB_SLAM3::LostTopologyReleaseResult::DEFERRED) {
                const ORB_SLAM3::LostTopologyReleaseWaitResult waitResult =
                    mpSLAM->WaitForLostBackendBlockRelease(generation);
                if (waitResult == ORB_SLAM3::LostTopologyReleaseWaitResult::RELEASED) {
                    releaseResult = ORB_SLAM3::LostTopologyReleaseResult::RELEASED;
                    break;
                }
                if (waitResult == ORB_SLAM3::LostTopologyReleaseWaitResult::TIMEOUT) {
                    if (!mbLostTransitionTimeoutLogged.exchange(true, std::memory_order_acq_rel)) {
                        std::cout << "[BackendWriteGate][WARNING] LOST topology release remains pending after 30000 ms; backend block remains active." << std::endl;
                    }
                    continue;
                }
                std::cout << "[BackendWriteGate][WARNING] NORMAL backend path remains blocked while LOST topology release is unresolved." << std::endl;
                last_state = state;
                return;
            }

            if (!backendBlockReleased &&
                releaseResult != ORB_SLAM3::LostTopologyReleaseResult::RELEASED) {
                std::cout << "[BackendWriteGate][WARNING] NORMAL backend path remains blocked because LOST topology release was not confirmed." << std::endl;
                last_state = state;
                return;
            }

            if (!backendBlockReleased) {
                mbLostRecoveryUploadPending.store(true, std::memory_order_release);
                backendBlockReleased = true;
            }

            LostUploadBoundarySnapshot boundarySnapshot;
            const LostUploadBoundaryDispatchResult dispatchResult =
                BeginLostUploadBoundaryDispatch(generation, boundarySnapshot);
            const bool dispatchAcquired =
                dispatchResult == LostUploadBoundaryDispatchResult::ACQUIRED_COMMITTED ||
                dispatchResult == LostUploadBoundaryDispatchResult::ACQUIRED_LEGACY;

            if (!dispatchAcquired) {
                std::cout << "[LostUpload][WARNING] Recovery upload dispatch rejected, generation="
                          << generation
                          << ", dispatch_result="
                          << static_cast<int>(dispatchResult)
                          << "; duplicate or stale callback will not start another upload."
                          << std::endl;
                last_state = state;
                return;
            }

            bool uploadPrepared = false;
            try {
                if (mPrepareRecoveryUploadCb) {
                    uploadPrepared = mPrepareRecoveryUploadCb(generation, boundarySnapshot);
                }
            } catch (const std::exception &exception) {
                std::cout << "[LostUpload][ERROR] Recovery upload prepare callback threw, generation="
                          << generation
                          << ", what="
                          << exception.what()
                          << std::endl;
                uploadPrepared = false;
            } catch (...) {
                std::cout << "[LostUpload][ERROR] Recovery upload prepare callback threw an unknown exception, generation="
                          << generation
                          << std::endl;
                uploadPrepared = false;
            }

            bool uploadStarted = false;
            if (uploadPrepared) {
                const bool boundaryCompleted = CompleteLostUploadBoundaryDispatch(generation);
                if (boundaryCompleted) {
                    bool pendingCompleted = false;
                    if (mFinalizeRecoveryUploadCb) {
                        pendingCompleted = mFinalizeRecoveryUploadCb(generation, true);
                    }
                    if (pendingCompleted) {
                        uploadStarted = true;
                    } else {
                        RestoreConsumedLostUploadBoundary(generation);
                    }
                } else {
                    if (mFinalizeRecoveryUploadCb) {
                        mFinalizeRecoveryUploadCb(generation, false);
                    }
                    RollbackLostUploadBoundaryDispatch(generation);
                }
            } else {
                if (mFinalizeRecoveryUploadCb) {
                    mFinalizeRecoveryUploadCb(generation, false);
                }
                RollbackLostUploadBoundaryDispatch(generation);
            }

            if (!uploadStarted) {
                std::cout << "[BackendWriteGate][WARNING] Recovery upload dispatch did not start for LOST generation="
                          << generation
                          << "; boundary and pending event remain available for retry."
                          << std::endl;
                last_state = state;
                return;
            }

            mbLostEventLatched.store(false, std::memory_order_release);
            mbLostTransitionTimeoutLogged.store(false, std::memory_order_release);
            mbLostRecoveryUploadPending.store(false, std::memory_order_release);
            mnLostEventGeneration.store(0, std::memory_order_release);
        }

        auto deferNormalDuringMerging = [&]() {
            const size_t deferred = PushMergeDeferredFrame(data);
            if (deferred > 0 && (deferred <= 5 || deferred % 30 == 0)) {
                std::cout << "[BackendWriteGate] Deferred NORMAL during MERGING, t="
                          << data.timestamp
                          << ", deferred="
                          << deferred
                          << std::endl;
            }

            size_t warningBufferSize = 0;
            {
                std::lock_guard<std::mutex> warningLock(gWarningBufferMutex);
                warningBufferSize = mWarningBuffer.size();
            }

            if (warningBufferSize > 0) {
                if (!mbWarningFlushBlockedLogged) {
                    std::cout << "[BackendWriteGate] Block warning-buffer flush during MERGING; buffer preserved." << std::endl;
                    mbWarningFlushBlockedLogged = true;
                }
            }
        };

        auto deferNormalDuringReplaying = [&]() {
            PushNormalDeferredFrame(data);
        };

        if (pBackendWriteController != nullptr) {
            const ORB_SLAM3::BackendWriteState backendState = pBackendWriteController->GetState();
            if (backendState == ORB_SLAM3::BackendWriteState::MERGING) {
                deferNormalDuringMerging();
                last_state = state;
                return;
            }

            if (backendState == ORB_SLAM3::BackendWriteState::REPLAYING) {
                deferNormalDuringReplaying();
                last_state = state;
                return;
            }
        }

        ORB_SLAM3::BackendInjectionScope backendInjectionScope(
            pBackendWriteController,
            ORB_SLAM3::BackendInjectionSource::LIVE);
        if (pBackendWriteController != nullptr) {
            if (!backendInjectionScope.IsActive()) {
                const ORB_SLAM3::BackendWriteState backendState = pBackendWriteController->GetState();
                if (backendState == ORB_SLAM3::BackendWriteState::MERGING) {
                    deferNormalDuringMerging();
                } else if (backendState == ORB_SLAM3::BackendWriteState::REPLAYING) {
                    deferNormalDuringReplaying();
                } else {
                    std::cout << "[BackendWriteGate][WARNING] LIVE backend injection scope inactive while backend state is IDLE." << std::endl;
                }

                last_state = state;
                return;
            }

            mbWarningFlushBlockedLogged = false;
        }

        std::vector<CachedMarginalizedData> warningSnapshot;
        {
            std::lock_guard<std::mutex> warningLock(gWarningBufferMutex);
            if (mWarningBuffer.size() > 0) {
                warningSnapshot = mWarningBuffer;
                mWarningBuffer.clear();
            }
        }

        for (size_t i = 0; i < warningSnapshot.size(); ++i) {
            MarginalizedData warningData = warningSnapshot[i].toRawData();
            const InjectionResult warningInjectionResult = this->ExecuteInjection(warningData);
            if (warningInjectionResult == InjectionResult::COMMITTED) {
                UpdateLastCommittedBackendTime(
                    warningData.timestamp,
                    CommittedTimeSource::REPLAY);
            }
        }
        const InjectionResult liveInjectionResult = this->ExecuteInjection(data);
        if (liveInjectionResult == InjectionResult::COMMITTED) {
            UpdateLastCommittedBackendTime(
                data.timestamp,
                CommittedTimeSource::LIVE);
        }
    } 
    else if (state == TrackingState::WARNING) {
        CachedMarginalizedData safe_data = MakeCachedMarginalizedData(data);
        {
            std::lock_guard<std::mutex> warningLock(gWarningBufferMutex);
            mWarningBuffer.push_back(safe_data);
        }
    } 
    else if (state == TrackingState::LOST) {
        ORB_SLAM3::BackendWriteController *pBackendWriteController = nullptr;
        if (mpSLAM != nullptr) {
            pBackendWriteController = mpSLAM->GetBackendWriteController();
        }

        bool bFirstLostEvent = false;
        if (pBackendWriteController != nullptr) {
            bool expected = false;
            bFirstLostEvent = mbLostEventLatched.compare_exchange_strong(
                expected,
                true,
                std::memory_order_acq_rel);
        }

        if (pBackendWriteController != nullptr) {
            const ORB_SLAM3::BackendWriteState backendState = pBackendWriteController->GetState();
            if (backendState == ORB_SLAM3::BackendWriteState::MERGING) {
                MarkMergeDeferredBufferInvalid("frontend_lost_during_merging");
                MarkNormalDeferredBufferInvalid("frontend_lost_during_merging");
            } else if (backendState == ORB_SLAM3::BackendWriteState::REPLAYING) {
                MarkMergeDeferredBufferInvalid("frontend_lost_during_replaying");
                MarkNormalDeferredBufferInvalid("frontend_lost_during_replaying");
            }
        }

        {
            std::lock_guard<std::mutex> warningLock(gWarningBufferMutex);
            mWarningBuffer.clear();
        }
        if (bFirstLostEvent) {
            const ORB_SLAM3::LostTopologyRequest request = mpSLAM->RequestLostTopologyTransition();
            if (request.result == ORB_SLAM3::LostTopologyRequestResult::REQUESTED) {
                mnLostEventGeneration.store(request.generation, std::memory_order_release);
                mbLostTransitionTimeoutLogged.store(false, std::memory_order_release);
                if (mLostTopologyRequestedCb) {
                    mLostTopologyRequestedCb(request.generation);
                }

                while (true) {
                    const ORB_SLAM3::LostTopologyWaitResult waitResult =
                        mpSLAM->WaitAndBeginLostTopologyTransition(request.generation);
                    if (waitResult == ORB_SLAM3::LostTopologyWaitResult::OWNER_ACQUIRED) {
                        LostTopologyTransitionGuard transitionGuard(mpSLAM, request.generation);
                        try {
                            if (!SnapshotLostUploadBoundary(request.generation)) {
                                std::cout << "[LostUpload][WARNING] Skip stale LOST boundary snapshot, generation="
                                          << request.generation
                                          << std::endl;
                            }
                            mpSLAM->GetAtlas()->CreateNewMap();
                            transitionGuard.Complete();
                            std::cout << "\033[1;35m[SVIn2Wrapper] 已成功切断拓扑并创建新子图！等待系统恢复...\033[0m" << std::endl;
                        } catch (...) {
                            std::cout << "[BackendWriteGate][ERROR] CreateNewMap threw during LOST topology transition." << std::endl;
                        }
                        break;
                    }
                    if (waitResult == ORB_SLAM3::LostTopologyWaitResult::TIMEOUT) {
                        if (!mbLostTransitionTimeoutLogged.exchange(true, std::memory_order_acq_rel)) {
                            std::cout << "[BackendWriteGate][WARNING] LOST topology transition remains REQUESTED after 30000 ms; backend block remains active and waiting will retry." << std::endl;
                        }
                        continue;
                    }
                    std::cout << "[BackendWriteGate][WARNING] LOST topology transition did not acquire an owner; backend block remains protected." << std::endl;
                    break;
                }
            } else {
                std::cout << "[BackendWriteGate][WARNING] LOST topology request was not accepted." << std::endl;
            }
        }
    }

    // 更新历史状态
    last_state = state; 
}

InjectionResult SVIn2ORBWrapper::ExecuteInjection(const MarginalizedData& data) {
    // [阶段5B修改] accepted pose 的读取、筛选、enqueue 和更新必须保持严格串行。
    std::lock_guard<std::mutex> injectionSequenceLock(mInjectionSequenceMutex);

    Eigen::Map<const Eigen::Matrix4f> Tcw_mat_mapped(data.Tcw_mat);
    Eigen::Matrix3f R_cw = Tcw_mat_mapped.block<3, 3>(0, 0);
    Eigen::Vector3f t_cw = Tcw_mat_mapped.block<3, 1>(0, 3);
    Eigen::Quaternionf q_cw(R_cw);
    q_cw.normalize();
    Sophus::SE3f Tcw_sophus(q_cw, t_cw);

    Sophus::SE3f current_Twc = Tcw_sophus.inverse(); // 当前相机的世界坐标系位姿

    if (mFrontendTrajFile.is_open()) {
        Eigen::Vector3f twc = current_Twc.translation();
        Eigen::Quaternionf q_wc(current_Twc.rotationMatrix());
        
        mFrontendTrajFile << std::fixed << std::setprecision(6) << data.timestamp << " "
                        << twc.x() << " " << twc.y() << " " << twc.z() << " "
                        << q_wc.x() << " " << q_wc.y() << " " << q_wc.z() << " " << q_wc.w() << std::endl;
    }
    
    if (mbLastAcceptedTwcValid) {
        // 1. 计算平移差值 (米)
        float delta_trans = (current_Twc.translation() - mLastAcceptedTwc.translation()).norm();
        
        // 2. 计算旋转差值 (使用 Sophus 李代数对数映射，自带防 NaN 保护)
        Sophus::SO3f delta_R = mLastAcceptedTwc.so3().inverse() * current_Twc.so3();
        float delta_rot_degree = delta_R.log().norm() * 180.0 / M_PI;

        // 设置物理阈值（可根据你的海下运动速度调节，建议：平移 0.1米，旋转 3度）
        const float THRESHOLD_TRANS = 0.1f;  
        const float THRESHOLD_ROT = 3.0f;    

        if (delta_trans < THRESHOLD_TRANS && delta_rot_degree < THRESHOLD_ROT) {
            return InjectionResult::FILTERED;
        }
    }

    // 确认接受该帧后，才开始输出日志并分配内存
    // std::cout << "\033[1;32m ----------------------------------------------------------------\033[0m" << std::endl;
    // std::cout << "\033[1;32m Received OKVIS edge-shifting keyframe! Timestamp: \033[0m" << data.timestamp << std::endl;

    // 1. 从 C 数组还原 2D 坐标，排爆 NaN
    std::vector<cv::KeyPoint> keypointsUn;
    keypointsUn.reserve(data.num_landmarks); // 提前分配内存，加速 push_back
    for (size_t i = 0; i < data.num_landmarks; i++) {
        float u = data.landmarks[i].kp_u;
        float v = data.landmarks[i].kp_v;
        if (std::isnan(u) || std::isnan(v) || std::isinf(u) || std::isinf(v)) { 
            u = 0.0f; v = 0.0f; 
        }
        keypointsUn.push_back(cv::KeyPoint(u, v, 1.0f));
    }

    // 2. 伪造全 0 描述子
    cv::Mat fake_descriptors = cv::Mat::zeros(data.num_landmarks, 32, CV_8U);

    ORB_SLAM3::Map* pCurrentMap = mpSLAM->GetAtlas()->GetCurrentMap();
    ORB_SLAM3::KeyFrameDatabase* pKFDB = mpSLAM->GetKeyFrameDatabase();
    ORB_SLAM3::GeometricCamera* pCamera = mpSLAM->GetCamera();

    // =========================================================================
    // 直到此时，真正需要修改地图拓扑时，才去抢占互斥锁！
    std::unique_lock<std::mutex> lock(pCurrentMap->mMutexMapUpdate);
    
    auto pVocabulary = mpSLAM->GetVocabulary();
    auto pExtractor = mpSLAM->GetExtractor();
    auto distCoef = mpSLAM->GetDistCoef();
    auto bf = mpSLAM->Getbf();
    auto thDepth = mpSLAM->GetThDepth();

    // 假图像注入
    cv::Mat fake_img = cv::Mat::zeros(260, 346, CV_8UC1);
    cv::rectangle(fake_img, cv::Point(150, 100), cv::Point(200, 150), cv::Scalar(255), -1);

    // 调用标准构造函数，完成所有的底层分配和安全边界计算
    ORB_SLAM3::Frame frame(fake_img, data.timestamp, pExtractor, pVocabulary, pCamera, distCoef, bf, thDepth);
    
    // 把假图像提取出的数据，全部覆盖为我们真实的 3D 点云
    frame.N = data.num_landmarks;
    frame.mvKeys = keypointsUn;
    frame.mvKeysUn = keypointsUn;
    frame.mDescriptors = fake_descriptors;
    frame.mvuRight.assign(data.num_landmarks, -1.0f);
    frame.mvDepth.assign(data.num_landmarks, -1.0f);
    frame.mvpMapPoints.assign(data.num_landmarks, nullptr);
    frame.mvbOutlier.assign(data.num_landmarks, false); 
    
    // 重建空间网格（清除假特征占用的网格，重新分配到 346x260 的安全网格中）
    for(int i=0; i<FRAME_GRID_COLS; i++) {
        for(int j=0; j<FRAME_GRID_ROWS; j++) {
            frame.mGrid[i][j].clear();
        }
    }
    for(int i = 0; i < frame.N; i++) {
        const cv::KeyPoint &kp = frame.mvKeysUn[i];
        int nGridPosX, nGridPosY;
        if(frame.PosInGrid(kp, nGridPosX, nGridPosY)) {
            frame.mGrid[nGridPosX][nGridPosY].push_back(i);
        }
    }

    // 生成 KeyFrame
    ORB_SLAM3::KeyFrame* pKF = new ORB_SLAM3::KeyFrame(frame, pCurrentMap, pKFDB);
    pKF->SetPose(Tcw_sophus);
    
    // 禁止局部图优化进行特征匹配
    pKF->mbIsFakeDescriptor = true; 

    int new_pts = 0;
    int tracked_pts = 0;

    std::vector<ORB_SLAM3::MapPoint*> vpAllMPs = pCurrentMap->GetAllMapPoints();
    std::set<ORB_SLAM3::MapPoint*> safe_mps(vpAllMPs.begin(), vpAllMPs.end());
    
    // ABA 幽灵重分配雷达
    static std::map<uint64_t, long unsigned int> mGlobalMapPoints_mnId;

    // 硬绑定 3D MapPoint
    for (size_t i = 0; i < data.num_landmarks; i++) {
        float px = data.landmarks[i].pt_x;
        float py = data.landmarks[i].pt_y;
        float pz = data.landmarks[i].pt_z;
        
        // 3D NaN 拦截
        if (std::isnan(px) || std::isnan(py) || std::isnan(pz) || 
            std::isinf(px) || std::isinf(py) || std::isinf(pz) ||
            std::abs(px) > 1e6 || std::abs(py) > 1e6 || std::abs(pz) > 1e6) {
            continue; 
        }

        uint64_t lm_id = data.landmarks[i].landmark_id;
        ORB_SLAM3::MapPoint* pMP = nullptr;

        // 查询缓存历史点
        if (mGlobalMapPoints.count(lm_id) > 0) {
            ORB_SLAM3::MapPoint* cached_pMP = mGlobalMapPoints[lm_id];
            long unsigned int expected_mnId = mGlobalMapPoints_mnId[lm_id];
            
            // 双重校验：不仅要在安全名单里，它内部的 ID 还必须没变过！
            if (safe_mps.count(cached_pMP) > 0 && cached_pMP->mnId == expected_mnId && !cached_pMP->isBad()) {
                pMP = cached_pMP;
                tracked_pts++;
            }
        }

        // 如果是全新点，或旧点已经被干掉
        if (pMP == nullptr) {
            Eigen::Vector3f pos3d(px, py, pz);
            pMP = new ORB_SLAM3::MapPoint(pos3d, pKF, pCurrentMap);
            mGlobalMapPoints[lm_id] = pMP; 
            mGlobalMapPoints_mnId[lm_id] = pMP->mnId; // 记录出生 ID，防止借尸还魂
            pCurrentMap->AddMapPoint(pMP);
            new_pts++;
        }

        // 核心拓扑注入
        pMP->AddObservation(pKF, i);
        pKF->AddMapPoint(pMP, i);
        pMP->UpdateNormalAndDepth();
        
        // 强制计算描述子，防图优化死机
        pMP->ComputeDistinctiveDescriptors();
    }

    // 5. 将处理好的伪装帧压入地图与局部建图线程
    pCurrentMap->AddKeyFrame(pKF);
    mpSLAM->GetLocalMapping()->InsertKeyFrame(pKF);

    // [阶段5B修改] 只有 LocalMapping enqueue 正常返回后，才更新运动筛选基准。
    mLastAcceptedTwc = current_Twc;
    mbLastAcceptedTwcValid = true;

    // 日志输出
    // std::cout << "\033[1;32m Successfully injected a fake keyframe into LocalMapping! (ID: " << pKF->mnId << ")\033[0m" << std::endl;
    // std::cout << "\033[1;32m Skeleton point cloud statistics -> Total " << data.num_landmarks <<"\n"
    //           << " | New points:" << new_pts 
    //           << " | Shared Historical Perspective: " << tracked_pts << "\033[0m" << std::endl;
    // std::cout << "\033[1;32m ----------------------------------------------------------------\033[0m\n" << std::endl;
    return InjectionResult::COMMITTED;
}
void SVIn2ORBWrapper::InitTrajectorySaver(const std::string& path) {
    mFrontendTrajFile.open(path, std::ios::out);
    if (mFrontendTrajFile.is_open()) {
        std::cout << "\033[1;32m[Wrapper] Successfully opened injection candidate trajectory file: "
                  << path << "\033[0m" << std::endl;
    } else {
        std::cout << "\033[1;31m[Wrapper] Failed to open injection candidate trajectory file!\033[0m"
                  << std::endl;
    }
}

// [阶段2B修改] 初始化完整 OKVIS / SVIn2 前端轨迹文件，数据来自 fullStateCallback -> CacheFrontendPose()。
void SVIn2ORBWrapper::InitOkvisFullTrajectorySaver(const std::string& path) {
    std::lock_guard<std::mutex> lock(mOkvisFullTrajFileMutex);

    if (mOkvisFullTrajFile.is_open()) {
        mOkvisFullTrajFile.close();
    }

    mOkvisFullTrajFile.open(path, std::ios::out);

    if (mOkvisFullTrajFile.is_open()) {
        std::cout << "\033[1;32m[Wrapper] Successfully opened OKVIS full trajectory file: "
                  << path << "\033[0m" << std::endl;

        mOkvisFullTrajFile << "# timestamp tx ty tz qx qy qz qw" << std::endl;
    } else {
        std::cout << "\033[1;31m[Wrapper] Failed to open OKVIS full trajectory file!\033[0m"
                  << std::endl;
    }
}

void SVIn2ORBWrapper::CloseTrajectorySaver() {
    if (mFrontendTrajFile.is_open()) {
        mFrontendTrajFile.close();
    }
}

// [阶段2B修改] 关闭完整 OKVIS / SVIn2 前端轨迹文件。
void SVIn2ORBWrapper::CloseOkvisFullTrajectorySaver() {
    std::lock_guard<std::mutex> lock(mOkvisFullTrajFileMutex);

    if (mOkvisFullTrajFile.is_open()) {
        mOkvisFullTrajFile.close();
    }
}
