#ifndef SVIN2_ORB_WRAPPER_H
#define SVIN2_ORB_WRAPPER_H

#include <Eigen/Dense>
#include <Eigen/StdVector> 
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <cstdint>
#include <functional>
#include <fstream> // [新增] 必须包含此头文件
#include <deque>   // [阶段2B修改] 前端位姿缓存队列
#include <mutex>   // [阶段2B修改] 前端位姿缓存互斥锁
#include <string>  // [阶段2B修改] 轨迹文件路径参数
#include <atomic>
#include "sophus/se3.hpp" // [阶段2B修改] 缓存 OKVIS / SVIn2 前端 Twc 位姿

// 前置声明 ORB-SLAM3 的核心类
namespace ORB_SLAM3 {
    class KeyFrame;
    class MapPoint;
    class System;
}

// 三态状态机枚举
enum class TrackingState {
    NORMAL,
    WARNING,
    LOST
};

// 承载单个3D点及其2D投影的结构体
struct ExtractedLandmark {
    uint64_t landmark_id;          
    float pt_x, pt_y, pt_z;        // 3D 点的纯净内存块
    float kp_u, kp_v;              // 2D 特征点的纯净内存块
};

// 承载整个被边缘化帧的结构体
struct MarginalizedData {
    double timestamp;              
    uint64_t frame_id;             
    float Tcw_mat[16];             // 4x4矩阵退化为 16 个 float 数组
    
    ExtractedLandmark* landmarks;  // 指向内存首地址的纯 C 指针
    size_t num_landmarks;          // 明确告诉后端数组有多长

    // [前端负载诊断] 仅用于 live MarginalizedData 与 Watchdog 的日志对齐，不参与后端注入逻辑。
    uint64_t diagnostic_sequence_id;
    double diagnostic_wall_timestamp;
    double diagnostic_wall_gap_ms;
    double diagnostic_frontend_timestamp_gap_ms;
    bool diagnostic_wall_gap_available;
    bool diagnostic_frontend_gap_available;
    bool diagnostic_log_emitted_by_estimator;
};

// 安全深拷贝结构体，彻底切断底层 C 指针生命周期依赖
struct CachedMarginalizedData {
    double timestamp;
    uint64_t frame_id;
    float Tcw_mat[16];
    std::vector<ExtractedLandmark> landmarks_vec;

    MarginalizedData toRawData() {
        MarginalizedData raw_data;
        raw_data.timestamp = timestamp;
        raw_data.frame_id = frame_id;
        for (int i = 0; i < 16; ++i) {
            raw_data.Tcw_mat[i] = Tcw_mat[i];
        }
        raw_data.num_landmarks = landmarks_vec.size();
        raw_data.landmarks = landmarks_vec.data(); 
        raw_data.diagnostic_sequence_id = 0;
        raw_data.diagnostic_wall_timestamp = 0.0;
        raw_data.diagnostic_wall_gap_ms = 0.0;
        raw_data.diagnostic_frontend_timestamp_gap_ms = 0.0;
        raw_data.diagnostic_wall_gap_available = false;
        raw_data.diagnostic_frontend_gap_available = false;
        raw_data.diagnostic_log_emitted_by_estimator = false;
        return raw_data;
    }
};

// [阶段5B修改] 只有进入 LocalMapping 队列的 edge frontend 帧才算 COMMITTED。
enum class InjectionResult {
    FILTERED = 0,
    COMMITTED = 1
};

// [阶段5B修改] edge backend 已提交时间的只读快照。
struct BackendCommittedTimeSnapshot {
    bool valid = false;
    double timestamp = 0.0;
};

// [阶段5B修改] 与 LOST generation 绑定的上传边界快照。
struct LostUploadBoundarySnapshot {
    bool valid = false;
    double frontend_timestamp = 0.0;
    std::uint64_t generation = 0;
};

// [阶段5B修改] LOST boundary 的一次性 dispatch 状态。
enum class LostUploadBoundaryDispatchState {
    EMPTY = 0,
    AVAILABLE = 1,
    DISPATCHING = 2,
    CONSUMED = 3
};

// [阶段5B修改] 仅 ACQUIRED_* 结果允许启动 recovery upload 准备流程。
enum class LostUploadBoundaryDispatchResult {
    ACQUIRED_COMMITTED = 0,
    ACQUIRED_LEGACY = 1,
    STALE_GENERATION = 2,
    ALREADY_DISPATCHING = 3,
    ALREADY_CONSUMED = 4,
    NOT_AVAILABLE = 5
};

// 跨系统数据中转与实例伪装层
class SVIn2ORBWrapper {
public:
    SVIn2ORBWrapper(ORB_SLAM3::System* pSLAM);
    ~SVIn2ORBWrapper();
    void RegisterStateCallbacks(std::function<void(const MarginalizedData&)> watchdog_cb, std::function<TrackingState()> get_state_cb);
    void RegisterLostTopologyCallbacks(
        std::function<void(std::uint64_t)> lost_requested_cb,
        std::function<bool(std::uint64_t, LostUploadBoundarySnapshot)> prepare_recovery_upload_cb,
        std::function<bool(std::uint64_t, bool)> finalize_recovery_upload_cb);
    void InjectSVIn2MarginalizedData(const MarginalizedData& data);
    void InitTrajectorySaver(const std::string& path);
    void CloseTrajectorySaver();
    void InitOkvisFullTrajectorySaver(const std::string& path); // [阶段2B修改] 保存 fullStateCallback 输出的完整 OKVIS / SVIn2 前端轨迹
    void CloseOkvisFullTrajectorySaver(); // [阶段2B修改] 关闭完整 OKVIS / SVIn2 前端轨迹文件

    // [阶段2B修改] 缓存 OKVIS / SVIn2 前端连续位姿。
    // 该接口由 cloud_edge_demo.cpp 中的 fullStateCallback 调用，独立于 ORB-SLAM3 后端是否 LOST。
    void CacheFrontendPose(const double timestamp, const Sophus::SE3f &Twc);

    // [阶段2B修改] 按时间戳查询最近的 OKVIS / SVIn2 前端位姿。
    // CloudMerging.cc 用该接口构造 frontend/OKVIS 与 CloudMap 的 Sim3 点对对齐。
    bool GetNearestFrontendPose(const double timestamp, const double tolerance, Sophus::SE3f &Twc);

    // [CloudMap校正诊断] 按时间戳查询最近前端位姿，并返回匹配时间差，仅用于诊断导出。
    bool GetNearestFrontendPoseWithTimeGap(const double timestamp, const double tolerance, Sophus::SE3f &Twc, double &timeGap);

    // [BackendWriteGate] CloudMerge 后 replay MERGING 期间缓存的 NORMAL 注入帧。
    enum class ReplayMergeDeferredOutcome {
        COMPLETED,
        SKIPPED_EMPTY,
        SKIPPED_LOST,
        SKIPPED_SHUTDOWN,
        ABORTED_WARNING,
        ABORTED_LOST,
        ABORTED_SHUTDOWN,
        ABORTED_INVALID_BUFFER,
        FAILED
    };

    void ReplayMergeDeferredBuffer();
    ReplayMergeDeferredOutcome ReplayMergeDeferredBufferWithOutcome();
    bool CanReplayDeferredBuffers() const;
    size_t GetMergeDeferredFrameCount();
    // [SecondMergeMatchDebug] 只读复制 MERGING 期间缓存帧的 timestamp，不清空或修改 buffer。
    std::vector<double> GetMergeDeferredTimestampsSnapshot();
    bool HasMergeDeferredFrames();
    bool IsMergeDeferredBufferInvalid();
    void MarkMergeDeferredBufferInvalid(const std::string &reason);
    void ClearMergeDeferredBuffer(const std::string &reason);
private:
    ORB_SLAM3::System* mpSLAM;
    std::map<uint64_t, ORB_SLAM3::MapPoint*> mGlobalMapPoints;
    // 回调对象与深拷贝拦截缓存
    std::function<void(const MarginalizedData&)> mWatchdogCb;
    std::function<TrackingState()> mGetStateCb;
    std::function<void(std::uint64_t)> mLostTopologyRequestedCb;
    std::function<bool(std::uint64_t, LostUploadBoundarySnapshot)> mPrepareRecoveryUploadCb;
    std::function<bool(std::uint64_t, bool)> mFinalizeRecoveryUploadCb;
    std::atomic<bool> mbLostEventLatched;
    std::atomic<bool> mbLostTransitionTimeoutLogged;
    std::atomic<bool> mbLostRecoveryUploadPending;
    std::atomic<std::uint64_t> mnLostEventGeneration;
    std::vector<CachedMarginalizedData> mWarningBuffer;
    std::vector<CachedMarginalizedData> mMergeDeferredBuffer;
    std::mutex mMergeDeferredBufferMutex;
    bool mbMergeDeferredBufferInvalid;
    size_t mnMergeDeferredTotalPushed;
    std::vector<CachedMarginalizedData> mNormalDeferredBuffer;
    std::mutex mNormalDeferredBufferMutex;
    bool mbNormalDeferredBufferInvalid;
    uint64_t mnNormalDeferredTotalPushed;
    uint64_t mnNormalDeferredTotalReplayedAttempted;
    uint64_t mnNormalDeferredDroppedAfterSingleRound;
    bool mbWarningFlushBlockedLogged;
    std::mutex mInjectionSequenceMutex;
    bool mbLastAcceptedTwcValid;
    Sophus::SE3f mLastAcceptedTwc;
    mutable std::mutex mCommittedBackendTimeMutex;
    bool mbCommittedBackendTimeValid;
    double mLastCommittedBackendTime;
    std::uint64_t mnCommittedUpdateCount;
    std::uint64_t mnCommittedInvalidTimestampCount;
    mutable std::mutex mLostUploadBoundaryMutex;
    LostUploadBoundarySnapshot mLostUploadBoundary;
    LostUploadBoundaryDispatchState mLostUploadBoundaryDispatchState;
    std::ofstream mFrontendTrajFile;
    std::ofstream mOkvisFullTrajFile; // [阶段2B修改] okvis_full_traj.txt，记录 fullStateCallback 连续前端轨迹
    std::mutex mOkvisFullTrajFileMutex; // [阶段2B修改] 独立文件锁，避免与前端位姿缓存锁嵌套持有

    // [阶段2B修改] OKVIS / SVIn2 前端位姿缓存，存 Twc，与 KeyFrame::GetPoseInverse() 的方向保持一致。
    struct FrontendPoseData {
        double timestamp;
        Sophus::SE3f Twc;
    };

    std::deque<FrontendPoseData> mFrontendPoseBuffer;
    std::mutex mFrontendPoseMutex;
    size_t mnMaxFrontendPoseBufferSize = 5000;

    // 分离出的原生注入逻辑层
    enum class CommittedTimeSource {
        LIVE = 0,
        REPLAY = 1
    };

    InjectionResult ExecuteInjection(const MarginalizedData& data);
    void UpdateLastCommittedBackendTime(double timestamp, CommittedTimeSource source);
    BackendCommittedTimeSnapshot GetLastCommittedBackendTimeSnapshot() const;
    bool SnapshotLostUploadBoundary(std::uint64_t generation);
    LostUploadBoundaryDispatchResult BeginLostUploadBoundaryDispatch(
        std::uint64_t generation,
        LostUploadBoundarySnapshot &snapshot);
    bool CompleteLostUploadBoundaryDispatch(std::uint64_t generation);
    bool RollbackLostUploadBoundaryDispatch(std::uint64_t generation);
    bool RestoreConsumedLostUploadBoundary(std::uint64_t generation);
    CachedMarginalizedData MakeCachedMarginalizedData(const MarginalizedData& data);
    size_t PushMergeDeferredFrame(const MarginalizedData& data);
    std::vector<CachedMarginalizedData> SnapshotAndClearMergeDeferredBuffer();
    size_t PushNormalDeferredFrame(const MarginalizedData& data);
    std::vector<CachedMarginalizedData> SnapshotAndClearNormalDeferredBuffer();
    void ClearNormalDeferredBuffer(const std::string &reason);
    bool HasNormalDeferredFrames();
    size_t GetNormalDeferredFrameCount();
    void MarkNormalDeferredBufferInvalid(const std::string &reason);
    bool IsNormalDeferredBufferInvalid();
    TrackingState GetCurrentFrontendHealthState();
    size_t ReplayCachedDeferredFrames(std::vector<CachedMarginalizedData> cachedFrames, bool &bAborted, std::string &abortReason);
};

#endif // SVIN2_ORB_WRAPPER_H
