#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#ifndef CLOUD_EDGE_UPLOAD_LIFECYCLE_ONLY

#include "CloudMergeTicket.h"
#include "CloudUploadTransactionGate.h"
#include "RuntimeEnvironment.h"
#include "System.h"
#include "Atlas.h"
#include "CloudImageSampler.h"
#include "Frame.h"
#include "KeyFrame.h"
#include "Map.h"
#include "MapPoint.h"
#include "Converter.h"
#include "ORBVocabulary.h"
#include "SVIn2ORBWrapper.h"

#include <actionlib/client/simple_action_client.h>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <opencv2/imgcodecs.hpp>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Int16.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

#include "cloud_edge_slam/CloudSlamAction.h"
#include "cloud_edge_slam/Descriptor.h"
#include "cloud_edge_slam/Evo.h"
#include "cloud_edge_slam/KeyFrame.h"
#include "cloud_edge_slam/KeyPoint.h"
#include "cloud_edge_slam/MapPoint.h"
#include "cloud_edge_slam/Observation.h"
#include "cloud_edge_slam/Sequence.h"

#include <okvis/Publisher.hpp>
#include <okvis/RosParametersReader.hpp>
#include <okvis/Subscriber.hpp>
#include <okvis/ThreadedKFVio.hpp>

#endif  // CLOUD_EDGE_UPLOAD_LIFECYCLE_ONLY

namespace cloud_edge_slam_sla {

struct UploadLifecycleState {
    std::mutex mutex;
    std::condition_variable cv;
    bool acceptingUploads = true;
    std::size_t activeUploadCount = 0U;
    std::size_t nextWakeCallbackId = 0U;
    std::map<std::size_t, std::function<void()>> wakeCallbacks;
};

enum class LostRecoveryUploadState {
    EMPTY,
    AWAITING_RECOVERY,
    AVAILABLE,
    DISPATCHING,
    CONSUMED
};

// 根据 sendGoal 是否已经启动，确定 LOST 状态机本轮请求的最终状态。
// dispatchStarted=true 仅表示上传事务已启动，不表示 Action 和地图合并已完成。
LostRecoveryUploadState FinalizedLostRecoveryUploadState(
    const bool dispatchStarted) {
    if (dispatchStarted) {
        return LostRecoveryUploadState::CONSUMED;
    }
    return LostRecoveryUploadState::AVAILABLE;
}

class UploadDispatchStartHandshake {
public:
    // 创建“上传事务已经实际启动”的一次性通知对象。
    UploadDispatchStartHandshake()
        : mFuture(mPromise.get_future().share()) {
    }

    // 返回供调用方等待的 future；true 表示 sendGoal 已提交，false 表示启动被拒绝。
    std::shared_future<bool> GetFuture() const {
        return mFuture;
    }

    // 只写入一次上传启动结果，避免异常路径与正常路径重复 set_value。
    void Report(const bool started) {
        std::lock_guard<std::mutex> lock(mMutex);
        if (mReported) {
            return;
        }
        mReported = true;
        mPromise.set_value(started);
    }

private:
    mutable std::mutex mMutex;
    bool mReported = false;
    std::promise<bool> mPromise;
    std::shared_future<bool> mFuture;
};

template <typename RegisterWake, typename SendGoal>
// 先注册 shutdown 唤醒回调，再提交 Goal，并向等待方报告提交结果。
// RegisterWake 失败时不得发送 Goal，防止 shutdown 后产生不可唤醒的上传事务。
bool StartUploadDispatchAndReport(
    UploadDispatchStartHandshake &handshake,
    const RegisterWake &registerWake,
    const SendGoal &sendGoal) {
    try {
        if (!registerWake()) {
            handshake.Report(false);
            return false;
        }
        sendGoal();
    } catch (...) {
        handshake.Report(false);
        throw;
    }

    handshake.Report(true);
    return true;
}

class ActiveUpload {
public:
    // 构造无效 lease，用于拒绝新上传或空生命周期状态。
    ActiveUpload() = default;

    // 持有上传生命周期 lease；析构时自动减少 activeUploadCount。
    explicit ActiveUpload(const std::shared_ptr<UploadLifecycleState> &state)
        : mState(state) {
    }

    // RAII 释放 lease，并在最后一个上传退出时唤醒 shutdown 等待方。
    ~ActiveUpload() {
        Release();
    }

    ActiveUpload(const ActiveUpload &) = delete;
    ActiveUpload &operator=(const ActiveUpload &) = delete;

    // 转移 lease 所有权，源对象不再参与 activeUploadCount 计数。
    ActiveUpload(ActiveUpload &&other) noexcept
        : mState(std::move(other.mState)) {
    }

    // 先释放当前 lease，再接管 other 的 lease。
    ActiveUpload &operator=(ActiveUpload &&other) noexcept {
        if (this != &other) {
            Release();
            mState = std::move(other.mState);
        }
        return *this;
    }

    // 判断本对象是否真正持有一个有效上传 lease。
    explicit operator bool() const {
        return mState != nullptr;
    }

private:
    // 将 activeUploadCount 减一；计数归零时通知 DrainUploadsAndCancelLateGoals。
    void Release() {
        if (mState == nullptr) {
            return;
        }

        bool becameIdle = false;
        {
            std::lock_guard<std::mutex> lock(mState->mutex);
            if (mState->activeUploadCount > 0U) {
                mState->activeUploadCount--;
            }
            becameIdle = mState->activeUploadCount == 0U;
        }
        if (becameIdle) {
            mState->cv.notify_all();
        }
        mState.reset();
    }

    std::shared_ptr<UploadLifecycleState> mState;
};

// 在 acceptingUploads=true 时登记一个活动上传；shutdown 开始后返回无效 lease。
ActiveUpload TryStartUpload(
    const std::shared_ptr<UploadLifecycleState> &state) {
    if (state == nullptr) {
        return ActiveUpload();
    }

    std::lock_guard<std::mutex> lock(state->mutex);
    if (!state->acceptingUploads) {
        return ActiveUpload();
    }

    state->activeUploadCount++;
    return ActiveUpload(state);
}

template <typename Callback>
// 用临时 lease 包裹短时回调，确保回调执行期间 shutdown 不会析构其依赖对象。
bool RunWithActiveUpload(
    const std::shared_ptr<UploadLifecycleState> &state,
    const Callback &callback) {
    ActiveUpload activeUpload = TryStartUpload(state);
    if (!activeUpload) {
        return false;
    }

    callback();
    return true;
}

// 关闭新的上传入口；不会取消已开始的上传，后续由唤醒和 drain 流程处理。
void StopAcceptingUploads(
    const std::shared_ptr<UploadLifecycleState> &state) {
    if (state == nullptr) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->acceptingUploads = false;
    }
    state->cv.notify_all();
}

// 为阻塞中的上传登记 shutdown 唤醒回调，返回非零回调编号。
// 若 gate 已关闭，立即执行 callback 并返回 0。
std::size_t RegisterUploadWakeCallback(
    const std::shared_ptr<UploadLifecycleState> &state,
    const std::function<void()> &callback) {
    if (state == nullptr || !callback) {
        return 0U;
    }

    std::size_t callbackId = 0U;
    bool wakeImmediately = false;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (!state->acceptingUploads) {
            wakeImmediately = true;
        } else {
            state->nextWakeCallbackId++;
            callbackId = state->nextWakeCallbackId;
            state->wakeCallbacks[callbackId] = callback;
        }
    }

    if (wakeImmediately) {
        callback();
    }
    return callbackId;
}

// 移除已完成事务的 shutdown 唤醒回调，避免保存过期 context。
void UnregisterUploadWakeCallback(
    const std::shared_ptr<UploadLifecycleState> &state,
    const std::size_t callbackId) {
    if (state == nullptr || callbackId == 0U) {
        return;
    }

    std::lock_guard<std::mutex> lock(state->mutex);
    state->wakeCallbacks.erase(callbackId);
}

// 复制并调用所有等待 Action/ticket 的唤醒回调；回调在生命周期 mutex 外执行。
void WakeUploadWaiters(
    const std::shared_ptr<UploadLifecycleState> &state) {
    if (state == nullptr) {
        return;
    }

    std::vector<std::function<void()>> callbacks;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        for (const auto &entry : state->wakeCallbacks) {
            callbacks.push_back(entry.second);
        }
    }

    for (const std::function<void()> &callback : callbacks) {
        callback();
    }
    state->cv.notify_all();
}

// 阻塞等待所有已开始的上传 lease 释放，用于 shutdown 的排空阶段。
void WaitForActiveUploads(
    const std::shared_ptr<UploadLifecycleState> &state) {
    if (state == nullptr) {
        return;
    }

    std::unique_lock<std::mutex> lock(state->mutex);
    state->cv.wait(lock, [state]() {
        return state->activeUploadCount == 0U;
    });
}

template <typename CancelGoals>
// 先等待在途上传退出，再取消可能在等待期间新增的 Action Goal。
void DrainUploadsAndCancelLateGoals(
    const std::shared_ptr<UploadLifecycleState> &state,
    const CancelGoals &cancelGoals) {
    WaitForActiveUploads(state);
    cancelGoals();
}

// 返回当前活动上传数量；测试和诊断使用，不改变生命周期状态。
std::size_t GetActiveUploadCountForTesting(
    const std::shared_ptr<UploadLifecycleState> &state) {
    if (state == nullptr) {
        return 0U;
    }

    std::lock_guard<std::mutex> lock(state->mutex);
    return state->activeUploadCount;
}

// 查询上传 gate 是否仍接受新请求；名称保留以兼容生命周期单元测试。
bool IsAcceptingUploadsForTesting(
    const std::shared_ptr<UploadLifecycleState> &state) {
    if (state == nullptr) {
        return false;
    }

    std::lock_guard<std::mutex> lock(state->mutex);
    return state->acceptingUploads;
}

template <typename Result>
struct TicketBatchResult {
    bool succeeded = false;
    bool waitFailed = false;
    std::vector<Result> results;
};

template <typename Ticket, typename Result, typename Waiter, typename Acceptor>
// 顺序等待一批 CloudMap completion ticket，并按 acceptor 判定每个终态是否成功。
// 任一 ticket 等待失败或终态不被接受时，batch.succeeded=false。
TicketBatchResult<Result> WaitForTicketBatch(
    const std::vector<Ticket> &tickets,
    const Waiter &waiter,
    const Acceptor &acceptor) {
    TicketBatchResult<Result> batch;
    if (tickets.empty()) {
        return batch;
    }

    batch.succeeded = true;
    for (const Ticket &ticket : tickets) {
        Result result;
        if (!waiter(ticket, result)) {
            batch.succeeded = false;
            batch.waitFailed = true;
            continue;
        }
        batch.results.push_back(result);
        if (!acceptor(result)) {
            batch.succeeded = false;
        }
    }
    return batch;
}

}  // namespace cloud_edge_slam_sla

#ifndef CLOUD_EDGE_UPLOAD_LIFECYCLE_ONLY


extern SVIn2ORBWrapper *pSVIn2ORBWrapper;

namespace cloud_edge_slam_sla {

using CloudClient = actionlib::SimpleActionClient<cloud_edge_slam::CloudSlamAction>;

namespace bfs = boost::filesystem;

const char kUdfRoot[] =
    "/home/lhf/vc-SLAM_ws/Edge-SLAM/Cloud-Edge-SLAM/"
    "Cloud-Edge-SLAM-master";

// 检查 UDF 桥接文件是否已经生成；path 为绝对或相对文件路径。
bool FileExists(const std::string &path) {
    struct stat fileStatus;
    return stat(path.c_str(), &fileStatus) == 0;
}

// 以 50 ms 周期等待外部 UDF 桥接产物；shutdown 后立即退出，避免析构阶段无限等待。
bool WaitForFile(
    const std::string &path,
    const std::shared_ptr<UploadLifecycleState> &lifecycleState) {
    while (ros::ok() && IsAcceptingUploadsForTesting(lifecycleState)) {
        if (FileExists(path)) {
            return true;
        }
        usleep(50 * 1000);
    }
    return false;
}

// 将 ROS 位姿转换为 ORB-SLAM3 使用的 Sophus SE3；输入为 camera-from-world 位姿。
Sophus::SE3f ToSophusPose(const geometry_msgs::Pose &rosPose) {
    Eigen::Quaternionf rotation(
        rosPose.orientation.w,
        rosPose.orientation.x,
        rosPose.orientation.y,
        rosPose.orientation.z);
    Eigen::Vector3f translation(
        rosPose.position.x,
        rosPose.position.y,
        rosPose.position.z);
    return Sophus::SE3f(rotation, translation);
}

// 将 ROS CloudMap 深拷贝为 ORB-SLAM3 Map。
// oldUdf=true 时等待旧 UDF 的 test2.xyz；返回 Map 所有权转交调用方/CloudMerging。
ORB_SLAM3::Map *ConvertCloudMap(
    ORB_SLAM3::System &slam,
    const cloud_edge_slam::CloudMapConstPtr &rosMap,
    const bool oldUdf,
    const std::shared_ptr<UploadLifecycleState> &lifecycleState) {
    if (!rosMap) {
        return nullptr;
    }

    std::map<std::string, cv::Mat> lastCloudImages =
        slam.GetCloudImageSampler()->mdLastCloudProcessImages;
    ORB_SLAM3::ORBVocabulary *vocabulary = slam.GetVocabulary();
    ORB_SLAM3::ORBextractor *extractor = slam.GetExtractor();
    ORB_SLAM3::GeometricCamera *camera = slam.GetCamera();
    cv::Mat distortion = slam.GetDistCoef();
    const float bf = slam.Getbf();
    const float thresholdDepth = slam.GetThDepth();

    // 云图 Map ID 从 1000 起递增，避免与本地 Atlas 初始地图编号冲突。
    static unsigned long nextCloudMapId = 1000;
    ORB_SLAM3::Map *cloudMap = new ORB_SLAM3::Map(nextCloudMapId++, true);
    cloudMap->edgeFrontMapMnId = rosMap->edge_front_map_mnid;
    cloudMap->edgeBackMapMnId = rosMap->edge_back_map_mnid;

    // 云关键帧 ID 采用独立递减区间，防止与边端 KeyFrame ID 重叠。
    static unsigned long nextCloudKeyFrameId = 3565536;
    nextCloudKeyFrameId -= 300000;
    if (nextCloudKeyFrameId < 100000) {
        nextCloudKeyFrameId = 3565536;
    }

    std::vector<ORB_SLAM3::KeyFrame *> keyFrames;
    keyFrames.reserve(rosMap->key_frames.size());
    for (const cloud_edge_slam::KeyFrame &rosKeyFrame : rosMap->key_frames) {
        std::vector<cv::KeyPoint> keyPoints;
        keyPoints.reserve(rosKeyFrame.key_points.size());
        for (const cloud_edge_slam::KeyPoint &rosKeyPoint : rosKeyFrame.key_points) {
            cv::KeyPoint keyPoint;
            if (std::isnan(rosKeyPoint.x)) {
                keyPoint.pt.x = 0.0F;
            } else {
                keyPoint.pt.x = rosKeyPoint.x;
            }
            if (std::isnan(rosKeyPoint.y)) {
                keyPoint.pt.y = 0.0F;
            } else {
                keyPoint.pt.y = rosKeyPoint.y;
            }
            keyPoints.push_back(keyPoint);
        }

        geometry_msgs::Pose pose = rosKeyFrame.pose_cw;
        if (std::isnan(pose.position.x) ||
            std::isnan(pose.position.y) ||
            std::isnan(pose.position.z)) {
            pose.position.x = 0.0F;
            pose.position.y = 0.0F;
            pose.position.z = 0.0F;
        }

        const float quaternionNorm = std::sqrt(
            pose.orientation.w * pose.orientation.w +
            pose.orientation.x * pose.orientation.x +
            pose.orientation.y * pose.orientation.y +
            pose.orientation.z * pose.orientation.z);
        if (std::isnan(quaternionNorm) || quaternionNorm < 1e-5F) {
            pose.orientation.w = 1.0F;
            pose.orientation.x = 0.0F;
            pose.orientation.y = 0.0F;
            pose.orientation.z = 0.0F;
        }

        cv::Mat descriptors = cv::Mat::zeros(
            static_cast<int>(keyPoints.size()), 32, CV_32FC1);
        ORB_SLAM3::Frame frame(
            keyPoints,
            rosKeyFrame.mTimeStamp,
            extractor,
            vocabulary,
            camera,
            distortion,
            bf,
            thresholdDepth);
        frame.mnId = nextCloudKeyFrameId--;
        frame.mDescriptors = descriptors;
        frame.mvpMapPoints = std::vector<ORB_SLAM3::MapPoint *>(
            rosKeyFrame.mvp_map_points_index.size(),
            nullptr);

        ORB_SLAM3::KeyFrame *keyFrame = new ORB_SLAM3::KeyFrame(
            frame,
            cloudMap,
            nullptr);
        keyFrame->SetPose(ToSophusPose(pose));
        keyFrame->SetCloudFlag();

        const std::string imageTimestamp = std::to_string(frame.mTimeStamp);
        const auto imageIt = lastCloudImages.find(imageTimestamp);
        if (imageIt != lastCloudImages.end()) {
            keyFrame->SetImgGray(imageIt->second);
        }
        keyFrames.push_back(keyFrame);
    }

    if (oldUdf) {
        const std::string pointPath =
            std::string(kUdfRoot) + "/trans_point/test.xyz";
        const std::string transformedPointPath =
            std::string(kUdfRoot) + "/trans_point/test2.xyz";
        std::ofstream pointStream(pointPath);
        for (const cloud_edge_slam::MapPoint &mapPoint : rosMap->map_points) {
            pointStream << mapPoint.point.x << " "
                        << mapPoint.point.y << " "
                        << mapPoint.point.z << std::endl;
        }
        pointStream.close();
        if (!WaitForFile(transformedPointPath, lifecycleState)) {
            delete cloudMap;
            throw std::runtime_error(
                "old UDF bridge stopped before trans_point/test2.xyz was ready");
        }
    }

    std::vector<ORB_SLAM3::MapPoint *> mapPoints;
    mapPoints.reserve(rosMap->map_points.size());
    for (const cloud_edge_slam::MapPoint &rosMapPoint : rosMap->map_points) {
        if (rosMapPoint.ref_keyframe_id < 0 ||
            static_cast<size_t>(rosMapPoint.ref_keyframe_id) >= keyFrames.size()) {
            mapPoints.push_back(nullptr);
            continue;
        }

        Eigen::Vector3f position(
            rosMapPoint.point.x,
            rosMapPoint.point.y,
            rosMapPoint.point.z);
        if (std::isnan(position.x()) ||
            std::isnan(position.y()) ||
            std::isnan(position.z())) {
            position = Eigen::Vector3f::Zero();
        }

        ORB_SLAM3::MapPoint *mapPoint = new ORB_SLAM3::MapPoint(
            nextCloudKeyFrameId--,
            position,
            keyFrames[rosMapPoint.ref_keyframe_id],
            cloudMap);
        mapPoint->isEdge = false;
        mapPoints.push_back(mapPoint);
    }

    for (size_t mapPointIndex = 0; mapPointIndex < mapPoints.size(); ++mapPointIndex) {
        ORB_SLAM3::MapPoint *mapPoint = mapPoints[mapPointIndex];
        if (mapPoint == nullptr) {
            continue;
        }

        for (const cloud_edge_slam::Observation &observation :
             rosMap->map_points[mapPointIndex].observations) {
            if (observation.keyframe_id < 0 ||
                static_cast<size_t>(observation.keyframe_id) >= keyFrames.size()) {
                continue;
            }
            ORB_SLAM3::KeyFrame *keyFrame = keyFrames[observation.keyframe_id];
            if (observation.refer_keypoint_index < 0 ||
                static_cast<size_t>(observation.refer_keypoint_index) >=
                    keyFrame->mvKeys.size()) {
                continue;
            }
            mapPoint->AddObservation(
                keyFrame,
                observation.refer_keypoint_index);
        }
        mapPoint->ComputeDistinctiveDescriptors();
        mapPoint->UpdateNormalAndDepth();
    }

    for (size_t keyFrameIndex = 0; keyFrameIndex < keyFrames.size(); ++keyFrameIndex) {
        ORB_SLAM3::KeyFrame *keyFrame = keyFrames[keyFrameIndex];
        keyFrame->ComputeBoW();
        int keyPointIndex = 0;
        for (const int mapPointIndex :
             rosMap->key_frames[keyFrameIndex].mvp_map_points_index) {
            if (mapPointIndex >= 0 &&
                static_cast<size_t>(mapPointIndex) < mapPoints.size() &&
                mapPoints[mapPointIndex] != nullptr) {
                keyFrame->AddMapPoint(mapPoints[mapPointIndex], keyPointIndex);
            }
            keyPointIndex++;
        }
        keyFrame->UpdateCloudConnections();
        cloudMap->AddKeyFrame(keyFrame);
    }

    for (ORB_SLAM3::MapPoint *mapPoint : mapPoints) {
        if (mapPoint != nullptr) {
            cloudMap->AddMapPoint(mapPoint);
        }
    }

    return cloudMap;
}

struct CloudUploadResult {
    bool actionSucceeded = false;
    bool mergeSucceeded = false;
    std::vector<ORB_SLAM3::CloudMergeResult> mergeResults;
    std::string failureReason;
};

struct CloudUploadContext {
    std::mutex mutex;
    std::condition_variable condition;
    bool actionFinished = false;
    bool actionSucceeded = false;
    bool shutdownRequested = false;
    std::string failureReason;
    cloud_edge_slam::CloudSlamResultConstPtr actionResult;
};

// 将 Goal、JPEG 发布、Action 返回和 CloudMap ticket 等待串成一个互斥上传事务。
class CloudUploadTransactionCoordinator {
public:
    using TicketRegistrar = std::function<std::vector<ORB_SLAM3::CloudMergeTicketPtr>(
        const cloud_edge_slam::CloudSlamResultConstPtr &result)>;
    using TicketWaiter = std::function<bool(
        const ORB_SLAM3::CloudMergeTicketPtr &ticket,
        ORB_SLAM3::CloudMergeResult &result)>;

    // 保存事务 gate 和上传生命周期；实际 gate lease 在 DispatchCloudUpload 内持有。
    explicit CloudUploadTransactionCoordinator(
        ORB_SLAM3::CloudUploadTransactionGate &transactionGate,
        const std::shared_ptr<UploadLifecycleState> &lifecycleState)
        : mTransactionGate(transactionGate),
          mLifecycleState(lifecycleState) {
    }

    // 执行一次完整上传事务。
    // images 为待发布 JPEG；sendGoal 后固定等待 1 秒建立云端接收上下文，再以 100 Hz 发布。
    // ticketRegistrar 登记 Action 返回地图，ticketWaiter 等待本轮所有地图合并终态。
    CloudUploadResult DispatchCloudUpload(
        CloudClient &client,
        ros::Publisher &imagePublisher,
        const cloud_edge_slam::CloudSlamGoal &goal,
        const std::vector<sensor_msgs::CompressedImagePtr> &images,
        const TicketRegistrar &ticketRegistrar,
        const TicketWaiter &ticketWaiter,
        UploadDispatchStartHandshake &startHandshake) {
        ORB_SLAM3::CloudUploadTransactionGate::Lease transactionLease =
            mTransactionGate.Acquire();
        if (!IsAcceptingUploadsForTesting(mLifecycleState)) {
            startHandshake.Report(false);
            CloudUploadResult result;
            result.failureReason = "cloud upload gate is closed";
            return result;
        }

        std::shared_ptr<CloudUploadContext> context =
            std::make_shared<CloudUploadContext>();
        std::weak_ptr<CloudUploadContext> weakContext(context);
        std::size_t wakeCallbackId = 0U;

        try {
            const bool dispatchStarted = StartUploadDispatchAndReport(
                startHandshake,
                [&]() {
                    wakeCallbackId = RegisterUploadWakeCallback(
                        mLifecycleState,
                        [weakContext]() {
                            std::shared_ptr<CloudUploadContext> lockedContext =
                                weakContext.lock();
                            if (lockedContext == nullptr) {
                                return;
                            }
                            {
                                std::lock_guard<std::mutex> lock(
                                    lockedContext->mutex);
                                lockedContext->shutdownRequested = true;
                            }
                            lockedContext->condition.notify_all();
                        });
                    return wakeCallbackId != 0U;
                },
                [&]() {
                    client.sendGoal(
                        goal,
                        [context](
                            const actionlib::SimpleClientGoalState &state,
                            const cloud_edge_slam::CloudSlamResultConstPtr
                                &result) {
                            ActionFinishCb(context, state, result);
                        },
                        CloudClient::SimpleActiveCallback(),
                        CloudClient::SimpleFeedbackCallback());
                });
            if (!dispatchStarted) {
                CloudUploadResult result;
                result.failureReason =
                    "cloud upload cancelled during shutdown";
                return result;
            }
        } catch (...) {
            UnregisterUploadWakeCallback(mLifecycleState, wakeCallbackId);
            throw;
        }

        // The initial delay is retained because the current cloud protocol may
        // use it to establish the image-receive context for this Goal.
        // 保留 1 秒协议等待，给云端建立当前 Goal 的图像接收上下文。
        ros::Duration(1.0).sleep();
        // 统一 JPEG 发送节拍：100 Hz，即相邻图片最短约 10 ms。
        ros::Rate burstRate(100.0);
        for (const sensor_msgs::CompressedImagePtr &image : images) {
            imagePublisher.publish(image);
            ros::spinOnce();
            burstRate.sleep();
        }

        CloudUploadResult result = WaitForTransaction(
            context,
            ticketRegistrar,
            ticketWaiter);
        UnregisterUploadWakeCallback(mLifecycleState, wakeCallbackId);
        return result;
    }

private:
    // Action done callback：仅保存结果并唤醒等待线程，不进入上传事务 gate。
    static void ActionFinishCb(
        const std::shared_ptr<CloudUploadContext> &context,
        const actionlib::SimpleClientGoalState &state,
        const cloud_edge_slam::CloudSlamResultConstPtr &result) {
        {
            std::lock_guard<std::mutex> lock(context->mutex);
            if (context->shutdownRequested) {
                return;
            }
            context->actionSucceeded =
                state == actionlib::SimpleClientGoalState::SUCCEEDED &&
                result != nullptr;
            if (context->actionSucceeded) {
                context->actionResult = result;
            } else {
                context->failureReason = state.toString();
            }
            context->actionFinished = true;
        }
        context->condition.notify_all();
    }

    // 等待 Action、登记全部 CloudMap ticket，并等待每个 ticket 的合并完成。
    // Action 成功但未产生 ticket 仍判失败，避免将空地图合并误报为成功。
    static CloudUploadResult WaitForTransaction(
        const std::shared_ptr<CloudUploadContext> &context,
        const TicketRegistrar &ticketRegistrar,
        const TicketWaiter &ticketWaiter) {
        CloudUploadResult uploadResult;
        cloud_edge_slam::CloudSlamResultConstPtr actionResult;
        std::vector<ORB_SLAM3::CloudMergeTicketPtr> tickets;

        {
            std::unique_lock<std::mutex> lock(context->mutex);
            context->condition.wait(lock, [context]() {
                return context->actionFinished || context->shutdownRequested;
            });
            if (context->shutdownRequested) {
                uploadResult.failureReason = "cloud upload cancelled during shutdown";
                return uploadResult;
            }
            uploadResult.actionSucceeded = context->actionSucceeded;
            uploadResult.failureReason = context->failureReason;
            actionResult = context->actionResult;
        }

        if (!uploadResult.actionSucceeded) {
            return uploadResult;
        }

        try {
            tickets = ticketRegistrar(actionResult);
        } catch (const std::exception &exception) {
            uploadResult.actionSucceeded = false;
            uploadResult.failureReason = exception.what();
            return uploadResult;
        } catch (...) {
            uploadResult.actionSucceeded = false;
            uploadResult.failureReason =
                "unknown exception while registering CloudMap tickets";
            return uploadResult;
        }

        if (tickets.empty()) {
            uploadResult.actionSucceeded = false;
            uploadResult.failureReason =
                "CloudSlam Action succeeded but registered zero CloudMap tickets";
            return uploadResult;
        }

        const TicketBatchResult<ORB_SLAM3::CloudMergeResult> batch =
            WaitForTicketBatch<
                ORB_SLAM3::CloudMergeTicketPtr,
                ORB_SLAM3::CloudMergeResult>(
                tickets,
                ticketWaiter,
                [](const ORB_SLAM3::CloudMergeResult &mergeResult) {
                    return mergeResult.outcome ==
                               ORB_SLAM3::CloudMergeOutcome::MERGED_REPLAY_COMPLETED ||
                           mergeResult.outcome ==
                               ORB_SLAM3::CloudMergeOutcome::MERGED_REPLAY_SKIPPED_LOST ||
                           mergeResult.outcome ==
                               ORB_SLAM3::CloudMergeOutcome::MERGED_REPLAY_SKIPPED_SHUTDOWN ||
                           mergeResult.outcome ==
                               ORB_SLAM3::CloudMergeOutcome::MERGE_COMPLETED_LAND_AIR;
                });
        uploadResult.mergeSucceeded = batch.succeeded;
        uploadResult.mergeResults = batch.results;
        if (batch.waitFailed) {
            uploadResult.failureReason = "CloudMap ticket wait failed";
        }

        return uploadResult;
    }

    ORB_SLAM3::CloudUploadTransactionGate &mTransactionGate;
    std::shared_ptr<UploadLifecycleState> mLifecycleState;
};

// 创建 Action 结果→ORB Map→completion ticket 的登记器。
// oldUdf/newUdf 选择既有 UDF 桥接方式；每张转换成功的 CloudMap 对应一个 ticket。
CloudUploadTransactionCoordinator::TicketRegistrar MakeTicketRegistrar(
    ORB_SLAM3::System &slam,
    const bool oldUdf,
    const bool newUdf,
    const std::shared_ptr<UploadLifecycleState> &lifecycleState) {
    return [&slam, oldUdf, newUdf, lifecycleState](
               const cloud_edge_slam::CloudSlamResultConstPtr &result) {
        if (!result) {
            throw std::runtime_error("CloudSlam Action returned no result");
        }

        std::vector<ORB_SLAM3::CloudMergeTicketPtr> tickets;
        auto registerMap = [&slam, oldUdf, lifecycleState, &tickets](
                               const cloud_edge_slam::CloudMap &map) {
            cloud_edge_slam::CloudMapConstPtr rosMap(
                new cloud_edge_slam::CloudMap(map));
            ORB_SLAM3::Map *cloudMap = ConvertCloudMap(
                slam,
                rosMap,
                oldUdf,
                lifecycleState);
            if (cloudMap == nullptr) {
                throw std::runtime_error("CloudMap conversion failed");
            }
            ORB_SLAM3::CloudMergeTicketPtr ticket =
                slam.InsertCloudMapWithTicket(cloudMap);
            tickets.push_back(ticket);
            std::uint64_t ticketSequence = 0U;
            if (ticket != nullptr) {
                ticketSequence = ticket->GetSequence();
            }
            ROS_INFO_STREAM(
                "[CloudUpload] CloudMap 已登记合并队列：ticket="
                << ticketSequence
                << ", cloud_map=" << cloudMap->GetId()
                << ", edge_front=" << cloudMap->edgeFrontMapMnId
                << ", edge_back=" << cloudMap->edgeBackMapMnId);
        };

        if (!newUdf) {
            registerMap(result->map);
            return tickets;
        }

        const std::string pointPath =
            std::string(kUdfRoot) + "/trans_point/test.xyz";
        const std::string idPath =
            std::string(kUdfRoot) + "/trans_others/id.txt";
        const std::string readyPath =
            std::string(kUdfRoot) + "/trans_others/data.ready";
        const std::string bagPath =
            std::string(kUdfRoot) + "/trans_bag/test.bag";
        std::remove(pointPath.c_str());
        std::remove(bagPath.c_str());
        std::remove(readyPath.c_str());

        std::ofstream pointStream(pointPath);
        for (const cloud_edge_slam::MapPoint &mapPoint : result->map.map_points) {
            pointStream << mapPoint.point.x << " "
                        << mapPoint.point.y << " "
                        << mapPoint.point.z << std::endl;
        }
        pointStream.close();

        std::ofstream idStream(idPath);
        idStream << result->map.edge_front_map_mnid << std::endl;
        idStream << result->map.edge_back_map_mnid << std::endl;
        idStream.close();
        sync();
        std::ofstream readyStream(readyPath);
        readyStream.close();

        if (!WaitForFile(bagPath, lifecycleState)) {
            throw std::runtime_error(
                "new UDF bridge stopped before trans_bag/test.bag was ready");
        }

        // 保留旧 new-UDF 外部进程完成 bag 写入后的 3 秒稳定等待。
        ros::Duration(3.0).sleep();
        ROS_INFO_STREAM(
            "[CloudUpload][UDF] 开始读取 Rosbag 合并结果：" << bagPath);
        rosbag::Bag resultBag;
        resultBag.open(bagPath, rosbag::bagmode::Read);
        std::vector<std::string> topics;
        topics.push_back("/test");
        rosbag::View view(resultBag, rosbag::TopicQuery(topics));
        std::size_t rosbagMapCount = 0U;
        for (const rosbag::MessageInstance &message : view) {
            cloud_edge_slam::CloudSlamResultConstPtr bridgedResult =
                message.instantiate<cloud_edge_slam::CloudSlamResult>();
            if (bridgedResult != nullptr) {
                registerMap(bridgedResult->map);
                rosbagMapCount++;
            }
        }
        resultBag.close();
        ROS_INFO_STREAM(
            "[CloudUpload][UDF] Rosbag 读取完成：CloudMap="
            << rosbagMapCount << "，已登记 ticket=" << tickets.size());
        std::remove(bagPath.c_str());
        std::remove(pointPath.c_str());
        return tickets;
    };
}

// 创建 System ticket 等待适配器，使上传层不直接访问 CloudMerging 队列。
CloudUploadTransactionCoordinator::TicketWaiter MakeTicketWaiter(
    ORB_SLAM3::System &slam) {
    return [&slam](
               const ORB_SLAM3::CloudMergeTicketPtr &ticket,
               ORB_SLAM3::CloudMergeResult &result) {
        return slam.WaitForCloudMergeCompletion(ticket, result);
    };
}

// 读取并校验唯一的运行环境参数 runtime_environment（sea、land 或 air）。
ORB_SLAM3::RuntimeEnvironment ParseRuntimeEnvironmentParameter(
    ros::NodeHandle &nodeHandle) {
    std::string runtimeEnvironmentValue;
    if (!nodeHandle.getParam("runtime_environment", runtimeEnvironmentValue)) {
        ROS_FATAL("Missing required private parameter: runtime_environment");
        throw std::runtime_error(
            "missing required private parameter: runtime_environment");
    }

    try {
        return ORB_SLAM3::ParseRuntimeEnvironment(runtimeEnvironmentValue);
    } catch (const std::exception &exception) {
        ROS_FATAL_STREAM("Invalid runtime_environment: " << exception.what());
        throw;
    }
}

template <typename T>
// 读取必填私有 ROS 参数；缺失时记录 FATAL 并抛异常，避免使用未初始化配置。
void RequireRosParameter(
    ros::NodeHandle &nodeHandle,
    const std::string &name,
    T &value) {
    if (!nodeHandle.getParam(name, value)) {
        ROS_FATAL_STREAM("Missing required private parameter: " << name);
        throw std::runtime_error("missing required private parameter: " + name);
    }
}

struct RuntimeConfig {
    // 统一运行环境；决定 sea 的 OKVIS 路径或 land/air 的常规图像路径。
    ORB_SLAM3::RuntimeEnvironment environment =
        ORB_SLAM3::RuntimeEnvironment::SEA;
    // 云端 Action 名称与 ORB/输入配置路径。
    std::string cloudTopicName;
    std::string vocabularyPath;
    std::string settingPath;
    std::string dataType;
    std::string dataPath;
    std::string resultPath;
    // land/air ROS 图像订阅话题；sea 同时作为原始图像缓存话题。
    std::string rawImageTopic = "/camera/rgb/image_color";
    // 云图合并、云端连接和本地调试行为开关。
    bool cloudMerge = true;
    bool saveCloudBag = false;
    bool realOnline = false;
    bool mergeAnyway = false;
    bool cloudOnline = true;
    bool waitCloudResult = true;
    bool keyFrameCulling = false;
    bool oldUdf = false;
    bool newUdf = false;
    // 主循环额外睡眠，单位毫秒；0 表示不额外限速。
    float mainLoopSleepMs = 0.0F;
    // CloudImageSampler 的前后子图 keyframe 数量与最短时间窗；时间单位为秒。
    int samplerEdgeFrontKeyFrames = 0;
    int samplerEdgeBackKeyFrames = 0;
    float samplerEdgeFrontMinTime = 0.0F;
    float samplerEdgeBackMinTime = 0.0F;
    // 光流采样 PD 控制参数与阈值。
    float samplerPdKp = 0.0F;
    float samplerPdKd = 0.0F;
    float samplerPdThreshold = 0.0F;
};

// 集中读取统一节点的 ROS 参数，形成 System、Grabber 和运行路径的配置快照。
// main_loop_sleep_ms 单位为毫秒；sampler_*_min_time 单位为秒。
RuntimeConfig LoadRuntimeConfig(ros::NodeHandle &nodeHandle) {
    RuntimeConfig config;
    config.environment = ParseRuntimeEnvironmentParameter(nodeHandle);
    RequireRosParameter(nodeHandle, "cloud_topic_name", config.cloudTopicName);
    RequireRosParameter(nodeHandle, "vocabulary_path", config.vocabularyPath);
    RequireRosParameter(nodeHandle, "setting_path", config.settingPath);
    RequireRosParameter(nodeHandle, "data_type", config.dataType);
    RequireRosParameter(nodeHandle, "data_path", config.dataPath);
    RequireRosParameter(nodeHandle, "result_path", config.resultPath);
    RequireRosParameter(nodeHandle, "cloud_merge", config.cloudMerge);
    RequireRosParameter(nodeHandle, "save_cloud_bag", config.saveCloudBag);
    RequireRosParameter(nodeHandle, "real_online", config.realOnline);
    RequireRosParameter(nodeHandle, "merge_anyway", config.mergeAnyway);
    RequireRosParameter(nodeHandle, "cloud_online", config.cloudOnline);
    RequireRosParameter(nodeHandle, "wait_cloud_result", config.waitCloudResult);
    RequireRosParameter(nodeHandle, "main_loop_sleep_ms", config.mainLoopSleepMs);
    RequireRosParameter(nodeHandle, "kf_culling", config.keyFrameCulling);
    RequireRosParameter(nodeHandle, "old_udf_cloud_edge", config.oldUdf);
    RequireRosParameter(nodeHandle, "new_udf_cloud_edge", config.newUdf);
    RequireRosParameter(
        nodeHandle,
        "sampler_edge_front_kf_num",
        config.samplerEdgeFrontKeyFrames);
    RequireRosParameter(
        nodeHandle,
        "sampler_edge_back_kf_num",
        config.samplerEdgeBackKeyFrames);
    RequireRosParameter(
        nodeHandle,
        "sampler_edge_front_min_time",
        config.samplerEdgeFrontMinTime);
    RequireRosParameter(
        nodeHandle,
        "sampler_edge_back_min_time",
        config.samplerEdgeBackMinTime);
    RequireRosParameter(nodeHandle, "sampler_pd_kp", config.samplerPdKp);
    RequireRosParameter(nodeHandle, "sampler_pd_kd", config.samplerPdKd);
    RequireRosParameter(
        nodeHandle,
        "sampler_pd_th",
        config.samplerPdThreshold);
    nodeHandle.param<std::string>(
        "raw_image_topic",
        config.rawImageTopic,
        "/camera/rgb/image_color");
    return config;
}

// 从 ORB 相机内参构造上传 Goal 的 CameraInfo；宽高单位为像素，intrinsics 为 3x3 K 矩阵。
sensor_msgs::CameraInfo MakeCameraInfo(
    const int imageWidth,
    const int imageHeight,
    const Eigen::Matrix3f &intrinsics) {
    sensor_msgs::CameraInfo cameraInfo;
    boost::array<double, 9> intrinsicsArray = {
        intrinsics(0, 0), intrinsics(0, 1), intrinsics(0, 2),
        intrinsics(1, 0), intrinsics(1, 1), intrinsics(1, 2),
        intrinsics(2, 0), intrinsics(2, 1), intrinsics(2, 2)};
    cameraInfo.width = imageWidth;
    cameraInfo.height = imageHeight;
    cameraInfo.distortion_model = "plumb_bob";
    cameraInfo.K = intrinsicsArray;
    cameraInfo.header.frame_id = "camera";
    cameraInfo.header.stamp = ros::Time::now();
    cameraInfo.header.stamp.nsec = 0;
    return cameraInfo;
}

// 将 Sophus SE3 位姿转换为 ROS Pose，供地图调试发布和 rosbag 保存。
geometry_msgs::Pose ToRosPose(const Sophus::SE3f &pose) {
    const Eigen::Matrix4f transform = pose.matrix();
    geometry_msgs::Pose rosPose;
    rosPose.position.x = transform(0, 3);
    rosPose.position.y = transform(1, 3);
    rosPose.position.z = transform(2, 3);
    Eigen::Quaternionf quaternion(transform.block<3, 3>(0, 0));
    rosPose.orientation.x = quaternion.x();
    rosPose.orientation.y = quaternion.y();
    rosPose.orientation.z = quaternion.z();
    rosPose.orientation.w = quaternion.w();
    return rosPose;
}

struct TrajectoryRecord {
    double timestamp = 0.0;
    Eigen::Vector3f position = Eigen::Vector3f::Zero();
    Eigen::Quaternionf quaternion = Eigen::Quaternionf::Identity();
    unsigned long keyFrameId = 0U;
    bool cloud = false;
};

// 提取单张地图中有效关键帧的轨迹记录，忽略空指针和 bad keyframe。
void CollectTrajectoryRecords(
    ORB_SLAM3::Map *map,
    std::vector<TrajectoryRecord> &records) {
    records.clear();
    if (map == nullptr) {
        return;
    }
    for (ORB_SLAM3::KeyFrame *keyFrame : map->GetAllKeyFrames()) {
        if (keyFrame == nullptr || keyFrame->isBad()) {
            continue;
        }
        const Sophus::SE3f worldFromCamera = keyFrame->GetPoseInverse();
        TrajectoryRecord record;
        record.timestamp = keyFrame->mTimeStamp;
        record.position = worldFromCamera.translation();
        record.quaternion = worldFromCamera.unit_quaternion();
        record.quaternion.normalize();
        record.keyFrameId = keyFrame->mnId;
        record.cloud = keyFrame->isCloud();
        records.push_back(record);
    }
}

// 汇总 Atlas 中所有地图的轨迹记录，供 sea 全局轨迹导出使用。
void CollectAtlasTrajectoryRecords(
    ORB_SLAM3::Atlas *atlas,
    std::vector<TrajectoryRecord> &records) {
    records.clear();
    if (atlas == nullptr) {
        return;
    }
    for (ORB_SLAM3::Map *map : atlas->GetAllMaps()) {
        std::vector<TrajectoryRecord> mapRecords;
        CollectTrajectoryRecords(map, mapRecords);
        records.insert(records.end(), mapRecords.begin(), mapRecords.end());
    }
}

enum class TrajectoryFilter {
    ALL,
    CLOUD_ONLY,
    EDGE_ONLY
};

// 按时间排序写出 TUM 格式轨迹；filter 选择全部、仅云端或仅边端关键帧。
std::size_t SaveTrajectoryRecords(
    std::vector<TrajectoryRecord> records,
    const std::string &path,
    const TrajectoryFilter filter) {
    std::sort(
        records.begin(),
        records.end(),
        [](const TrajectoryRecord &left, const TrajectoryRecord &right) {
            if (left.timestamp == right.timestamp) {
                return left.keyFrameId < right.keyFrameId;
            }
            return left.timestamp < right.timestamp;
        });
    std::ofstream stream(path);
    if (!stream.is_open()) {
        ROS_ERROR_STREAM("Failed to open trajectory file: " << path);
        return 0U;
    }
    std::size_t count = 0U;
    for (const TrajectoryRecord &record : records) {
        if (filter == TrajectoryFilter::CLOUD_ONLY && !record.cloud) {
            continue;
        }
        if (filter == TrajectoryFilter::EDGE_ONLY && record.cloud) {
            continue;
        }
        stream << std::fixed << std::setprecision(9)
               << record.timestamp << " "
               << record.position.x() << " "
               << record.position.y() << " "
               << record.position.z() << " "
               << record.quaternion.x() << " "
               << record.quaternion.y() << " "
               << record.quaternion.z() << " "
               << record.quaternion.w() << std::endl;
        count++;
    }
    return count;
}

enum class LostUploadStartMode {
    COMMITTED_BOUNDARY,
    LEGACY_FALLBACK
};

struct LostUploadRequest {
    std::uint64_t generation = 0U;
    std::uint64_t boundaryGeneration = 0U;
    double originalStartFrontend = 0.0;
    double endFrontend = 0.0;
    bool committedValid = false;
    double committedFrontend = 0.0;
    double imageDelay = 0.0;
    double overlapSeconds = 0.5;
    LostUploadStartMode candidateMode = LostUploadStartMode::LEGACY_FALLBACK;
};

class RuntimePhaseDiagnosticScope {
public:
    // 进入某个诊断阶段时递增活跃计数，用于 watchdog 输出 CLOUD_PARSING/JPEG_UPLOAD。
    explicit RuntimePhaseDiagnosticScope(std::atomic<int> &activeCount)
        : mActiveCount(activeCount) {
        mActiveCount.fetch_add(1, std::memory_order_acq_rel);
    }

    // 离开作用域时递减诊断计数；不参与上传事务互斥。
    ~RuntimePhaseDiagnosticScope() {
        mActiveCount.fetch_sub(1, std::memory_order_acq_rel);
    }

private:
    std::atomic<int> &mActiveCount;
};

class UnifiedGrabber {
public:
    UnifiedGrabber(
        ORB_SLAM3::System *slam,
        const RuntimeConfig &config,
        const std::string &saveDirectory);
    ~UnifiedGrabber();

    void SetCloudClient(CloudClient *client);
    void SetImagePublisher(ros::Publisher *publisher);
    void SetOrbMapPublisher(ros::Publisher *publisher);
    void SetWrapper(SVIn2ORBWrapper *wrapper);
    void SetImageDelay(double imageDelay);
    void RunLandAirRuntime();
    void RunSeaRuntime(ros::NodeHandle &nodeHandle);
    void RunBag(const std::string &bagPath);
    void RunTxt(const std::string &txtPath);
    void GrabImage(const sensor_msgs::ImageConstPtr &message);
    void TrackImage(const cv::Mat &image, double timestamp, float imageScale);
    static void LoadImages(
        const std::string &file,
        std::vector<std::string> &imageFiles,
        std::vector<double> &timestamps);

    void RawImageCb(const sensor_msgs::ImageConstPtr &message);
    void TrackingWatchdog(const MarginalizedData &data);
    bool UploadLostImages(LostUploadRequest request);
    void BeginLostRecoveryUploadEvent(std::uint64_t generation);
    bool PreparePendingLostRecoveryUpload(
        std::uint64_t generation,
        LostUploadBoundarySnapshot boundary);
    bool FinalizePendingLostRecoveryUpload(
        std::uint64_t generation,
        bool commit);
    TrackingState GetTrackingState();

    void MemoryCb(const std_msgs::Float32ConstPtr &message);
    void GrabCloudMapCb(const cloud_edge_slam::CloudMapConstPtr &message);
    void PubORBMapCb(const std_msgs::Int16ConstPtr &message);
    void SaveORBMapCb(const std_msgs::Int16ConstPtr &message);
    static cloud_edge_slam::CloudMap ORBMapToROSMap(ORB_SLAM3::Map *map);
    static void WriteCloudMapBag(
        const cloud_edge_slam::CloudMap &map,
        const std::string &savePath);
    static void WriteCloudImagesBag(
        const std::vector<ORB_SLAM3::CloudImage> &images,
        const std::string &savePath);
    static void WriteSeqBag(
        const cloud_edge_slam::Sequence &sequence,
        const std::string &savePath);
    void StopWatchdogCallbacks();
    void ShutdownUploads();

private:
    bool DispatchCloudImages(
        const std::vector<ORB_SLAM3::CloudImage> &images,
        int edgeFrontMapId,
        int edgeBackMapId,
        int jpegQuality,
        bool waitInCaller);
    std::vector<ORB_SLAM3::CloudImage> SnapshotLostImages(
        const LostUploadRequest &request,
        int &edgeFrontMapId,
        int &edgeBackMapId);

    ORB_SLAM3::System *mpSLAM = nullptr;
    RuntimeConfig mConfig;
    std::string mSaveDirectory;
    CloudClient *mpCloudClient = nullptr;
    ros::Publisher *mpImagePublisher = nullptr;
    ros::Publisher *mpOrbMapPublisher = nullptr;
    SVIn2ORBWrapper *mpWrapper = nullptr;
    std::shared_ptr<ORB_SLAM3::CloudUploadTransactionGate> mpTransactionGate;
    std::shared_ptr<UploadLifecycleState> mpUploadLifecycle;
    std::atomic<bool> mbUploadShutdownStarted{false};
    std::map<double, cv::Mat> mImageBuffer;
    std::mutex mImageBufferMutex;
    std::mutex mTrackingStateMutex;
    // 限制 watchdog 终端监视器最多每秒输出一次，避免高频前端回调刷屏。
    std::mutex mWatchdogLogMutex;
    std::chrono::steady_clock::time_point mLastWatchdogLogTime;
    std::deque<int> mInliersWindow;
    TrackingState mTrackingState = TrackingState::NORMAL;
    double mWarningStartTime = 0.0;
    double mImageDelay = 0.0;
    std::mutex mLostUploadMutex;
    std::uint64_t mLostUploadGeneration = 0U;
    double mLostUploadStartTime = 0.0;
    double mLostUploadEndTime = 0.0;
    LostRecoveryUploadState mLostUploadState = LostRecoveryUploadState::EMPTY;
    LostUploadRequest mPreparedLostUpload;
    std::atomic<int> mCloudParsingActive{0};
    std::atomic<int> mJpegUploadActive{0};
    std::vector<float> mMemorySamples;
};

// 创建统一抓取器并初始化上传事务 gate 与生命周期状态。
// slam 为外部拥有的 System，config 为启动时读取的运行配置，saveDirectory 为本轮结果目录。
UnifiedGrabber::UnifiedGrabber(
    ORB_SLAM3::System *slam,
    const RuntimeConfig &config,
    const std::string &saveDirectory)
    : mpSLAM(slam),
      mConfig(config),
      mSaveDirectory(saveDirectory),
      mpTransactionGate(
          std::make_shared<ORB_SLAM3::CloudUploadTransactionGate>()),
      mpUploadLifecycle(std::make_shared<UploadLifecycleState>()) {
}

// 析构保护：确保 detached 上传已被唤醒并排空后才释放成员依赖。
UnifiedGrabber::~UnifiedGrabber() {
    ShutdownUploads();
}

// 注入 Action client；client 生命周期必须覆盖 UnifiedGrabber 的上传线程排空阶段。
void UnifiedGrabber::SetCloudClient(CloudClient *client) {
    mpCloudClient = client;
}

// 注入 /cloud_edge_images 发布器；用于发送压缩 JPEG 图像。
void UnifiedGrabber::SetImagePublisher(ros::Publisher *publisher) {
    mpImagePublisher = publisher;
}

// 注入调试 CloudMap 发布器，供 /test_cloud_map 请求回传 Atlas 中的地图。
void UnifiedGrabber::SetOrbMapPublisher(ros::Publisher *publisher) {
    mpOrbMapPublisher = publisher;
}

// 注入仅 sea 使用的 SVIn2/OKVIS 包装器，用于注册 watchdog 和 LOST 拓扑回调。
void UnifiedGrabber::SetWrapper(SVIn2ORBWrapper *wrapper) {
    mpWrapper = wrapper;
}

// 保存 raw_image_time = frontend_time + imageDelay 的时钟偏移。
// imageDelay 单位为秒，来自 OKVIS 传感器配置；非有限值回退为 0。
void UnifiedGrabber::SetImageDelay(const double imageDelay) {
    if (std::isfinite(imageDelay)) {
        mImageDelay = imageDelay;
    } else {
        mImageDelay = 0.0;
        ROS_WARN("Non-finite imageDelay; using 0.0");
    }
    ROS_INFO_STREAM(
        "Lost upload time conversion: raw=frontend+" << mImageDelay);
}

// 注销 sea wrapper 的状态机和 LOST 回调，shutdown 的第一步，阻止产生新的 LOST 上传。
void UnifiedGrabber::StopWatchdogCallbacks() {
    if (mpWrapper == nullptr) {
        return;
    }
    mpWrapper->RegisterStateCallbacks(
        std::function<void(const MarginalizedData &)>(),
        std::function<TrackingState()>());
    mpWrapper->RegisterLostTopologyCallbacks(
        std::function<void(std::uint64_t)>(),
        std::function<bool(std::uint64_t, LostUploadBoundarySnapshot)>(),
        std::function<bool(std::uint64_t, bool)>());
}

// 按生命周期顺序关闭上传：停 watchdog→关 gate→结束 merger→唤醒等待→排空上传。
// 本函数幂等；不得在 activeUploadCount 非零时析构 Action client、Grabber 或 System。
void UnifiedGrabber::ShutdownUploads() {
    bool expected = false;
    if (!mbUploadShutdownStarted.compare_exchange_strong(expected, true)) {
        return;
    }

    StopAcceptingUploads(mpUploadLifecycle);
    ORB_SLAM3::CloudMerging *cloudMerger = nullptr;
    if (mpSLAM != nullptr) {
        cloudMerger = mpSLAM->GetCloudMerger();
    }
    if (cloudMerger != nullptr) {
        cloudMerger->RequestFinish();
    }
    if (mpCloudClient != nullptr) {
        mpCloudClient->cancelAllGoals();
    }
    WakeUploadWaiters(mpUploadLifecycle);
    DrainUploadsAndCancelLateGoals(
        mpUploadLifecycle,
        [this]() {
            if (mpCloudClient != nullptr) {
                mpCloudClient->cancelAllGoals();
            }
        });
    if (cloudMerger != nullptr) {
        while (!cloudMerger->isFinished()) {
            usleep(5000);
        }
    }
    StopWatchdogCallbacks();
}

// 将一批 CloudImage 编码并交由公共事务层上传。
// jpegQuality 为 OpenCV JPEG 质量（0-100）；waitInCaller=true 同步执行，false 启动 detached 事务。
// jpeg 发布速率固定为事务层的 100 Hz，且 sendGoal 后保留 1 秒协议延时。
bool UnifiedGrabber::DispatchCloudImages(
    const std::vector<ORB_SLAM3::CloudImage> &images,
    const int edgeFrontMapId,
    const int edgeBackMapId,
    const int jpegQuality,
    const bool waitInCaller) {
    if (images.empty() || mpSLAM == nullptr || mpCloudClient == nullptr ||
        mpImagePublisher == nullptr) {
        return false;
    }

    ActiveUpload activeUpload = TryStartUpload(mpUploadLifecycle);
    if (!activeUpload) {
        ROS_WARN("Cloud upload rejected because shutdown has started");
        return false;
    }

    RuntimePhaseDiagnosticScope jpegScope(mJpegUploadActive);
    cloud_edge_slam::Sequence sequence;
    sequence.Header.stamp = ros::Time::now();
    sequence.camera = MakeCameraInfo(
        mpSLAM->GetSetting()->newImSize().width,
        mpSLAM->GetSetting()->newImSize().height,
        mpSLAM->GetCamera()->toK_());
    sequence.edge_front_map_mnid = edgeFrontMapId;
    sequence.edge_back_map_mnid = edgeBackMapId;

    std::vector<sensor_msgs::CompressedImagePtr> messages;
    messages.reserve(images.size());
    std::vector<int> compressionParameters;
    compressionParameters.push_back(cv::IMWRITE_JPEG_QUALITY);
    compressionParameters.push_back(jpegQuality);
    for (const ORB_SLAM3::CloudImage &image : images) {
        sequence.timestamps.push_back(image.timestamp);
        sensor_msgs::CompressedImagePtr message(
            new sensor_msgs::CompressedImage());
        message->header.stamp = ros::Time(image.timestamp);
        message->header.frame_id = image.type;
        message->format = "jpeg";
        if (!cv::imencode(
                ".jpg",
                image.img,
                message->data,
                compressionParameters)) {
            ROS_ERROR_STREAM(
                "JPEG encoding failed at timestamp " << image.timestamp);
            return false;
        }
        messages.push_back(message);
    }

    cloud_edge_slam::CloudSlamGoal goal;
    goal.sequence = sequence;
    goal.total_image_count = static_cast<int>(messages.size());

    std::shared_ptr<ActiveUpload> activeUploadHolder =
        std::make_shared<ActiveUpload>(std::move(activeUpload));
    std::shared_ptr<ORB_SLAM3::CloudUploadTransactionGate> transactionGate =
        mpTransactionGate;
    std::shared_ptr<UploadLifecycleState> lifecycleState = mpUploadLifecycle;
    CloudClient *client = mpCloudClient;
    ros::Publisher *publisher = mpImagePublisher;
    ORB_SLAM3::System *slam = mpSLAM;
    const bool oldUdf = mConfig.oldUdf;
    const bool newUdf = mConfig.newUdf;
    std::shared_ptr<UploadDispatchStartHandshake> startHandshake =
        std::make_shared<UploadDispatchStartHandshake>();
    std::shared_future<bool> dispatchStarted = startHandshake->GetFuture();

    std::function<void()> dispatch = [
        activeUploadHolder,
        transactionGate,
        lifecycleState,
        startHandshake,
        client,
        publisher,
        slam,
        oldUdf,
        newUdf,
        goal,
        messages]() {
        static_cast<void>(activeUploadHolder);
        try {
            CloudUploadTransactionCoordinator coordinator(
                *transactionGate,
                lifecycleState);
            CloudUploadResult result = coordinator.DispatchCloudUpload(
                *client,
                *publisher,
                goal,
                messages,
                MakeTicketRegistrar(
                    *slam,
                    oldUdf,
                    newUdf,
                    lifecycleState),
                MakeTicketWaiter(*slam),
                *startHandshake);
            if (!result.actionSucceeded || !result.mergeSucceeded) {
                ROS_ERROR_STREAM(
                    "Cloud upload transaction failed: "
                    << result.failureReason
                    << ", tickets="
                    << result.mergeResults.size());
            } else {
                ROS_INFO_STREAM(
                    "Cloud upload transaction completed, tickets="
                    << result.mergeResults.size());
            }
        } catch (const std::exception &exception) {
            startHandshake->Report(false);
            ROS_ERROR_STREAM(
                "Cloud upload transaction threw: " << exception.what());
        } catch (...) {
            startHandshake->Report(false);
            ROS_ERROR("Cloud upload transaction threw an unknown exception");
        }
    };

    if (waitInCaller) {
        dispatch();
        return dispatchStarted.get();
    }

    std::thread uploadThread;
    try {
        uploadThread = std::thread(dispatch);
        uploadThread.detach();
    } catch (const std::system_error &exception) {
        if (uploadThread.joinable()) {
            ROS_ERROR_STREAM(
                "Failed to detach upload worker; joining it: "
                << exception.what());
            uploadThread.join();
        } else {
            ROS_ERROR_STREAM(
                "Failed to start upload worker, running inline: "
                << exception.what());
            dispatch();
        }
    }
    return dispatchStarted.get();
}

// land/air 图像跟踪入口：执行单目跟踪、取得采样后的云图片批次，并按配置上传。
// imageScale 为 System 图像缩放比例；上传成功后记录本帧为 cloud LOST 时间参考。
void UnifiedGrabber::TrackImage(
    const cv::Mat &image,
    const double timestamp,
    const float imageScale) {
    cv::Mat trackingImage = image;
    if (imageScale != 1.0F) {
        const int width = static_cast<int>(image.cols * imageScale);
        const int height = static_cast<int>(image.rows * imageScale);
        cv::resize(image, trackingImage, cv::Size(width, height));
    }

    mpSLAM->TrackMonocular(trackingImage, timestamp);
    std::vector<ORB_SLAM3::CloudImage> sampledImages;
    std::vector<ORB_SLAM3::CloudImage> unsampledImages;
    int edgeFrontMapId = 0;
    int edgeBackMapId = 0;
    mpSLAM->GetCloudProcessImages(
        sampledImages,
        unsampledImages,
        edgeFrontMapId,
        edgeBackMapId);
    if (sampledImages.empty()) {
        return;
    }

    static std::atomic<int> uploadIndex(0);
    const int currentUploadIndex = uploadIndex.fetch_add(1) + 1;
    if (mConfig.saveCloudBag) {
        const std::string sampledPath =
            (bfs::path(mSaveDirectory) /
             ("cloud_" + std::to_string(currentUploadIndex) + ".bag")).string();
        const std::string unsampledPath =
            (bfs::path(mSaveDirectory) /
             ("cloud_nosampling_" +
              std::to_string(currentUploadIndex) + ".bag")).string();
        WriteCloudImagesBag(sampledImages, sampledPath);
        WriteCloudImagesBag(unsampledImages, unsampledPath);
    }

    if (mpCloudClient != nullptr) {
        // 常规采样上传使用 JPEG 质量 100；waitCloudResult 决定调用线程是否等待整笔事务完成。
        const bool dispatched = DispatchCloudImages(
            sampledImages,
            edgeFrontMapId,
            edgeBackMapId,
            100,
            mConfig.waitCloudResult);
        if (dispatched) {
            mpSLAM->SetTrackLostTimestamp(timestamp);
        }
    }
    mpSLAM->ResetCloudProcessImages();
}

// ROS 图像订阅回调：转换为 OpenCV 图像后转发到 land/air TrackImage。
void UnifiedGrabber::GrabImage(
    const sensor_msgs::ImageConstPtr &message) {
    cv_bridge::CvImageConstPtr cvImage;
    try {
        cvImage = cv_bridge::toCvShare(message);
    } catch (const cv_bridge::Exception &exception) {
        ROS_ERROR("cv_bridge exception: %s", exception.what());
        return;
    }
    TrackImage(
        cvImage->image,
        message->header.stamp.toSec(),
        mpSLAM->GetImageScale());
}

// 解析 txt 数据集索引；file 前三行是格式头，后续每行是“时间戳 图像相对路径”。
void UnifiedGrabber::LoadImages(
    const std::string &file,
    std::vector<std::string> &imageFiles,
    std::vector<double> &timestamps) {
    std::ifstream stream(file);
    if (!stream.is_open()) {
        throw std::runtime_error("failed to open image list: " + file);
    }
    std::string line;
    for (int lineIndex = 0; lineIndex < 3; ++lineIndex) {
        std::getline(stream, line);
    }
    while (std::getline(stream, line)) {
        if (line.empty()) {
            continue;
        }
        std::stringstream lineStream(line);
        double timestamp = 0.0;
        std::string imageFile;
        lineStream >> timestamp >> imageFile;
        if (!imageFile.empty()) {
            timestamps.push_back(timestamp);
            imageFiles.push_back(imageFile);
        }
    }
}

// 回放 txt 图像序列；按相邻帧时间戳限速，最长额外等待 0.5 秒。
// mainLoopSleepMs 是额外节拍，单位毫秒。
void UnifiedGrabber::RunTxt(const std::string &txtPath) {
    const std::string dataDirectory = txtPath.substr(0, txtPath.rfind('/'));
    std::vector<std::string> imageFiles;
    std::vector<double> timestamps;
    LoadImages(txtPath, imageFiles, timestamps);
    const float imageScale = mpSLAM->GetImageScale();
    for (std::size_t index = 0U; index < imageFiles.size() && ros::ok(); ++index) {
        const std::string imagePath =
            (bfs::path(dataDirectory) / imageFiles[index]).string();
        cv::Mat image = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
        if (image.empty()) {
            throw std::runtime_error("failed to load image: " + imagePath);
        }
        const std::chrono::steady_clock::time_point start =
            std::chrono::steady_clock::now();
        TrackImage(image, timestamps[index], imageScale);
        const double trackingSeconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::steady_clock::now() - start).count();
        double framePeriod = 0.0;
        if (index + 1U < timestamps.size()) {
            framePeriod = timestamps[index + 1U] - timestamps[index];
        } else if (index > 0U) {
            framePeriod = timestamps[index] - timestamps[index - 1U];
        }
        if (trackingSeconds < framePeriod) {
            usleep(static_cast<useconds_t>(
                std::min(framePeriod - trackingSeconds, 0.5) * 1e6));
        }
        if (mConfig.mainLoopSleepMs != 0.0F) {
            usleep(static_cast<useconds_t>(mConfig.mainLoopSleepMs * 1e3F));
        }
        ros::spinOnce();
    }
}

// 回放 rosbag 中 rawImageTopic 图像；使用消息时间戳维持输入节奏。
void UnifiedGrabber::RunBag(const std::string &bagPath) {
    rosbag::Bag bag;
    bag.open(bagPath, rosbag::bagmode::Read);
    std::vector<std::string> topics;
    topics.push_back(mConfig.rawImageTopic);
    rosbag::View view(bag, rosbag::TopicQuery(topics));
    std::vector<double> timestamps;
    for (const rosbag::MessageInstance &message : view) {
        sensor_msgs::ImageConstPtr image = message.instantiate<sensor_msgs::Image>();
        if (image != nullptr) {
            timestamps.push_back(image->header.stamp.toSec());
        }
    }

    std::size_t index = 0U;
    const float imageScale = mpSLAM->GetImageScale();
    for (const rosbag::MessageInstance &message : view) {
        if (!ros::ok()) {
            break;
        }
        sensor_msgs::ImageConstPtr image = message.instantiate<sensor_msgs::Image>();
        if (image == nullptr || index >= timestamps.size()) {
            continue;
        }
        cv_bridge::CvImageConstPtr cvImage;
        try {
            cvImage = cv_bridge::toCvCopy(image);
        } catch (const cv_bridge::Exception &exception) {
            ROS_ERROR("cv_bridge exception: %s", exception.what());
            return;
        }
        const std::chrono::steady_clock::time_point start =
            std::chrono::steady_clock::now();
        TrackImage(cvImage->image, timestamps[index], imageScale);
        const double trackingSeconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::steady_clock::now() - start).count();
        double framePeriod = 0.0;
        if (index + 1U < timestamps.size()) {
            framePeriod = timestamps[index + 1U] - timestamps[index];
        } else if (index > 0U) {
            framePeriod = timestamps[index] - timestamps[index - 1U];
        }
        if (trackingSeconds < framePeriod) {
            usleep(static_cast<useconds_t>(
                std::min(framePeriod - trackingSeconds, 0.5) * 1e6));
        }
        if (mConfig.mainLoopSleepMs != 0.0F) {
            usleep(static_cast<useconds_t>(mConfig.mainLoopSleepMs * 1e3F));
        }
        index++;
        ros::spinOnce();
    }
    bag.close();
}

// 执行 land/air 路径：根据 data_type 选择 txt、bag 或实时 ROS 图像输入。
void UnifiedGrabber::RunLandAirRuntime() {
    if (!mConfig.cloudOnline) {
        return;
    }
    if (mConfig.dataType == "txt") {
        RunTxt(mConfig.dataPath);
        return;
    }
    if (mConfig.dataType == "bag") {
        RunBag(mConfig.dataPath);
        return;
    }
    if (mConfig.dataType == "ros") {
        ros::spin();
        return;
    }
    throw std::runtime_error(
        "land/air data_type must be one of: txt, bag, ros");
}

// sea 原始图像缓存回调：保存最近 15 秒 BGR 图像，用于 LOST 时间窗回溯上传。
void UnifiedGrabber::RawImageCb(
    const sensor_msgs::ImageConstPtr &message) {
    cv_bridge::CvImageConstPtr cvImage;
    try {
        cvImage = cv_bridge::toCvShare(
            message,
            sensor_msgs::image_encodings::BGR8);
    } catch (const cv_bridge::Exception &exception) {
        ROS_ERROR("cv_bridge exception: %s", exception.what());
        return;
    }

    const double timestamp = message->header.stamp.toSec();
    std::lock_guard<std::mutex> lock(mImageBufferMutex);
    mImageBuffer[timestamp] = cvImage->image.clone();
    // 缓存窗口固定为 15 秒；LOST 回溯范围超过该长度会被截断。
    while (!mImageBuffer.empty() &&
           timestamp - mImageBuffer.begin()->first > 15.0) {
        mImageBuffer.erase(mImageBuffer.begin());
    }
}

// 接收新的 LOST generation，清空上一轮候选时间窗并进入 AWAITING_RECOVERY。
void UnifiedGrabber::BeginLostRecoveryUploadEvent(
    const std::uint64_t generation) {
    std::lock_guard<std::mutex> lock(mLostUploadMutex);
    mLostUploadGeneration = generation;
    mLostUploadStartTime = 0.0;
    mLostUploadEndTime = 0.0;
    mLostUploadState = LostRecoveryUploadState::AWAITING_RECOVERY;
    mPreparedLostUpload = LostUploadRequest();
}

// 在恢复后校验 generation 与边界快照，固化本轮 LOST 上传请求并进入 DISPATCHING。
// boundary 包含拓扑提交的前端时间戳；成功时优先作为图片截取起点。
bool UnifiedGrabber::PreparePendingLostRecoveryUpload(
    const std::uint64_t generation,
    const LostUploadBoundarySnapshot boundary) {
    std::lock_guard<std::mutex> lock(mLostUploadMutex);
    if (mLostUploadGeneration != generation ||
        mLostUploadState != LostRecoveryUploadState::AVAILABLE) {
        return false;
    }

    LostUploadRequest request;
    request.generation = generation;
    request.boundaryGeneration = boundary.generation;
    request.originalStartFrontend = mLostUploadStartTime;
    request.endFrontend = mLostUploadEndTime;
    request.imageDelay = mImageDelay;
    request.committedValid = false;
    request.candidateMode = LostUploadStartMode::LEGACY_FALLBACK;
    if (boundary.generation == generation && boundary.valid) {
        request.committedValid = true;
        request.committedFrontend = boundary.frontend_timestamp;
        request.candidateMode = LostUploadStartMode::COMMITTED_BOUNDARY;
    }

    mPreparedLostUpload = request;
    mLostUploadState = LostRecoveryUploadState::DISPATCHING;
    return true;
}

// 根据拓扑提交结果决定是否启动已准备的 LOST 上传。
// commit=true 只等待 dispatch 已发起；Action 和 ticket 合并完成由 detached 事务继续等待。
bool UnifiedGrabber::FinalizePendingLostRecoveryUpload(
    const std::uint64_t generation,
    const bool commit) {
    LostUploadRequest request;
    {
        std::lock_guard<std::mutex> lock(mLostUploadMutex);
        if (mLostUploadGeneration != generation ||
            mLostUploadState != LostRecoveryUploadState::DISPATCHING) {
            return false;
        }
        if (!commit) {
            mLostUploadState = LostRecoveryUploadState::AVAILABLE;
            return true;
        }
        request = mPreparedLostUpload;
    }

    const bool dispatchStarted = UploadLostImages(request);
    {
        std::lock_guard<std::mutex> lock(mLostUploadMutex);
        if (mLostUploadGeneration != generation ||
            mLostUploadState != LostRecoveryUploadState::DISPATCHING) {
            return false;
        }
        mLostUploadState =
            FinalizedLostRecoveryUploadState(dispatchStarted);
    }
    return dispatchStarted;
}

// 线程安全读取 watchdog 当前状态，供 SVIn2 wrapper 查询是否处于 LOST。
TrackingState UnifiedGrabber::GetTrackingState() {
    std::lock_guard<std::mutex> lock(mTrackingStateMutex);
    return mTrackingState;
}

// sea 跟踪质量状态机：使用最近 5 帧 landmark 平均数驱动 NORMAL/WARNING/LOST。
// 阈值：WARNING<70，LOST<50，恢复>=90；WARNING 最长 5 秒，时间戳单位为秒。
void UnifiedGrabber::TrackingWatchdog(const MarginalizedData &data) {
    const int landmarkCount = static_cast<int>(data.num_landmarks);
    bool queueRecoveryUpload = false;
    double recoveryStartTime = 0.0;
    double recoveryEndTime = 0.0;
    TrackingState previousState = TrackingState::NORMAL;
    TrackingState nextState = TrackingState::NORMAL;
    int averageInliers = 0;
    {
        std::lock_guard<std::mutex> lock(mTrackingStateMutex);
        previousState = mTrackingState;
        mInliersWindow.push_back(landmarkCount);
        if (mInliersWindow.size() > 5U) {
            mInliersWindow.pop_front();
        }
        int sum = 0;
        for (const int inliers : mInliersWindow) {
            sum += inliers;
        }
        averageInliers = sum / static_cast<int>(mInliersWindow.size());

        // 水下跟踪质量阈值：5 帧平均 landmark 数低于 70 进入 WARNING。
        const int warningThreshold = 70;
        // WARNING 期间低于 50 立即判 LOST。
        const int lostThreshold = 50;
        // 平均 landmark 数达到 90 才允许从 WARNING/LOST 恢复 NORMAL。
        const int recoveryThreshold = 90;
        // WARNING 持续超过 5 秒，即使未低于 lostThreshold 也转为 LOST。
        const double maximumWarningDuration = 5.0;

        if (mTrackingState == TrackingState::NORMAL) {
            if (averageInliers < warningThreshold) {
                mTrackingState = TrackingState::WARNING;
                mWarningStartTime = data.timestamp;
            }
        } else if (mTrackingState == TrackingState::WARNING) {
            if (averageInliers >= recoveryThreshold) {
                mTrackingState = TrackingState::NORMAL;
            } else if (averageInliers < lostThreshold ||
                       data.timestamp - mWarningStartTime >
                           maximumWarningDuration) {
                mTrackingState = TrackingState::LOST;
            }
        } else if (mTrackingState == TrackingState::LOST) {
            if (averageInliers >= recoveryThreshold) {
                mTrackingState = TrackingState::NORMAL;
                queueRecoveryUpload = true;
                recoveryStartTime = mWarningStartTime;
                recoveryEndTime = data.timestamp;
            }
        }
        nextState = mTrackingState;
    }

    std::string runtimePhase = "IDLE";
    if (mpSLAM != nullptr) {
        const ORB_SLAM3::BackendWriteState backendState =
            mpSLAM->GetBackendWriteState();
        if (backendState == ORB_SLAM3::BackendWriteState::MERGING) {
            runtimePhase = "MERGING";
        } else if (backendState == ORB_SLAM3::BackendWriteState::REPLAYING) {
            runtimePhase = "REPLAYING";
        }
    }
    if (runtimePhase == "IDLE" &&
        mCloudParsingActive.load(std::memory_order_acquire) > 0) {
        runtimePhase = "CLOUD_PARSING";
    }
    if (runtimePhase == "IDLE" &&
        mJpegUploadActive.load(std::memory_order_acquire) > 0) {
        runtimePhase = "JPEG_UPLOAD";
    }

    const auto trackingStateToString = [](const TrackingState state) {
        if (state == TrackingState::NORMAL) {
            return "NORMAL";
        }
        if (state == TrackingState::WARNING) {
            return "WARNING";
        }
        return "LOST";
    };

    bool emitWatchdogMonitor = false;
    {
        std::lock_guard<std::mutex> lock(mWatchdogLogMutex);
        const std::chrono::steady_clock::time_point now =
            std::chrono::steady_clock::now();
        if (mLastWatchdogLogTime.time_since_epoch().count() == 0 ||
            std::chrono::duration_cast<std::chrono::seconds>(
                now - mLastWatchdogLogTime).count() >= 1) {
            mLastWatchdogLogTime = now;
            emitWatchdogMonitor = true;
        }
    }
    if (emitWatchdogMonitor) {
        std::string wallGap = "N/A";
        if (data.diagnostic_wall_gap_available) {
            wallGap = std::to_string(data.diagnostic_wall_gap_ms) + " ms";
        }
        std::string frontendGap = "N/A";
        if (data.diagnostic_frontend_gap_available) {
            frontendGap =
                std::to_string(data.diagnostic_frontend_timestamp_gap_ms) +
                " ms";
        }
        ROS_INFO_STREAM(
            "\n\033[1;37m"
            "==================== TRACKING WATCHDOG ===================="
            "\033[0m\n"
            "\033[1;36m"
            "[Time] " << std::fixed << std::setprecision(3)
            << data.timestamp << "  |  [Sequence] "
            << data.diagnostic_sequence_id << "\033[0m\n"
            "[Tracking] " << trackingStateToString(previousState)
            << " -> " << trackingStateToString(nextState)
            << "  |  [Landmarks] " << landmarkCount
            << "  |  [Average-5] " << averageInliers << "\n"
            "[Runtime] " << runtimePhase
            << "  |  [Wall gap] " << wallGap
            << "  |  [Frontend gap] " << frontendGap
            << "\n\033[1;37m"
            "============================================================"
            "\033[0m");
    }

    if (queueRecoveryUpload) {
        std::lock_guard<std::mutex> lock(mLostUploadMutex);
        if (mLostUploadGeneration != 0U &&
            mLostUploadState == LostRecoveryUploadState::AWAITING_RECOVERY) {
            mLostUploadStartTime = recoveryStartTime;
            mLostUploadEndTime = recoveryEndTime;
            mLostUploadState = LostRecoveryUploadState::AVAILABLE;
        }
    }
}

// 从 15 秒原始图像缓存中截取本轮 LOST 图片。
// request.imageDelay、overlapSeconds 的单位均为秒；边界模式优先使用提交边界并保留 0.5 秒重叠。
std::vector<ORB_SLAM3::CloudImage> UnifiedGrabber::SnapshotLostImages(
    const LostUploadRequest &request,
    int &edgeFrontMapId,
    int &edgeBackMapId) {
    std::vector<ORB_SLAM3::CloudImage> images;
    ORB_SLAM3::Map *frontMap = nullptr;
    ORB_SLAM3::Atlas *atlas = mpSLAM->GetAtlas();
    if (atlas->CountMaps() >= 2) {
        frontMap = atlas->GetLastMap();
        edgeFrontMapId = frontMap->GetId();
        edgeBackMapId = atlas->GetCurrentMap()->GetId();
    } else {
        frontMap = atlas->GetCurrentMap();
        if (frontMap == nullptr) {
            return images;
        }
        edgeFrontMapId = frontMap->GetId();
        edgeBackMapId = edgeFrontMapId + 1;
    }

    double legacyStartFrontend = request.originalStartFrontend;
    if (frontMap != nullptr) {
        double lastKeyFrameTimestamp = 0.0;
        for (ORB_SLAM3::KeyFrame *keyFrame : frontMap->GetAllKeyFrames()) {
            if (keyFrame != nullptr && !keyFrame->isBad() &&
                keyFrame->mTimeStamp > lastKeyFrameTimestamp) {
                lastKeyFrameTimestamp = keyFrame->mTimeStamp;
            }
        }
        if (lastKeyFrameTimestamp > 0.1 &&
            lastKeyFrameTimestamp < legacyStartFrontend) {
            legacyStartFrontend = lastKeyFrameTimestamp;
        }
    }

    std::lock_guard<std::mutex> lock(mImageBufferMutex);
    if (mImageBuffer.empty()) {
        return images;
    }
    const double cacheFirst = mImageBuffer.begin()->first;
    const double cacheLast = mImageBuffer.rbegin()->first;
    double startRaw = legacyStartFrontend + request.imageDelay;
    const double endRaw = request.endFrontend + request.imageDelay;
    LostUploadStartMode actualMode = LostUploadStartMode::LEGACY_FALLBACK;
    if (request.candidateMode == LostUploadStartMode::COMMITTED_BOUNDARY &&
        request.committedValid &&
        request.boundaryGeneration == request.generation &&
        std::isfinite(request.committedFrontend) &&
        std::isfinite(request.imageDelay)) {
        const double committedStartRaw =
            request.committedFrontend + request.imageDelay -
            request.overlapSeconds;
        if (std::isfinite(committedStartRaw) && committedStartRaw <= cacheLast &&
            endRaw >= committedStartRaw) {
            startRaw = std::max(committedStartRaw, cacheFirst);
            actualMode = LostUploadStartMode::COMMITTED_BOUNDARY;
        }
    }
    if (!std::isfinite(startRaw) || !std::isfinite(endRaw) ||
        endRaw < startRaw) {
        return images;
    }

    ROS_INFO_STREAM(
        "[LostUpload] generation=" << request.generation
        << ", mode=" << static_cast<int>(actualMode)
        << ", start_raw=" << std::fixed << std::setprecision(9) << startRaw
        << ", end_raw=" << endRaw
        << ", cache_first=" << cacheFirst
        << ", cache_last=" << cacheLast);

    std::map<double, cv::Mat>::const_iterator begin =
        mImageBuffer.lower_bound(startRaw - 0.05);
    const std::map<double, cv::Mat>::const_iterator end =
        mImageBuffer.upper_bound(endRaw + 0.05);
    double lastExtractedTimestamp = -1.0;
    std::map<double, cv::Mat>::const_iterator exact =
        mImageBuffer.lower_bound(startRaw - 0.005);
    if (exact != mImageBuffer.end() &&
        std::abs(exact->first - startRaw) < 0.01) {
        images.emplace_back(exact->second.clone(), exact->first, "RGB");
        lastExtractedTimestamp = exact->first;
    }
    for (std::map<double, cv::Mat>::const_iterator iterator = begin;
         iterator != mImageBuffer.end() && iterator != end;
         ++iterator) {
        if (lastExtractedTimestamp > 0.0 &&
            iterator->first - lastExtractedTimestamp <= 0.0) {
            continue;
        }
        images.emplace_back(iterator->second.clone(), iterator->first, "RGB");
        lastExtractedTimestamp = iterator->first;
    }
    return images;
}

// 组装并启动 LOST 图片上传；使用 JPEG 质量 100，异步事务完成 Action 与地图合并等待。
bool UnifiedGrabber::UploadLostImages(LostUploadRequest request) {
    RuntimePhaseDiagnosticScope jpegScope(mJpegUploadActive);
    int edgeFrontMapId = 0;
    int edgeBackMapId = 0;
    std::vector<ORB_SLAM3::CloudImage> images = SnapshotLostImages(
        request,
        edgeFrontMapId,
        edgeBackMapId);
    if (images.empty()) {
        ROS_WARN_STREAM(
            "No images found for LOST upload generation "
            << request.generation);
        return false;
    }
    ROS_INFO_STREAM(
        "Dispatching LOST upload generation " << request.generation
        << " with " << images.size() << " images");
    // LOST 恢复上传使用 JPEG 质量 100：与常规采样上传一致，优先保留特征细节。
    return DispatchCloudImages(
        images,
        edgeFrontMapId,
        edgeBackMapId,
        100,
        false);
}

// 配置 sea 的 OKVIS estimator/publisher 回调链，并将其状态、关键帧和调试数据接入 ROS。
// parameters 来自 setting_path，决定相机、IMU 和可视化参数。
void ConfigureOkvisEstimator(
    okvis::ThreadedKFVio *estimator,
    okvis::Publisher *publisher,
    okvis::VioParameters &parameters) {
    publisher->setParameters(parameters);
    estimator->setFullStateCallback(
        [publisher, &parameters](
            const okvis::Time &timestamp,
            const okvis::kinematics::Transformation &worldFromSensor,
            const Eigen::Matrix<double, 9, 1> &speedAndBiases,
            const Eigen::Matrix<double, 3, 1> &angularVelocity,
            const okvis::kinematics::Transformation &driftCorrectedWorldFromSensor) {
            publisher->publishFullStateAsCallback(
                timestamp,
                worldFromSensor,
                speedAndBiases,
                angularVelocity,
                driftCorrectedWorldFromSensor);
            if (::pSVIn2ORBWrapper == nullptr) {
                return;
            }
            std::shared_ptr<const okvis::kinematics::Transformation>
                sensorFromCamera = parameters.nCameraSystem.T_SC(0);
            if (sensorFromCamera == nullptr) {
                return;
            }
            const okvis::kinematics::Transformation worldFromCamera =
                worldFromSensor * (*sensorFromCamera);
            Eigen::Quaterniond quaternion = worldFromCamera.q();
            quaternion.normalize();
            const Eigen::Vector3d translation = worldFromCamera.r();
            const Sophus::SE3f pose(
                quaternion.cast<float>(),
                translation.cast<float>());
            ::pSVIn2ORBWrapper->CacheFrontendPose(timestamp.toSec(), pose);
        });
    estimator->setLandmarksCallback(
        [](const okvis::Time &,
           const okvis::MapPointVector &,
           const okvis::MapPointVector &) {
        });
    estimator->setStateCallback(std::bind(
        &okvis::Publisher::publishStateAsCallback,
        publisher,
        std::placeholders::_1,
        std::placeholders::_2));
    estimator->setKeyframeCallback(std::bind(
        &okvis::Publisher::publishKeyframeAsCallback,
        publisher,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3,
        std::placeholders::_4));
    if (parameters.visualization.publishDebugImages) {
        estimator->setDebugImgCallback(std::bind(
            &okvis::Publisher::publishDebugImageAsCallback,
            publisher,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3));
    }
    estimator->setRelocRelativePoseCallback(std::bind(
        &okvis::Publisher::publishRelocRelativePoseAsCallback,
        publisher,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3,
        std::placeholders::_4,
        std::placeholders::_5));
    estimator->setImuCsvFile("imu_data.csv");
    for (std::size_t cameraIndex = 0U; cameraIndex < 2U; ++cameraIndex) {
        estimator->setTracksCsvFile(
            cameraIndex,
            "slave" + std::to_string(cameraIndex + 1U) + "_tracks.csv");
    }
}

// 执行 sea 路径：注册 watchdog/LOST 拓扑回调，启动 OKVIS/SVIn2，并订阅原始图像缓存。
// rawImageTopic 必须与 OKVIS 前端时间基准一致，才能正确应用 imageDelay。
void UnifiedGrabber::RunSeaRuntime(ros::NodeHandle &nodeHandle) {
    if (mpWrapper == nullptr) {
        throw std::runtime_error("sea runtime requires SVIn2ORBWrapper");
    }
    const std::shared_ptr<UploadLifecycleState> lifecycleState =
        mpUploadLifecycle;
    mpWrapper->RegisterStateCallbacks(
        [this, lifecycleState](const MarginalizedData &data) {
            RunWithActiveUpload(
                lifecycleState,
                [this, &data]() {
                    TrackingWatchdog(data);
                });
        },
        [this, lifecycleState]() {
            ActiveUpload activeUpload = TryStartUpload(lifecycleState);
            if (!activeUpload) {
                return TrackingState::LOST;
            }
            return GetTrackingState();
        });
    mpWrapper->RegisterLostTopologyCallbacks(
        [this, lifecycleState](const std::uint64_t generation) {
            RunWithActiveUpload(
                lifecycleState,
                [this, generation]() {
                    BeginLostRecoveryUploadEvent(generation);
                });
        },
        [this, lifecycleState](
            const std::uint64_t generation,
            const LostUploadBoundarySnapshot boundary) {
            ActiveUpload activeUpload = TryStartUpload(lifecycleState);
            if (!activeUpload) {
                return false;
            }
            return PreparePendingLostRecoveryUpload(generation, boundary);
        },
        [this, lifecycleState](
            const std::uint64_t generation,
            const bool commit) {
            ActiveUpload activeUpload = TryStartUpload(lifecycleState);
            if (!activeUpload) {
                return false;
            }
            return FinalizePendingLostRecoveryUpload(generation, commit);
        });

    okvis::RosParametersReader parameterReader(mConfig.settingPath);
    okvis::VioParameters parameters;
    parameterReader.getParameters(parameters);
    SetImageDelay(parameters.sensors_information.imageDelay);
    okvis::Publisher publisher(nodeHandle);
    publisher.setCsvFile(mSaveDirectory + "/okvis_estimator_output.csv");
    publisher.setLandmarksCsvFile(
        mSaveDirectory + "/okvis_estimator_landmarks.csv");
    {
        okvis::ThreadedKFVio estimator(parameters);
        ConfigureOkvisEstimator(&estimator, &publisher, parameters);
        okvis::Subscriber subscriber(nodeHandle, &estimator, parameterReader);
        ros::Subscriber rawImageSubscriber = nodeHandle.subscribe(
            mConfig.rawImageTopic,
            200,
            &UnifiedGrabber::RawImageCb,
            this);
        static_cast<void>(subscriber);
        static_cast<void>(rawImageSubscriber);
        if (mConfig.cloudOnline) {
            ROS_INFO("Sea runtime ready for sensor streams");
            ros::MultiThreadedSpinner spinner(4);
            spinner.spin();
        }
    }
}

// 接收边端内存监控样本；仅记录，当前不参与限流或状态机决策。
void UnifiedGrabber::MemoryCb(
    const std_msgs::Float32ConstPtr &message) {
    mMemorySamples.push_back(message->data);
}

// 调试 CloudMap 订阅回调：转换地图、登记 ticket，并同步等待其合并终态。
void UnifiedGrabber::GrabCloudMapCb(
    const cloud_edge_slam::CloudMapConstPtr &message) {
    RuntimePhaseDiagnosticScope parsingScope(mCloudParsingActive);
    ORB_SLAM3::Map *cloudMap = ConvertCloudMap(
        *mpSLAM,
        message,
        mConfig.oldUdf,
        mpUploadLifecycle);
    if (cloudMap == nullptr) {
        ROS_ERROR("Debug CloudMap conversion returned null");
        return;
    }
    ORB_SLAM3::CloudMergeTicketPtr ticket =
        mpSLAM->InsertCloudMapWithTicket(cloudMap);
    ORB_SLAM3::CloudMergeResult result;
    if (!mpSLAM->WaitForCloudMergeCompletion(ticket, result)) {
        ROS_ERROR("Debug CloudMap ticket wait failed");
    }
}

// 处理地图发布请求：message->data=-1 发布当前地图，否则发布指定 map ID。
void UnifiedGrabber::PubORBMapCb(
    const std_msgs::Int16ConstPtr &message) {
    if (mpOrbMapPublisher == nullptr || mpSLAM == nullptr) {
        return;
    }
    ORB_SLAM3::Map *map = nullptr;
    if (message->data == -1) {
        map = mpSLAM->GetAtlas()->GetCurrentMap();
    } else {
        map = mpSLAM->GetAtlas()->GetSpecifyMap(message->data);
    }
    if (map != nullptr) {
        mpOrbMapPublisher->publish(ORBMapToROSMap(map));
    }
}

// 处理地图保存请求：message->data=-1 保存当前地图，否则保存指定 map ID 到结果目录。
void UnifiedGrabber::SaveORBMapCb(
    const std_msgs::Int16ConstPtr &message) {
    if (mpSLAM == nullptr) {
        return;
    }
    ORB_SLAM3::Map *map = nullptr;
    if (message->data == -1) {
        map = mpSLAM->GetAtlas()->GetCurrentMap();
    } else {
        map = mpSLAM->GetAtlas()->GetSpecifyMap(message->data);
    }
    if (map == nullptr) {
        return;
    }
    const std::string path =
        (bfs::path(mSaveDirectory) /
         ("orb_map_" + std::to_string(message->data) + ".bag")).string();
    WriteCloudMapBag(ORBMapToROSMap(map), path);
}

// 将 ORB-SLAM3 Map 序列化为 ROS CloudMap，用于调试发布和 rosbag 导出。
// 保留关键帧、地图点、观测关系和描述子索引，不转移原 Atlas 的所有权。
cloud_edge_slam::CloudMap UnifiedGrabber::ORBMapToROSMap(
    ORB_SLAM3::Map *map) {
    cloud_edge_slam::CloudMap rosMap;
    if (map == nullptr) {
        return rosMap;
    }
    const std::vector<ORB_SLAM3::KeyFrame *> keyFrames =
        map->GetAllKeyFrames();
    const std::vector<ORB_SLAM3::MapPoint *> mapPoints =
        map->GetAllMapPoints();
    rosMap.header.seq = 0U;
    rosMap.edge_front_map_mnid = map->GetId();
    rosMap.edge_back_map_mnid = map->GetId() + 1U;

    std::map<ORB_SLAM3::KeyFrame *, int> keyFrameIndexes;
    std::map<ORB_SLAM3::MapPoint *, int> mapPointIndexes;
    for (std::size_t index = 0U; index < keyFrames.size(); ++index) {
        keyFrameIndexes[keyFrames[index]] = static_cast<int>(index);
    }
    for (std::size_t index = 0U; index < mapPoints.size(); ++index) {
        mapPointIndexes[mapPoints[index]] = static_cast<int>(index);
    }

    for (ORB_SLAM3::MapPoint *mapPoint : mapPoints) {
        cloud_edge_slam::MapPoint rosMapPoint;
        if (mapPoint == nullptr) {
            rosMapPoint.ref_keyframe_id = -1;
            rosMap.map_points.push_back(rosMapPoint);
            continue;
        }
        rosMapPoint.mnId = static_cast<int>(mapPoint->mnId);
        const auto reference = keyFrameIndexes.find(
            mapPoint->GetReferenceKeyFrame());
        if (reference == keyFrameIndexes.end()) {
            rosMapPoint.ref_keyframe_id = -1;
        } else {
            rosMapPoint.ref_keyframe_id = reference->second;
        }
        const Eigen::Vector3f position = mapPoint->GetWorldPos();
        rosMapPoint.point.x = position.x();
        rosMapPoint.point.y = position.y();
        rosMapPoint.point.z = position.z();
        const std::map<ORB_SLAM3::KeyFrame *, std::tuple<int, int>> observations =
            mapPoint->GetObservations();
        for (const auto &observation : observations) {
            const auto keyFrame = keyFrameIndexes.find(observation.first);
            if (keyFrame == keyFrameIndexes.end()) {
                continue;
            }
            cloud_edge_slam::Observation rosObservation;
            rosObservation.keyframe_id = keyFrame->second;
            rosObservation.refer_keypoint_index =
                std::get<0>(observation.second);
            rosMapPoint.observations.push_back(rosObservation);
        }
        rosMapPoint.num_obs =
            static_cast<int>(rosMapPoint.observations.size());
        rosMap.map_points.push_back(rosMapPoint);
    }

    for (ORB_SLAM3::KeyFrame *keyFrame : keyFrames) {
        if (keyFrame == nullptr) {
            continue;
        }
        cloud_edge_slam::KeyFrame rosKeyFrame;
        rosKeyFrame.mnId = static_cast<int>(keyFrame->mnId);
        rosKeyFrame.mTimeStamp = keyFrame->mTimeStamp;
        rosKeyFrame.pose_cw = ToRosPose(keyFrame->GetPose());
        for (const cv::KeyPoint &keyPoint : keyFrame->mvKeys) {
            cloud_edge_slam::KeyPoint rosKeyPoint;
            rosKeyPoint.x = keyPoint.pt.x;
            rosKeyPoint.y = keyPoint.pt.y;
            rosKeyFrame.key_points.push_back(rosKeyPoint);
        }
        const std::vector<ORB_SLAM3::MapPoint *> matches =
            keyFrame->GetMapPointMatches();
        for (ORB_SLAM3::MapPoint *match : matches) {
            const auto matchIndex = mapPointIndexes.find(match);
            if (match == nullptr || matchIndex == mapPointIndexes.end()) {
                rosKeyFrame.mvp_map_points_index.push_back(-1);
            } else {
                rosKeyFrame.mvp_map_points_index.push_back(matchIndex->second);
            }
        }
        for (int row = 0; row < keyFrame->mDescriptors.rows; ++row) {
            cloud_edge_slam::Descriptor descriptor;
            for (int column = 0; column < 32; ++column) {
                if (keyFrame->mDescriptors.type() == CV_8U) {
                    descriptor.descriptor[column] =
                        static_cast<float>(
                            keyFrame->mDescriptors.at<unsigned char>(row, column));
                } else {
                    descriptor.descriptor[column] =
                        keyFrame->mDescriptors.at<float>(row, column);
                }
            }
            rosKeyFrame.descriptors.push_back(descriptor);
        }
        rosMap.key_frames.push_back(rosKeyFrame);
    }
    return rosMap;
}

// 将一张 ROS CloudMap 写入 rosbag；savePath 为目标 bag 的完整路径。
void UnifiedGrabber::WriteCloudMapBag(
    const cloud_edge_slam::CloudMap &map,
    const std::string &savePath) {
    rosbag::Bag bag;
    bag.open(savePath, rosbag::bagmode::Write);
    bag.write("/test_cloud_map", ros::Time::now(), map);
    bag.close();
}

// 将 CloudImage 批次写入 rosbag，三通道按 bgr8、单通道按 mono8 编码。
void UnifiedGrabber::WriteCloudImagesBag(
    const std::vector<ORB_SLAM3::CloudImage> &images,
    const std::string &savePath) {
    rosbag::Bag bag;
    bag.open(savePath, rosbag::bagmode::Write);
    for (const ORB_SLAM3::CloudImage &image : images) {
        std_msgs::Header header;
        header.stamp = ros::Time(image.timestamp);
        header.frame_id = image.type;
        std::string encoding;
        if (image.img.channels() == 3) {
            encoding = "bgr8";
        } else if (image.img.channels() == 1) {
            encoding = "mono8";
        } else {
            continue;
        }
        sensor_msgs::ImagePtr message = cv_bridge::CvImage(
            header,
            encoding,
            image.img).toImageMsg();
        bag.write("/test_cloud_images", ros::Time(image.timestamp), message);
    }
    bag.close();
}

// 将上传 Sequence 元数据写入 rosbag，用于复现实验的时间戳和前后地图 ID。
void UnifiedGrabber::WriteSeqBag(
    const cloud_edge_slam::Sequence &sequence,
    const std::string &savePath) {
    rosbag::Bag bag;
    bag.open(savePath, rosbag::bagmode::Write);
    bag.write("/test_seq", ros::Time::now(), sequence);
    bag.close();
}

struct OutputDirectories {
    bfs::path active;
    bfs::path finished;
};

class WrapperGlobalPointerGuard {
public:
    // 记录已发布到全局指针的 wrapper，以便析构时只清除本实例。
    void Set(SVIn2ORBWrapper *wrapper) {
        mpWrapper = wrapper;
    }

    // 防止 wrapper 已销毁而全局 pSVIn2ORBWrapper 仍被其他回调访问。
    ~WrapperGlobalPointerGuard() {
        if (::pSVIn2ORBWrapper == mpWrapper) {
            ::pSVIn2ORBWrapper = nullptr;
        }
    }

private:
    SVIn2ORBWrapper *mpWrapper = nullptr;
};

// 根据 dataPath 和当前时间创建本轮结果目录。
// active 为运行中目录，finished 为成功退出后重命名的 Full# 目录。
OutputDirectories CreateOutputDirectories(const RuntimeConfig &config) {
    bfs::path dataPath(config.dataPath);
    std::string datasetName = dataPath.parent_path().filename().string();
    if (datasetName.empty()) {
        datasetName = dataPath.filename().string();
    }
    if (datasetName.empty()) {
        datasetName = "runtime";
    }

    const std::time_t now = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());
    std::tm localTime;
    localtime_r(&now, &localTime);
    std::stringstream timestamp;
    timestamp << std::put_time(&localTime, "%m-%d-%H_%M_%S");

    OutputDirectories directories;
    directories.active = bfs::path(config.resultPath) /
        (datasetName + "#" + timestamp.str());
    directories.finished = bfs::path(config.resultPath) /
        ("Full#" + datasetName + "#" + timestamp.str());
    bfs::create_directories(directories.active);
    return directories;
}

// 导出运行结果：land/air 使用最长地图的既有 TUM 导出，sea 额外导出全部/边端/云端轨迹。
// durationSeconds 为单次运行总时长，单位秒。
void ExportRuntimeTrajectories(
    ORB_SLAM3::System &slam,
    const ORB_SLAM3::RuntimeEnvironment environment,
    const std::string &outputDirectory,
    const double durationSeconds) {
    std::ofstream pointStream(outputDirectory + "/point.txt");
    pointStream << slam.GetAtlas()->MapPointsInMap() << std::endl;
    pointStream << durationSeconds << std::endl;
    pointStream.close();

    if (ORB_SLAM3::IsLandAirEnvironment(environment)) {
        ORB_SLAM3::Map *longestMap =
            slam.GetAtlas()->GetDurationLongestMap();
        if (longestMap != nullptr) {
            ORB_SLAM3::CloudSaveKeyFrameTrajectoryTUM(
                longestMap,
                outputDirectory + "/whole_map.txt");
        }
        return;
    }

    std::vector<TrajectoryRecord> atlasRecords;
    CollectAtlasTrajectoryRecords(slam.GetAtlas(), atlasRecords);
    SaveTrajectoryRecords(
        atlasRecords,
        outputDirectory + "/whole_map.txt",
        TrajectoryFilter::ALL);
    SaveTrajectoryRecords(
        atlasRecords,
        outputDirectory + "/whole_map_no_cloud.txt",
        TrajectoryFilter::EDGE_ONLY);
    SaveTrajectoryRecords(
        atlasRecords,
        outputDirectory + "/whole_map_cloud_only.txt",
        TrajectoryFilter::CLOUD_ONLY);
    for (ORB_SLAM3::Map *map : slam.GetAtlas()->GetAllMaps()) {
        if (map == nullptr) {
            continue;
        }
        std::vector<TrajectoryRecord> mapRecords;
        CollectTrajectoryRecords(map, mapRecords);
        std::size_t cloudCount = 0U;
        for (const TrajectoryRecord &record : mapRecords) {
            if (record.cloud) {
                cloudCount++;
            }
        }
        if (cloudCount == 0U) {
            continue;
        }
        SaveTrajectoryRecords(
            mapRecords,
            outputDirectory + "/map_" + std::to_string(map->GetId()) +
                "_cloud_only.txt",
            TrajectoryFilter::CLOUD_ONLY);
    }
}

// 统一节点主流程：读取配置、构造带 RuntimeEnvironment 的 System、选择 sea 或 land/air 运行路径并收尾导出。
// argc/argv 仅传递给 ROS 初始化；运行环境只在 LoadRuntimeConfig 中解析一次。
int RunUnifiedMain(int argc, char **argv) {
    ros::init(argc, argv, "Mono-temp");
    ros::start();
    ros::NodeHandle nodeHandle("~");
    const RuntimeConfig config = LoadRuntimeConfig(nodeHandle);
    const OutputDirectories outputDirectories =
        CreateOutputDirectories(config);
    ROS_INFO_STREAM(
        "Starting unified runtime environment="
        << ORB_SLAM3::RuntimeEnvironmentToString(config.environment)
        << ", output=" << outputDirectories.active.string());

    ORB_SLAM3::System slam(
        config.vocabularyPath,
        config.settingPath,
        ORB_SLAM3::System::MONOCULAR,
        true,
        config.cloudMerge,
        config.cloudOnline,
        config.mergeAnyway,
        config.keyFrameCulling,
        config.samplerEdgeFrontKeyFrames,
        config.samplerEdgeBackKeyFrames,
        static_cast<int>(config.samplerEdgeFrontMinTime),
        static_cast<int>(config.samplerEdgeBackMinTime),
        config.samplerPdKp,
        config.samplerPdKd,
        config.samplerPdThreshold,
        config.oldUdf,
        config.newUdf,
        0,
        std::string(),
        config.environment);
    if (slam.GetCloudMerger() != nullptr) {
        slam.GetCloudMerger()->SetCloudMergeDebugOutputDir(
            outputDirectories.active.string());
    }

    std::unique_ptr<SVIn2ORBWrapper> wrapper;
    WrapperGlobalPointerGuard wrapperGlobalPointerGuard;
    if (ORB_SLAM3::IsSeaEnvironment(config.environment)) {
        wrapper.reset(new SVIn2ORBWrapper(&slam));
        ::pSVIn2ORBWrapper = wrapper.get();
        wrapperGlobalPointerGuard.Set(wrapper.get());
        wrapper->InitOkvisFullTrajectorySaver(
            outputDirectories.active.string() + "/okvis_full_traj.txt");
    }

    ros::Publisher orbMapPublisher = nodeHandle.advertise<
        cloud_edge_slam::CloudMap>("/test_cloud_map", 1);
    ros::Publisher imagePublisher = nodeHandle.advertise<
        sensor_msgs::CompressedImage>("/cloud_edge_images", 100);
    CloudClient cloudClient(config.cloudTopicName, true);
    UnifiedGrabber grabber(
        &slam,
        config,
        outputDirectories.active.string());
    grabber.SetOrbMapPublisher(&orbMapPublisher);
    grabber.SetImagePublisher(&imagePublisher);
    if (wrapper != nullptr) {
        grabber.SetWrapper(wrapper.get());
    }

    if (config.realOnline) {
        ROS_INFO("Waiting for CloudSlam Action server");
        cloudClient.waitForServer();
        grabber.SetCloudClient(&cloudClient);
    }

    ros::Subscriber memorySubscriber = nodeHandle.subscribe(
        "/cloud_edge_memory_temp",
        1,
        &UnifiedGrabber::MemoryCb,
        &grabber);
    ros::Subscriber saveMapSubscriber = nodeHandle.subscribe(
        "/test_save_orb_map",
        1,
        &UnifiedGrabber::SaveORBMapCb,
        &grabber);
    ros::Subscriber publishMapSubscriber = nodeHandle.subscribe(
        "/test_pub_orb_map",
        1,
        &UnifiedGrabber::PubORBMapCb,
        &grabber);
    ros::Subscriber cloudMapSubscriber = nodeHandle.subscribe(
        "/test_cloud_map",
        1,
        &UnifiedGrabber::GrabCloudMapCb,
        &grabber);
    ros::Subscriber landImageSubscriber;
    if (ORB_SLAM3::IsLandAirEnvironment(config.environment)) {
        landImageSubscriber = nodeHandle.subscribe(
            config.rawImageTopic,
            10,
            &UnifiedGrabber::GrabImage,
            &grabber);
    }
    ros::ServiceClient evoClient = nodeHandle.serviceClient<
        cloud_edge_slam::Evo>("/cloud_edge_evo_temp");
    static_cast<void>(memorySubscriber);
    static_cast<void>(saveMapSubscriber);
    static_cast<void>(publishMapSubscriber);
    static_cast<void>(cloudMapSubscriber);
    static_cast<void>(landImageSubscriber);
    static_cast<void>(evoClient);
    static_cast<void>(wrapperGlobalPointerGuard);

    int exitCode = 0;
    const std::chrono::steady_clock::time_point start =
        std::chrono::steady_clock::now();
    try {
        if (ORB_SLAM3::IsSeaEnvironment(config.environment)) {
            grabber.RunSeaRuntime(nodeHandle);
        } else {
            grabber.RunLandAirRuntime();
        }
    } catch (const std::exception &exception) {
        ROS_FATAL_STREAM("Runtime failed: " << exception.what());
        exitCode = 1;
    } catch (...) {
        ROS_FATAL("Runtime failed with an unknown exception");
        exitCode = 1;
    }
    const double durationSeconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - start).count();

    grabber.ShutdownUploads();
    slam.Shutdown();
    ExportRuntimeTrajectories(
        slam,
        config.environment,
        outputDirectories.active.string(),
        durationSeconds);

    if (wrapper != nullptr) {
        wrapper->CloseOkvisFullTrajectorySaver();
        wrapper->CloseTrajectorySaver();
        ::pSVIn2ORBWrapper = nullptr;
        wrapper.reset();
    }

    if (exitCode == 0) {
        try {
            bfs::rename(outputDirectories.active, outputDirectories.finished);
        } catch (const bfs::filesystem_error &exception) {
            ROS_ERROR_STREAM(
                "Failed to mark result directory complete: "
                << exception.what());
            exitCode = 1;
        }
    }
    ros::shutdown();
    return exitCode;
}

}  // namespace cloud_edge_slam_sla

// 进程入口：将初始化异常转换为 ROS FATAL 和非零退出码。
int main(int argc, char **argv) {
    try {
        return cloud_edge_slam_sla::RunUnifiedMain(argc, argv);
    } catch (const std::exception &exception) {
        ROS_FATAL_STREAM("Unified runtime initialization failed: " << exception.what());
    } catch (...) {
        ROS_FATAL("Unified runtime initialization failed with an unknown exception");
    }
    ros::shutdown();
    return 1;
}

#endif  // CLOUD_EDGE_UPLOAD_LIFECYCLE_ONLY
