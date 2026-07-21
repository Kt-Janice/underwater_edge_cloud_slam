#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

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

LostRecoveryUploadState FinalizedLostRecoveryUploadState(
    const bool dispatchStarted) {
    if (dispatchStarted) {
        return LostRecoveryUploadState::CONSUMED;
    }
    return LostRecoveryUploadState::AVAILABLE;
}

class UploadDispatchStartHandshake {
public:
    UploadDispatchStartHandshake()
        : mFuture(mPromise.get_future().share()) {
    }

    std::shared_future<bool> GetFuture() const {
        return mFuture;
    }

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
    ActiveUpload() = default;

    explicit ActiveUpload(const std::shared_ptr<UploadLifecycleState> &state)
        : mState(state) {
    }

    ~ActiveUpload() {
        Release();
    }

    ActiveUpload(const ActiveUpload &) = delete;
    ActiveUpload &operator=(const ActiveUpload &) = delete;

    ActiveUpload(ActiveUpload &&other) noexcept
        : mState(std::move(other.mState)) {
    }

    ActiveUpload &operator=(ActiveUpload &&other) noexcept {
        if (this != &other) {
            Release();
            mState = std::move(other.mState);
        }
        return *this;
    }

    explicit operator bool() const {
        return mState != nullptr;
    }

private:
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

void UnregisterUploadWakeCallback(
    const std::shared_ptr<UploadLifecycleState> &state,
    const std::size_t callbackId) {
    if (state == nullptr || callbackId == 0U) {
        return;
    }

    std::lock_guard<std::mutex> lock(state->mutex);
    state->wakeCallbacks.erase(callbackId);
}

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
void DrainUploadsAndCancelLateGoals(
    const std::shared_ptr<UploadLifecycleState> &state,
    const CancelGoals &cancelGoals) {
    WaitForActiveUploads(state);
    cancelGoals();
}

std::size_t GetActiveUploadCountForTesting(
    const std::shared_ptr<UploadLifecycleState> &state) {
    if (state == nullptr) {
        return 0U;
    }

    std::lock_guard<std::mutex> lock(state->mutex);
    return state->activeUploadCount;
}

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

extern SVIn2ORBWrapper *pSVIn2ORBWrapper;

namespace cloud_edge_slam_sla {

using CloudClient = actionlib::SimpleActionClient<cloud_edge_slam::CloudSlamAction>;

namespace bfs = boost::filesystem;

const char kUdfRoot[] =
    "/home/lhf/vc-SLAM_ws/Edge-SLAM/Cloud-Edge-SLAM/"
    "Cloud-Edge-SLAM-master";

bool FileExists(const std::string &path) {
    struct stat fileStatus;
    return stat(path.c_str(), &fileStatus) == 0;
}

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

    static unsigned long nextCloudMapId = 1000;
    ORB_SLAM3::Map *cloudMap = new ORB_SLAM3::Map(nextCloudMapId++, true);
    cloudMap->edgeFrontMapMnId = rosMap->edge_front_map_mnid;
    cloudMap->edgeBackMapMnId = rosMap->edge_back_map_mnid;

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

class CloudUploadTransactionCoordinator {
public:
    using TicketRegistrar = std::function<std::vector<ORB_SLAM3::CloudMergeTicketPtr>(
        const cloud_edge_slam::CloudSlamResultConstPtr &result)>;
    using TicketWaiter = std::function<bool(
        const ORB_SLAM3::CloudMergeTicketPtr &ticket,
        ORB_SLAM3::CloudMergeResult &result)>;

    explicit CloudUploadTransactionCoordinator(
        ORB_SLAM3::CloudUploadTransactionGate &transactionGate,
        const std::shared_ptr<UploadLifecycleState> &lifecycleState)
        : mTransactionGate(transactionGate),
          mLifecycleState(lifecycleState) {
    }

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
        ros::Duration(1.0).sleep();
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
            tickets.push_back(slam.InsertCloudMapWithTicket(cloudMap));
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

        ros::Duration(3.0).sleep();
        rosbag::Bag resultBag;
        resultBag.open(bagPath, rosbag::bagmode::Read);
        std::vector<std::string> topics;
        topics.push_back("/test");
        rosbag::View view(resultBag, rosbag::TopicQuery(topics));
        for (const rosbag::MessageInstance &message : view) {
            cloud_edge_slam::CloudSlamResultConstPtr bridgedResult =
                message.instantiate<cloud_edge_slam::CloudSlamResult>();
            if (bridgedResult != nullptr) {
                registerMap(bridgedResult->map);
            }
        }
        resultBag.close();
        std::remove(bagPath.c_str());
        std::remove(pointPath.c_str());
        return tickets;
    };
}

CloudUploadTransactionCoordinator::TicketWaiter MakeTicketWaiter(
    ORB_SLAM3::System &slam) {
    return [&slam](
               const ORB_SLAM3::CloudMergeTicketPtr &ticket,
               ORB_SLAM3::CloudMergeResult &result) {
        return slam.WaitForCloudMergeCompletion(ticket, result);
    };
}

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
    ORB_SLAM3::RuntimeEnvironment environment =
        ORB_SLAM3::RuntimeEnvironment::SEA;
    std::string cloudTopicName;
    std::string vocabularyPath;
    std::string settingPath;
    std::string dataType;
    std::string dataPath;
    std::string resultPath;
    std::string rawImageTopic = "/camera/rgb/image_color";
    bool cloudMerge = true;
    bool saveCloudBag = false;
    bool realOnline = false;
    bool mergeAnyway = false;
    bool cloudOnline = true;
    bool waitCloudResult = true;
    bool keyFrameCulling = false;
    bool oldUdf = false;
    bool newUdf = false;
    float mainLoopSleepMs = 0.0F;
    int samplerEdgeFrontKeyFrames = 0;
    int samplerEdgeBackKeyFrames = 0;
    float samplerEdgeFrontMinTime = 0.0F;
    float samplerEdgeBackMinTime = 0.0F;
    float samplerPdKp = 0.0F;
    float samplerPdKd = 0.0F;
    float samplerPdThreshold = 0.0F;
};

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
    explicit RuntimePhaseDiagnosticScope(std::atomic<int> &activeCount)
        : mActiveCount(activeCount) {
        mActiveCount.fetch_add(1, std::memory_order_acq_rel);
    }

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

UnifiedGrabber::~UnifiedGrabber() {
    ShutdownUploads();
}

void UnifiedGrabber::SetCloudClient(CloudClient *client) {
    mpCloudClient = client;
}

void UnifiedGrabber::SetImagePublisher(ros::Publisher *publisher) {
    mpImagePublisher = publisher;
}

void UnifiedGrabber::SetOrbMapPublisher(ros::Publisher *publisher) {
    mpOrbMapPublisher = publisher;
}

void UnifiedGrabber::SetWrapper(SVIn2ORBWrapper *wrapper) {
    mpWrapper = wrapper;
}

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
    while (!mImageBuffer.empty() &&
           timestamp - mImageBuffer.begin()->first > 15.0) {
        mImageBuffer.erase(mImageBuffer.begin());
    }
}

void UnifiedGrabber::BeginLostRecoveryUploadEvent(
    const std::uint64_t generation) {
    std::lock_guard<std::mutex> lock(mLostUploadMutex);
    mLostUploadGeneration = generation;
    mLostUploadStartTime = 0.0;
    mLostUploadEndTime = 0.0;
    mLostUploadState = LostRecoveryUploadState::AWAITING_RECOVERY;
    mPreparedLostUpload = LostUploadRequest();
}

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

TrackingState UnifiedGrabber::GetTrackingState() {
    std::lock_guard<std::mutex> lock(mTrackingStateMutex);
    return mTrackingState;
}

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

        const int warningThreshold = 70;
        const int lostThreshold = 50;
        const int recoveryThreshold = 90;
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

    bool emitDebug = previousState != nextState || landmarkCount < 90;
    if (data.diagnostic_wall_gap_available &&
        data.diagnostic_wall_gap_ms > 100.0) {
        emitDebug = true;
    }
    if (data.diagnostic_frontend_gap_available &&
        (data.diagnostic_frontend_timestamp_gap_ms > 100.0 ||
         data.diagnostic_frontend_timestamp_gap_ms < 0.0)) {
        emitDebug = true;
    }
    if (emitDebug) {
        std::string runtimePhase = "IDLE";
        if (mpSLAM != nullptr) {
            const ORB_SLAM3::BackendWriteState backendState =
                mpSLAM->GetBackendWriteState();
            if (backendState == ORB_SLAM3::BackendWriteState::MERGING) {
                runtimePhase = "MERGING";
            } else if (
                backendState == ORB_SLAM3::BackendWriteState::REPLAYING) {
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
        ROS_INFO_STREAM(
            "[Watchdog] sequence=" << data.diagnostic_sequence_id
            << ", frontend_timestamp=" << std::fixed
            << std::setprecision(9) << data.timestamp
            << ", landmarks=" << landmarkCount
            << ", average=" << averageInliers
            << ", previous_state=" << static_cast<int>(previousState)
            << ", state=" << static_cast<int>(nextState)
            << ", phase=" << runtimePhase);
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
    return DispatchCloudImages(
        images,
        edgeFrontMapId,
        edgeBackMapId,
        90,
        false);
}

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

void UnifiedGrabber::MemoryCb(
    const std_msgs::Float32ConstPtr &message) {
    mMemorySamples.push_back(message->data);
}

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

void UnifiedGrabber::WriteCloudMapBag(
    const cloud_edge_slam::CloudMap &map,
    const std::string &savePath) {
    rosbag::Bag bag;
    bag.open(savePath, rosbag::bagmode::Write);
    bag.write("/test_cloud_map", ros::Time::now(), map);
    bag.close();
}

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
    void Set(SVIn2ORBWrapper *wrapper) {
        mpWrapper = wrapper;
    }

    ~WrapperGlobalPointerGuard() {
        if (::pSVIn2ORBWrapper == mpWrapper) {
            ::pSVIn2ORBWrapper = nullptr;
        }
    }

private:
    SVIn2ORBWrapper *mpWrapper = nullptr;
};

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
