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

#include <actionlib/client/simple_action_client.h>
#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>

#include <condition_variable>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cloud_edge_slam/CloudSlamAction.h"

namespace cloud_edge_slam_sla {

using CloudClient = actionlib::SimpleActionClient<cloud_edge_slam::CloudSlamAction>;

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
    const cloud_edge_slam::CloudMapConstPtr &rosMap) {
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
    std::string failureReason;
    std::vector<ORB_SLAM3::CloudMergeTicketPtr> tickets;
};

class CloudUploadTransactionCoordinator {
public:
    using TicketRegistrar = std::function<std::vector<ORB_SLAM3::CloudMergeTicketPtr>(
        const cloud_edge_slam::CloudSlamResultConstPtr &result)>;
    using TicketWaiter = std::function<bool(
        const ORB_SLAM3::CloudMergeTicketPtr &ticket,
        ORB_SLAM3::CloudMergeResult &result)>;

    explicit CloudUploadTransactionCoordinator(
        ORB_SLAM3::CloudUploadTransactionGate &transactionGate)
        : mTransactionGate(transactionGate) {
    }

    CloudUploadResult DispatchCloudUpload(
        CloudClient &client,
        ros::Publisher &imagePublisher,
        const cloud_edge_slam::CloudSlamGoal &goal,
        const std::vector<sensor_msgs::CompressedImagePtr> &images,
        const TicketRegistrar &ticketRegistrar,
        const TicketWaiter &ticketWaiter) {
        ORB_SLAM3::CloudUploadTransactionGate::Lease transactionLease =
            mTransactionGate.Acquire();
        std::shared_ptr<CloudUploadContext> context =
            std::make_shared<CloudUploadContext>();

        client.sendGoal(
            goal,
            [context, ticketRegistrar](
                const actionlib::SimpleClientGoalState &state,
                const cloud_edge_slam::CloudSlamResultConstPtr &result) {
                CompleteAction(context, state, result, ticketRegistrar);
            },
            CloudClient::SimpleActiveCallback(),
            CloudClient::SimpleFeedbackCallback());

        // The initial delay is retained because the current cloud protocol may
        // use it to establish the image-receive context for this Goal.
        ros::Duration(1.0).sleep();
        for (const sensor_msgs::CompressedImagePtr &image : images) {
            imagePublisher.publish(image);
            ros::spinOnce();
        }

        return WaitForTransaction(context, ticketWaiter);
    }

private:
    static void CompleteAction(
        const std::shared_ptr<CloudUploadContext> &context,
        const actionlib::SimpleClientGoalState &state,
        const cloud_edge_slam::CloudSlamResultConstPtr &result,
        const TicketRegistrar &ticketRegistrar) {
        std::vector<ORB_SLAM3::CloudMergeTicketPtr> tickets;
        bool actionSucceeded = false;
        std::string failureReason;

        if (state == actionlib::SimpleClientGoalState::SUCCEEDED && result) {
            try {
                tickets = ticketRegistrar(result);
                actionSucceeded = true;
            } catch (const std::exception &exception) {
                failureReason = exception.what();
            } catch (...) {
                failureReason = "unknown exception while registering CloudMap tickets";
            }
        } else {
            failureReason = state.toString();
        }

        {
            std::lock_guard<std::mutex> lock(context->mutex);
            context->tickets = std::move(tickets);
            context->actionSucceeded = actionSucceeded;
            context->failureReason = std::move(failureReason);
            context->actionFinished = true;
        }
        context->condition.notify_all();
    }

    static CloudUploadResult WaitForTransaction(
        const std::shared_ptr<CloudUploadContext> &context,
        const TicketWaiter &ticketWaiter) {
        CloudUploadResult uploadResult;
        std::vector<ORB_SLAM3::CloudMergeTicketPtr> tickets;

        {
            std::unique_lock<std::mutex> lock(context->mutex);
            context->condition.wait(lock, [context]() {
                return context->actionFinished;
            });
            uploadResult.actionSucceeded = context->actionSucceeded;
            uploadResult.failureReason = context->failureReason;
            tickets = context->tickets;
        }

        if (!uploadResult.actionSucceeded) {
            return uploadResult;
        }

        uploadResult.mergeSucceeded = true;
        for (const ORB_SLAM3::CloudMergeTicketPtr &ticket : tickets) {
            ORB_SLAM3::CloudMergeResult mergeResult;
            if (!ticketWaiter(ticket, mergeResult)) {
                uploadResult.mergeSucceeded = false;
                uploadResult.failureReason = "CloudMap ticket wait failed";
                continue;
            }

            uploadResult.mergeResults.push_back(mergeResult);
            if (mergeResult.outcome != ORB_SLAM3::CloudMergeOutcome::MERGED_REPLAY_COMPLETED &&
                mergeResult.outcome != ORB_SLAM3::CloudMergeOutcome::MERGED_REPLAY_SKIPPED_LOST &&
                mergeResult.outcome != ORB_SLAM3::CloudMergeOutcome::MERGED_REPLAY_SKIPPED_SHUTDOWN &&
                mergeResult.outcome != ORB_SLAM3::CloudMergeOutcome::MERGE_COMPLETED_LAND_AIR) {
                uploadResult.mergeSucceeded = false;
            }
        }

        return uploadResult;
    }

    ORB_SLAM3::CloudUploadTransactionGate &mTransactionGate;
};

CloudUploadTransactionCoordinator::TicketRegistrar MakeTicketRegistrar(
    ORB_SLAM3::System &slam) {
    return [&slam](const cloud_edge_slam::CloudSlamResultConstPtr &result) {
        if (!result) {
            throw std::runtime_error("CloudSlam Action returned no result");
        }

        cloud_edge_slam::CloudMapConstPtr rosMap(
            new cloud_edge_slam::CloudMap(result->map));
        ORB_SLAM3::Map *cloudMap = ConvertCloudMap(slam, rosMap);
        if (cloudMap == nullptr) {
            throw std::runtime_error("CloudMap conversion failed");
        }

        std::vector<ORB_SLAM3::CloudMergeTicketPtr> tickets;
        tickets.push_back(slam.InsertCloudMapWithTicket(cloudMap));
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
        runtimeEnvironmentValue = "sea";
    }

    return ORB_SLAM3::ParseRuntimeEnvironment(runtimeEnvironmentValue);
}

}  // namespace cloud_edge_slam_sla
