#include "CloudMergeTicket.h"
#include "CloudUploadTransactionGate.h"
#include "RuntimeEnvironment.h"

#include <actionlib/client/simple_action_client.h>
#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "cloud_edge_slam/CloudSlamAction.h"

namespace cloud_edge_slam_sla {

using CloudClient = actionlib::SimpleActionClient<cloud_edge_slam::CloudSlamAction>;

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

ORB_SLAM3::RuntimeEnvironment ParseRuntimeEnvironmentParameter(
    ros::NodeHandle &nodeHandle) {
    std::string runtimeEnvironmentValue;
    if (!nodeHandle.getParam("runtime_environment", runtimeEnvironmentValue)) {
        runtimeEnvironmentValue = "sea";
    }

    return ORB_SLAM3::ParseRuntimeEnvironment(runtimeEnvironmentValue);
}

}  // namespace cloud_edge_slam_sla
