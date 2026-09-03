#ifndef CLOUD_EDGE_SLAM_LIB_LAND_AIR_INERTIAL_INPUT_H
#define CLOUD_EDGE_SLAM_LIB_LAND_AIR_INERTIAL_INPUT_H

#include <condition_variable>
#include <deque>
#include <mutex>
#include <vector>

#include <opencv2/core.hpp>

#include "ImuTypes.h"

namespace cloud_edge_slam_sla {

struct LandAirTrackingPacket {
    cv::Mat image;
    double timestamp = 0.0;
    float imageScale = 1.0F;
    std::vector<ORB_SLAM3::IMU::Point> imuMeasurements;
};

class LandAirInertialInput {
public:
    enum class PushResult {
        ACCEPTED,
        INVALID_TIMESTAMP,
        NON_MONOTONIC_TIMESTAMP
    };

    enum class PacketResult {
        READY,
        WAITING_FOR_IMU,
        DROPPED_FIRST_IMAGE,
        STOPPED
    };

    PushResult PushImu(const ORB_SLAM3::IMU::Point &measurement);
    PushResult PushImage(
        const cv::Mat &image,
        double timestamp,
        float imageScale);

    PacketResult TryPopReadyPacket(
        LandAirTrackingPacket &packet,
        double &droppedTimestamp);
    PacketResult WaitForReadyPacket(
        LandAirTrackingPacket &packet,
        double &droppedTimestamp);
    void Stop();

private:
    struct PendingImage {
        cv::Mat image;
        double timestamp = 0.0;
        float imageScale = 1.0F;
    };

    PacketResult ExtractNextPacketLocked(
        LandAirTrackingPacket &packet,
        double &droppedTimestamp);

    std::mutex mMutex;
    std::condition_variable mCondition;
    std::deque<ORB_SLAM3::IMU::Point> mImuMeasurements;
    std::deque<PendingImage> mImages;
    bool mStopped = false;
    bool mHasLastImuTimestamp = false;
    bool mHasLastImageTimestamp = false;
    bool mFirstImageSubmitted = false;
    bool mHasSubmittedRightBoundary = false;
    double mLastImuTimestamp = 0.0;
    double mLastImageTimestamp = 0.0;
    double mLastSubmittedRightBoundaryTimestamp = 0.0;
};

}  // namespace cloud_edge_slam_sla

#endif  // CLOUD_EDGE_SLAM_LIB_LAND_AIR_INERTIAL_INPUT_H
