#include "LandAirInertialInput.h"

#include <cmath>
#include <iterator>

namespace cloud_edge_slam_sla {

LandAirInertialInput::PushResult LandAirInertialInput::PushImu(
    const ORB_SLAM3::IMU::Point &measurement) {
    if (!std::isfinite(measurement.t)) {
        return PushResult::INVALID_TIMESTAMP;
    }

    std::lock_guard<std::mutex> lock(mMutex);
    if (mHasLastImuTimestamp && measurement.t <= mLastImuTimestamp) {
        return PushResult::NON_MONOTONIC_TIMESTAMP;
    }

    mImuMeasurements.push_back(measurement);
    mLastImuTimestamp = measurement.t;
    mHasLastImuTimestamp = true;
    mCondition.notify_one();
    return PushResult::ACCEPTED;
}

LandAirInertialInput::PushResult LandAirInertialInput::PushImage(
    const cv::Mat &image,
    const double timestamp,
    const float imageScale) {
    if (!std::isfinite(timestamp)) {
        return PushResult::INVALID_TIMESTAMP;
    }

    std::lock_guard<std::mutex> lock(mMutex);
    if (mHasLastImageTimestamp && timestamp <= mLastImageTimestamp) {
        return PushResult::NON_MONOTONIC_TIMESTAMP;
    }

    PendingImage pending;
    pending.image = image.clone();
    pending.timestamp = timestamp;
    pending.imageScale = imageScale;
    mImages.push_back(pending);
    mLastImageTimestamp = timestamp;
    mHasLastImageTimestamp = true;
    mCondition.notify_one();
    return PushResult::ACCEPTED;
}

LandAirInertialInput::PacketResult
LandAirInertialInput::TryPopReadyPacket(
    LandAirTrackingPacket &packet,
    double &droppedTimestamp) {
    std::lock_guard<std::mutex> lock(mMutex);
    return ExtractNextPacketLocked(packet, droppedTimestamp);
}

LandAirInertialInput::PacketResult
LandAirInertialInput::WaitForReadyPacket(
    LandAirTrackingPacket &packet,
    double &droppedTimestamp) {
    std::unique_lock<std::mutex> lock(mMutex);
    while (true) {
        const PacketResult result = ExtractNextPacketLocked(
            packet,
            droppedTimestamp);
        if (result != PacketResult::WAITING_FOR_IMU) {
            return result;
        }
        mCondition.wait(lock);
    }
}

void LandAirInertialInput::Stop() {
    std::lock_guard<std::mutex> lock(mMutex);
    mStopped = true;
    mCondition.notify_all();
}

LandAirInertialInput::PacketResult
LandAirInertialInput::ExtractNextPacketLocked(
    LandAirTrackingPacket &packet,
    double &droppedTimestamp) {
    packet = LandAirTrackingPacket();
    droppedTimestamp = 0.0;
    if (mStopped) {
        return PacketResult::STOPPED;
    }
    if (mImages.empty()) {
        return PacketResult::WAITING_FOR_IMU;
    }
    if (mImuMeasurements.empty()) {
        return PacketResult::WAITING_FOR_IMU;
    }

    const PendingImage &image = mImages.front();
    if (!mFirstImageSubmitted) {
        if (mImuMeasurements.front().t > image.timestamp) {
            droppedTimestamp = image.timestamp;
            mImages.pop_front();
            return PacketResult::DROPPED_FIRST_IMAGE;
        }
        if (mImuMeasurements.back().t < image.timestamp) {
            return PacketResult::WAITING_FOR_IMU;
        }

        std::deque<ORB_SLAM3::IMU::Point>::iterator leftBoundary =
            mImuMeasurements.begin();
        for (
            std::deque<ORB_SLAM3::IMU::Point>::iterator iterator =
                mImuMeasurements.begin();
            iterator != mImuMeasurements.end() &&
            iterator->t <= image.timestamp;
            ++iterator) {
            leftBoundary = iterator;
        }
        mImuMeasurements.erase(mImuMeasurements.begin(), leftBoundary);
        packet.image = image.image;
        packet.timestamp = image.timestamp;
        packet.imageScale = image.imageScale;
        mImages.pop_front();
        mFirstImageSubmitted = true;
        return PacketResult::READY;
    }

    if (mHasSubmittedRightBoundary &&
        mLastSubmittedRightBoundaryTimestamp >= image.timestamp) {
        packet.image = image.image;
        packet.timestamp = image.timestamp;
        packet.imageScale = image.imageScale;
        mImages.pop_front();
        return PacketResult::READY;
    }
    if (mImuMeasurements.back().t < image.timestamp) {
        return PacketResult::WAITING_FOR_IMU;
    }

    std::deque<ORB_SLAM3::IMU::Point>::iterator rightBoundary =
        mImuMeasurements.begin();
    while (rightBoundary != mImuMeasurements.end() &&
           rightBoundary->t < image.timestamp) {
        ++rightBoundary;
    }
    if (rightBoundary == mImuMeasurements.end()) {
        return PacketResult::WAITING_FOR_IMU;
    }

    packet.image = image.image;
    packet.timestamp = image.timestamp;
    packet.imageScale = image.imageScale;
    packet.imuMeasurements.assign(
        mImuMeasurements.begin(),
        std::next(rightBoundary));
    mLastSubmittedRightBoundaryTimestamp = rightBoundary->t;
    mHasSubmittedRightBoundary = true;
    mImuMeasurements.erase(
        mImuMeasurements.begin(),
        std::next(rightBoundary));
    mImages.pop_front();
    return PacketResult::READY;
}

}  // namespace cloud_edge_slam_sla
