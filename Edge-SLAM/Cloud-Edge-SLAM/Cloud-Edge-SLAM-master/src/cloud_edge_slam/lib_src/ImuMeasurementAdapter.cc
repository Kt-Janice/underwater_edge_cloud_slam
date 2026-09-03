#include "ImuMeasurementAdapter.h"

#include <cmath>
#include <stdexcept>

namespace cloud_edge_slam_sla {

ImuMeasurementAdapter::ImuMeasurementAdapter(const double accelScale)
    : mAccelScale(accelScale) {
    if (!IsValidAccelScale(mAccelScale)) {
        throw std::invalid_argument(
            "imu_accel_scale must be finite and greater than zero");
    }
}

ORB_SLAM3::IMU::Point ImuMeasurementAdapter::Adapt(
    const sensor_msgs::Imu &message) const {
    return ORB_SLAM3::IMU::Point(
        static_cast<float>(message.linear_acceleration.x * mAccelScale),
        static_cast<float>(message.linear_acceleration.y * mAccelScale),
        static_cast<float>(message.linear_acceleration.z * mAccelScale),
        static_cast<float>(message.angular_velocity.x),
        static_cast<float>(message.angular_velocity.y),
        static_cast<float>(message.angular_velocity.z),
        message.header.stamp.toSec());
}

bool ImuMeasurementAdapter::IsValidAccelScale(const double accelScale) {
    return std::isfinite(accelScale) && accelScale > 0.0;
}

}  // namespace cloud_edge_slam_sla
