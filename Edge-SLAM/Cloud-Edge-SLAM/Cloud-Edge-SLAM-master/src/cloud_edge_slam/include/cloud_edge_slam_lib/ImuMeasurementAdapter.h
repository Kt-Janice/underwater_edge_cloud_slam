#ifndef CLOUD_EDGE_SLAM_LIB_IMU_MEASUREMENT_ADAPTER_H
#define CLOUD_EDGE_SLAM_LIB_IMU_MEASUREMENT_ADAPTER_H

#include <sensor_msgs/Imu.h>

#include "ImuTypes.h"

namespace cloud_edge_slam_sla {

class ImuMeasurementAdapter {
public:
    explicit ImuMeasurementAdapter(double accelScale);

    ORB_SLAM3::IMU::Point Adapt(const sensor_msgs::Imu &message) const;

    static bool IsValidAccelScale(double accelScale);

private:
    double mAccelScale;
};

}  // namespace cloud_edge_slam_sla

#endif  // CLOUD_EDGE_SLAM_LIB_IMU_MEASUREMENT_ADAPTER_H
