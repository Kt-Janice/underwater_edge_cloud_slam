#include "Orb3Mode.h"

#include <stdexcept>

namespace cloud_edge_slam_sla {

ORB_SLAM3::System::eSensor ParseOrb3Mode(const std::string &mode) {
    if (mode == "mono") {
        return ORB_SLAM3::System::MONOCULAR;
    }

    if (mode == "mono_imu") {
        return ORB_SLAM3::System::IMU_MONOCULAR;
    }

    throw std::invalid_argument(
        "orb3_mode must be one of: mono, mono_imu");
}

bool Orb3ModeUsesImu(const ORB_SLAM3::System::eSensor sensorMode) {
    return sensorMode == ORB_SLAM3::System::IMU_MONOCULAR;
}

}  // namespace cloud_edge_slam_sla
