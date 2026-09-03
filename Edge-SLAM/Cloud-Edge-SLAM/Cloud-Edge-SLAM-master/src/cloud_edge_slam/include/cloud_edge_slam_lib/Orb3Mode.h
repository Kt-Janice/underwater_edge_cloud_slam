#ifndef CLOUD_EDGE_SLAM_LIB_ORB3_MODE_H
#define CLOUD_EDGE_SLAM_LIB_ORB3_MODE_H

#include <string>

#include "System.h"

namespace cloud_edge_slam_sla {

ORB_SLAM3::System::eSensor ParseOrb3Mode(const std::string &mode);
bool Orb3ModeUsesImu(ORB_SLAM3::System::eSensor sensorMode);

}  // namespace cloud_edge_slam_sla

#endif  // CLOUD_EDGE_SLAM_LIB_ORB3_MODE_H
