#ifndef CLOUD_EDGE_SLAM_LIB_ROS_IMAGE_INPUT_H
#define CLOUD_EDGE_SLAM_LIB_ROS_IMAGE_INPUT_H

#include <opencv2/core.hpp>
#include <sensor_msgs/CompressedImage.h>

#include <string>

namespace cloud_edge_slam_sla {

enum class RosImageMessageKind {
    RAW,
    COMPRESSED,
    UNSUPPORTED
};

RosImageMessageKind ClassifyRosImageDatatype(const std::string &datatype);

bool DecodeCompressedImage(
    const sensor_msgs::CompressedImage &message,
    cv::Mat &image);

}  // namespace cloud_edge_slam_sla

#endif  // CLOUD_EDGE_SLAM_LIB_ROS_IMAGE_INPUT_H
