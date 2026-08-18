#include "RosImageInput.h"

#include <opencv2/imgcodecs.hpp>

namespace cloud_edge_slam_sla {

RosImageMessageKind ClassifyRosImageDatatype(const std::string &datatype) {
    if (datatype == "sensor_msgs/Image") {
        return RosImageMessageKind::RAW;
    }

    if (datatype == "sensor_msgs/CompressedImage") {
        return RosImageMessageKind::COMPRESSED;
    }

    return RosImageMessageKind::UNSUPPORTED;
}

bool DecodeCompressedImage(
    const sensor_msgs::CompressedImage &message,
    cv::Mat &image) {
    image.release();
    if (message.data.empty()) {
        return false;
    }

    image = cv::imdecode(message.data, cv::IMREAD_UNCHANGED);
    return !image.empty();
}

}  // namespace cloud_edge_slam_sla
