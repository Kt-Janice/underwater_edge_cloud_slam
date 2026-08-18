#include <gtest/gtest.h>

#include <opencv2/imgcodecs.hpp>
#include <sensor_msgs/CompressedImage.h>

#include "RosImageInput.h"

namespace cloud_edge_slam_sla {
namespace {

TEST(RosImageInputTest, ClassifiesSupportedDatatypes) {
    EXPECT_EQ(
        ClassifyRosImageDatatype("sensor_msgs/Image"),
        RosImageMessageKind::RAW);
    EXPECT_EQ(
        ClassifyRosImageDatatype("sensor_msgs/CompressedImage"),
        RosImageMessageKind::COMPRESSED);
}

TEST(RosImageInputTest, RejectsUnsupportedDatatype) {
    EXPECT_EQ(
        ClassifyRosImageDatatype("sensor_msgs/CameraInfo"),
        RosImageMessageKind::UNSUPPORTED);
}

TEST(RosImageInputTest, DecodesCompressedImageWithoutForcingGrayscale) {
    cv::Mat source(2, 2, CV_8UC3, cv::Scalar(10, 20, 30));
    sensor_msgs::CompressedImage message;
    ASSERT_TRUE(cv::imencode(".jpg", source, message.data));

    cv::Mat decoded;
    ASSERT_TRUE(DecodeCompressedImage(message, decoded));
    EXPECT_FALSE(decoded.empty());
    EXPECT_EQ(decoded.type(), CV_8UC3);
}

TEST(RosImageInputTest, RejectsEmptyAndInvalidCompressedPayloads) {
    sensor_msgs::CompressedImage emptyMessage;
    cv::Mat decoded;
    EXPECT_FALSE(DecodeCompressedImage(emptyMessage, decoded));

    sensor_msgs::CompressedImage invalidMessage;
    invalidMessage.data = {0U, 1U, 2U, 3U};
    EXPECT_FALSE(DecodeCompressedImage(invalidMessage, decoded));
}

}  // namespace
}  // namespace cloud_edge_slam_sla
