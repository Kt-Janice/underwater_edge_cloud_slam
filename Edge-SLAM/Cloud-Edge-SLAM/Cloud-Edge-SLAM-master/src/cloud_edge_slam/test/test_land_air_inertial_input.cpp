#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

#include <ros/time.h>
#include <sensor_msgs/Imu.h>

#include "ImuMeasurementAdapter.h"
#include "LandAirInertialInput.h"
#include "Orb3Mode.h"

namespace cloud_edge_slam_sla {
namespace {

TEST(Orb3ModeTest, MapsSupportedModesToTheirOrbSlam3Sensors) {
    EXPECT_EQ(
        ParseOrb3Mode("mono"),
        ORB_SLAM3::System::MONOCULAR);
    EXPECT_EQ(
        ParseOrb3Mode("mono_imu"),
        ORB_SLAM3::System::IMU_MONOCULAR);
}

TEST(Orb3ModeTest, RejectsUnsupportedMode) {
    EXPECT_THROW(ParseOrb3Mode("stereo"), std::invalid_argument);
    EXPECT_THROW(ParseOrb3Mode(""), std::invalid_argument);
}

TEST(Orb3ModeTest, IdentifiesTheOnlyInertialMode) {
    EXPECT_FALSE(Orb3ModeUsesImu(ORB_SLAM3::System::MONOCULAR));
    EXPECT_TRUE(Orb3ModeUsesImu(ORB_SLAM3::System::IMU_MONOCULAR));
}

sensor_msgs::Imu MakeImuMessage(
    const double timestamp,
    const double ax,
    const double ay,
    const double az,
    const double gx,
    const double gy,
    const double gz) {
    sensor_msgs::Imu message;
    message.header.stamp.fromSec(timestamp);
    message.linear_acceleration.x = ax;
    message.linear_acceleration.y = ay;
    message.linear_acceleration.z = az;
    message.angular_velocity.x = gx;
    message.angular_velocity.y = gy;
    message.angular_velocity.z = gz;
    return message;
}

TEST(ImuMeasurementAdapterTest, ScalesOnlyAcceleration) {
    const ImuMeasurementAdapter adapter(9.80665);
    const sensor_msgs::Imu message = MakeImuMessage(
        12.5,
        -0.99,
        -0.02,
        -0.02,
        0.1,
        -0.2,
        0.3);

    const ORB_SLAM3::IMU::Point point = adapter.Adapt(message);

    EXPECT_FLOAT_EQ(point.a.x(), static_cast<float>(-0.99 * 9.80665));
    EXPECT_FLOAT_EQ(point.a.y(), static_cast<float>(-0.02 * 9.80665));
    EXPECT_FLOAT_EQ(point.a.z(), static_cast<float>(-0.02 * 9.80665));
    EXPECT_FLOAT_EQ(point.w.x(), 0.1F);
    EXPECT_FLOAT_EQ(point.w.y(), -0.2F);
    EXPECT_FLOAT_EQ(point.w.z(), 0.3F);
    EXPECT_DOUBLE_EQ(point.t, 12.5);
}

TEST(ImuMeasurementAdapterTest, UnitScalePreservesAcceleration) {
    const ImuMeasurementAdapter adapter(1.0);
    const ORB_SLAM3::IMU::Point point = adapter.Adapt(MakeImuMessage(
        4.0,
        1.25,
        -2.5,
        3.75,
        0.0,
        0.0,
        0.0));

    EXPECT_FLOAT_EQ(point.a.x(), 1.25F);
    EXPECT_FLOAT_EQ(point.a.y(), -2.5F);
    EXPECT_FLOAT_EQ(point.a.z(), 3.75F);
}

TEST(ImuMeasurementAdapterTest, RejectsNonPositiveAndNonFiniteScales) {
    EXPECT_FALSE(ImuMeasurementAdapter::IsValidAccelScale(0.0));
    EXPECT_FALSE(ImuMeasurementAdapter::IsValidAccelScale(-1.0));
    EXPECT_FALSE(ImuMeasurementAdapter::IsValidAccelScale(
        std::numeric_limits<double>::quiet_NaN()));
    EXPECT_FALSE(ImuMeasurementAdapter::IsValidAccelScale(
        std::numeric_limits<double>::infinity()));
    EXPECT_TRUE(ImuMeasurementAdapter::IsValidAccelScale(1.0));
}

TEST(LandAirInertialInputTest, RejectsNonMonotonicImuAndImageTimestamps) {
    LandAirInertialInput input;
    const ORB_SLAM3::IMU::Point imu(0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0);

    EXPECT_EQ(input.PushImu(imu), LandAirInertialInput::PushResult::ACCEPTED);
    EXPECT_EQ(
        input.PushImu(imu),
        LandAirInertialInput::PushResult::NON_MONOTONIC_TIMESTAMP);
    EXPECT_EQ(
        input.PushImage(cv::Mat::ones(1, 1, CV_8UC1), 2.0, 1.0F),
        LandAirInertialInput::PushResult::ACCEPTED);
    EXPECT_EQ(
        input.PushImage(cv::Mat::ones(1, 1, CV_8UC1), 2.0, 1.0F),
        LandAirInertialInput::PushResult::NON_MONOTONIC_TIMESTAMP);
}

TEST(LandAirInertialInputTest, FirstImageRequiresBothImuBoundaries) {
    LandAirInertialInput input;
    LandAirTrackingPacket packet;
    double droppedTimestamp = 0.0;

    ASSERT_EQ(
        input.PushImu(ORB_SLAM3::IMU::Point(
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0)),
        LandAirInertialInput::PushResult::ACCEPTED);
    ASSERT_EQ(
        input.PushImage(cv::Mat::ones(1, 1, CV_8UC1), 1.5, 1.0F),
        LandAirInertialInput::PushResult::ACCEPTED);
    EXPECT_EQ(
        input.TryPopReadyPacket(packet, droppedTimestamp),
        LandAirInertialInput::PacketResult::WAITING_FOR_IMU);

    ASSERT_EQ(
        input.PushImu(ORB_SLAM3::IMU::Point(
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 2.0)),
        LandAirInertialInput::PushResult::ACCEPTED);
    ASSERT_EQ(
        input.TryPopReadyPacket(packet, droppedTimestamp),
        LandAirInertialInput::PacketResult::READY);
    EXPECT_DOUBLE_EQ(packet.timestamp, 1.5);
    EXPECT_TRUE(packet.imuMeasurements.empty());
}

TEST(LandAirInertialInputTest, DropsFirstImageWhenAllImuAreLater) {
    LandAirInertialInput input;
    LandAirTrackingPacket packet;
    double droppedTimestamp = 0.0;

    ASSERT_EQ(
        input.PushImu(ORB_SLAM3::IMU::Point(
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 2.0)),
        LandAirInertialInput::PushResult::ACCEPTED);
    ASSERT_EQ(
        input.PushImage(cv::Mat::ones(1, 1, CV_8UC1), 1.5, 1.0F),
        LandAirInertialInput::PushResult::ACCEPTED);

    EXPECT_EQ(
        input.TryPopReadyPacket(packet, droppedTimestamp),
        LandAirInertialInput::PacketResult::DROPPED_FIRST_IMAGE);
    EXPECT_DOUBLE_EQ(droppedTimestamp, 1.5);
}

TEST(LandAirInertialInputTest, ExactFirstImageImuTimestampIsValidBoundary) {
    LandAirInertialInput input;
    LandAirTrackingPacket packet;
    double droppedTimestamp = 0.0;

    ASSERT_EQ(
        input.PushImu(ORB_SLAM3::IMU::Point(
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.5)),
        LandAirInertialInput::PushResult::ACCEPTED);
    ASSERT_EQ(
        input.PushImage(cv::Mat::ones(1, 1, CV_8UC1), 1.5, 1.0F),
        LandAirInertialInput::PushResult::ACCEPTED);

    ASSERT_EQ(
        input.TryPopReadyPacket(packet, droppedTimestamp),
        LandAirInertialInput::PacketResult::READY);
    EXPECT_TRUE(packet.imuMeasurements.empty());
}

TEST(LandAirInertialInputTest, SubmitsRightBoundaryOnlyOnce) {
    LandAirInertialInput input;
    LandAirTrackingPacket packet;
    double droppedTimestamp = 0.0;

    ASSERT_EQ(
        input.PushImu(ORB_SLAM3::IMU::Point(
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0)),
        LandAirInertialInput::PushResult::ACCEPTED);
    ASSERT_EQ(
        input.PushImu(ORB_SLAM3::IMU::Point(
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 2.0)),
        LandAirInertialInput::PushResult::ACCEPTED);
    ASSERT_EQ(
        input.PushImage(cv::Mat::ones(1, 1, CV_8UC1), 1.5, 1.0F),
        LandAirInertialInput::PushResult::ACCEPTED);
    ASSERT_EQ(
        input.TryPopReadyPacket(packet, droppedTimestamp),
        LandAirInertialInput::PacketResult::READY);

    ASSERT_EQ(
        input.PushImu(ORB_SLAM3::IMU::Point(
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 3.0)),
        LandAirInertialInput::PushResult::ACCEPTED);
    ASSERT_EQ(
        input.PushImage(cv::Mat::ones(1, 1, CV_8UC1), 2.5, 1.0F),
        LandAirInertialInput::PushResult::ACCEPTED);
    ASSERT_EQ(
        input.TryPopReadyPacket(packet, droppedTimestamp),
        LandAirInertialInput::PacketResult::READY);
    ASSERT_EQ(packet.imuMeasurements.size(), 3U);
    EXPECT_DOUBLE_EQ(packet.imuMeasurements.front().t, 1.0);
    EXPECT_DOUBLE_EQ(packet.imuMeasurements.back().t, 3.0);

    ASSERT_EQ(
        input.PushImu(ORB_SLAM3::IMU::Point(
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 4.0)),
        LandAirInertialInput::PushResult::ACCEPTED);
    ASSERT_EQ(
        input.PushImage(cv::Mat::ones(1, 1, CV_8UC1), 3.5, 1.0F),
        LandAirInertialInput::PushResult::ACCEPTED);
    ASSERT_EQ(
        input.TryPopReadyPacket(packet, droppedTimestamp),
        LandAirInertialInput::PacketResult::READY);
    ASSERT_EQ(packet.imuMeasurements.size(), 1U);
    EXPECT_DOUBLE_EQ(packet.imuMeasurements.front().t, 4.0);
}

}  // namespace
}  // namespace cloud_edge_slam_sla
