#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include "KFDSample.h"
#include "LandAirKfdEpisode.h"
#include "pd.hpp"

namespace ORB_SLAM3 {
namespace {

TEST(LandAirKfdEpisodeTest, StartsOnlyWhenEnteringLostSampling) {
    LandAirKfdEpisode episode;

    EXPECT_EQ(
        episode.Observe(false, 1.0),
        LandAirKfdEpisodeAction::NO_ACTION);
    EXPECT_EQ(
        episode.Observe(true, 2.0),
        LandAirKfdEpisodeAction::RESET_AND_STEP);
    EXPECT_TRUE(episode.IsActive());
}

TEST(LandAirKfdEpisodeTest, KeepsOneEpisodeAcrossRecentlyLostAndLost) {
    LandAirKfdEpisode episode;
    EXPECT_EQ(
        episode.Observe(true, 1.0),
        LandAirKfdEpisodeAction::RESET_AND_STEP);
    episode.CommitStepTimestamp(1.0);

    EXPECT_EQ(
        episode.Observe(true, 2.0),
        LandAirKfdEpisodeAction::STEP);
    episode.CommitStepTimestamp(2.0);

    EXPECT_EQ(
        episode.Observe(false, 3.0),
        LandAirKfdEpisodeAction::RESET_AND_SKIP);
    EXPECT_FALSE(episode.IsActive());
}

TEST(LandAirKfdEpisodeTest, SkipsDuplicateAndOutOfOrderWithoutResetting) {
    LandAirKfdEpisode episode;
    EXPECT_EQ(
        episode.Observe(true, 5.0),
        LandAirKfdEpisodeAction::RESET_AND_STEP);
    episode.CommitStepTimestamp(5.0);

    EXPECT_EQ(
        episode.Observe(true, 5.0),
        LandAirKfdEpisodeAction::SKIP_DUPLICATE_TIMESTAMP);
    EXPECT_TRUE(episode.IsActive());
    EXPECT_DOUBLE_EQ(episode.GetLastTimestamp(), 5.0);

    EXPECT_EQ(
        episode.Observe(true, 4.0),
        LandAirKfdEpisodeAction::SKIP_OUT_OF_ORDER_TIMESTAMP);
    EXPECT_TRUE(episode.IsActive());
    EXPECT_DOUBLE_EQ(episode.GetLastTimestamp(), 5.0);
}

TEST(LandAirKfdEpisodeTest, InvalidBaselineRearmsWithoutEndingLostEpisode) {
    LandAirKfdEpisode episode;
    ASSERT_EQ(
        episode.Observe(true, 5.0),
        LandAirKfdEpisodeAction::RESET_AND_STEP);
    episode.CommitStepTimestamp(5.0);

    episode.ResetTimestampForRearm();
    EXPECT_TRUE(episode.IsActive());
    EXPECT_FALSE(episode.HasLastTimestamp());
    EXPECT_EQ(
        episode.Observe(true, 4.0),
        LandAirKfdEpisodeAction::STEP);
}

TEST(KFDSampleLandAirTest, RejectsInvalidBaselineAndAcceptsLegalBaseline) {
    KFDSample sampler;
    EXPECT_FALSE(sampler.StepForLandAir(cv::Mat(), 1.0));

    cv::Mat blank(96, 96, CV_8UC1, cv::Scalar::all(0));
    EXPECT_FALSE(sampler.StepForLandAir(blank, 1.5));

    cv::Mat image(128, 128, CV_8UC1, cv::Scalar::all(0));
    cv::rectangle(image, cv::Rect(20, 20, 50, 50), cv::Scalar::all(255), -1);
    EXPECT_TRUE(sampler.StepForLandAir(image, 2.0));
    EXPECT_EQ(sampler.GetLandAirStepCallCount(), 3U);
    EXPECT_EQ(sampler.GetLandAirLkCallCount(), 0U);
}

TEST(KFDSampleLandAirTest, ReArmsAfterIncompatibleImageWithoutSelectingIt) {
    KFDSample sampler;
    cv::Mat image(128, 128, CV_8UC1, cv::Scalar::all(0));
    cv::rectangle(image, cv::Rect(20, 20, 50, 50), cv::Scalar::all(255), -1);
    ASSERT_TRUE(sampler.StepForLandAir(image, 1.0));

    cv::Mat incompatible(64, 64, CV_8UC1, cv::Scalar::all(0));
    EXPECT_FALSE(sampler.StepForLandAir(incompatible, 2.0));
    EXPECT_TRUE(sampler.StepForLandAir(image, 3.0));
    EXPECT_EQ(sampler.GetLandAirStepCallCount(), 3U);
    EXPECT_EQ(sampler.GetLandAirLkCallCount(), 0U);
}

TEST(PDTest, ResetClearsOnlyDerivativeHistory) {
    PD controller(0.0F, 1.0F);
    controller.setSetpoint(0.0F);
    EXPECT_FLOAT_EQ(controller.update(1.0F, 1.0), -1.0F);
    EXPECT_FLOAT_EQ(controller.update(1.0F, 1.0), 0.0F);

    controller.Reset();
    EXPECT_FLOAT_EQ(controller.update(1.0F, 1.0), -1.0F);
}

} // namespace
} // namespace ORB_SLAM3
