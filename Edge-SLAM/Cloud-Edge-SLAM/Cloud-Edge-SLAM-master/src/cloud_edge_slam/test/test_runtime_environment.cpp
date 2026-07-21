#include <gtest/gtest.h>

#include "RuntimeEnvironment.h"

namespace ORB_SLAM3 {
namespace {

TEST(RuntimeEnvironmentTest, ParsesSupportedValues) {
    EXPECT_EQ(ParseRuntimeEnvironment("sea"), RuntimeEnvironment::SEA);
    EXPECT_EQ(ParseRuntimeEnvironment("land"), RuntimeEnvironment::LAND);
    EXPECT_EQ(ParseRuntimeEnvironment("air"), RuntimeEnvironment::AIR);
}

TEST(RuntimeEnvironmentTest, RejectsUnsupportedValues) {
    EXPECT_THROW(ParseRuntimeEnvironment("SEA"), std::invalid_argument);
    EXPECT_THROW(ParseRuntimeEnvironment("unknown"), std::invalid_argument);
}

TEST(RuntimeEnvironmentTest, ClassifiesEnvironmentFamilies) {
    EXPECT_TRUE(IsSeaEnvironment(RuntimeEnvironment::SEA));
    EXPECT_FALSE(IsSeaEnvironment(RuntimeEnvironment::LAND));
    EXPECT_TRUE(IsLandAirEnvironment(RuntimeEnvironment::LAND));
    EXPECT_TRUE(IsLandAirEnvironment(RuntimeEnvironment::AIR));
    EXPECT_FALSE(IsLandAirEnvironment(RuntimeEnvironment::SEA));
}

}  // namespace
}  // namespace ORB_SLAM3
