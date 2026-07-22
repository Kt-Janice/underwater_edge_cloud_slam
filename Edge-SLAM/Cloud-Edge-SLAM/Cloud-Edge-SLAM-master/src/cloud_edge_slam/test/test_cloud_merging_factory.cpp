#include "CloudMergingFactory.h"

#include <gtest/gtest.h>

namespace ORB_SLAM3 {
namespace {

TEST(CloudMergingFactoryTest, SlaBuildUsesLandAsLegacyDefaultEnvironment) {
    EXPECT_EQ(
        RuntimeEnvironment::LAND,
        GetFactoryDefaultRuntimeEnvironment());
}

}  // namespace
}  // namespace ORB_SLAM3
