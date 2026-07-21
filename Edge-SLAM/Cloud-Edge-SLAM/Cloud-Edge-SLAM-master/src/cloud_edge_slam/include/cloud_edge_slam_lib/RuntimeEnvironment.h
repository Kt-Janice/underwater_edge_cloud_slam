#ifndef RUNTIME_ENVIRONMENT_H
#define RUNTIME_ENVIRONMENT_H

#include <string>

namespace ORB_SLAM3 {

enum class RuntimeEnvironment {
    SEA = 0,
    LAND = 1,
    AIR = 2
};

RuntimeEnvironment ParseRuntimeEnvironment(const std::string &value);
const char *RuntimeEnvironmentToString(RuntimeEnvironment environment);
bool IsSeaEnvironment(RuntimeEnvironment environment);
bool IsLandAirEnvironment(RuntimeEnvironment environment);

}  // namespace ORB_SLAM3

#endif
