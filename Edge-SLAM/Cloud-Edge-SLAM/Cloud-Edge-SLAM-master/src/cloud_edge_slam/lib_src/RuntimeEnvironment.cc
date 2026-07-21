#include "RuntimeEnvironment.h"

#include <stdexcept>

namespace ORB_SLAM3 {

RuntimeEnvironment ParseRuntimeEnvironment(const std::string &value) {
    if (value == "sea") {
        return RuntimeEnvironment::SEA;
    }

    if (value == "land") {
        return RuntimeEnvironment::LAND;
    }

    if (value == "air") {
        return RuntimeEnvironment::AIR;
    }

    throw std::invalid_argument("runtime_environment must be one of: sea, land, air");
}

const char *RuntimeEnvironmentToString(const RuntimeEnvironment environment) {
    if (environment == RuntimeEnvironment::SEA) {
        return "sea";
    }

    if (environment == RuntimeEnvironment::LAND) {
        return "land";
    }

    return "air";
}

bool IsSeaEnvironment(const RuntimeEnvironment environment) {
    return environment == RuntimeEnvironment::SEA;
}

bool IsLandAirEnvironment(const RuntimeEnvironment environment) {
    return environment == RuntimeEnvironment::LAND ||
           environment == RuntimeEnvironment::AIR;
}

}  // namespace ORB_SLAM3
