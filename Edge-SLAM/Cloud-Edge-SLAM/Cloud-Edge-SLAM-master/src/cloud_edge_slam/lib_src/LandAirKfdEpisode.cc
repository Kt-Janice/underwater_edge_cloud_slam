#include "LandAirKfdEpisode.h"

#include <cmath>

namespace ORB_SLAM3 {

LandAirKfdEpisode::LandAirKfdEpisode()
    : mbActive(false), mbHasLastTimestamp(false), mLastTimestamp(0.0) {
}

LandAirKfdEpisodeAction LandAirKfdEpisode::Observe(bool isLostSamplingState, double timestamp) {
    if (!isLostSamplingState) {
        if (mbActive) {
            mbActive = false;
            mbHasLastTimestamp = false;
            mLastTimestamp = 0.0;
            return LandAirKfdEpisodeAction::RESET_AND_SKIP;
        }
        return LandAirKfdEpisodeAction::NO_ACTION;
    }

    if (!std::isfinite(timestamp)) {
        return LandAirKfdEpisodeAction::SKIP_INVALID_TIMESTAMP;
    }

    if (!mbActive) {
        mbActive = true;
        mbHasLastTimestamp = false;
        mLastTimestamp = 0.0;
        return LandAirKfdEpisodeAction::RESET_AND_STEP;
    }

    if (!mbHasLastTimestamp) {
        return LandAirKfdEpisodeAction::STEP;
    }

    if (timestamp == mLastTimestamp) {
        return LandAirKfdEpisodeAction::SKIP_DUPLICATE_TIMESTAMP;
    }

    if (timestamp < mLastTimestamp) {
        return LandAirKfdEpisodeAction::SKIP_OUT_OF_ORDER_TIMESTAMP;
    }

    return LandAirKfdEpisodeAction::STEP;
}

void LandAirKfdEpisode::CommitStepTimestamp(double timestamp) {
    if (!mbActive || !std::isfinite(timestamp)) {
        return;
    }

    mbHasLastTimestamp = true;
    mLastTimestamp = timestamp;
}

void LandAirKfdEpisode::ResetTimestampForRearm() {
    mbHasLastTimestamp = false;
    mLastTimestamp = 0.0;
}

bool LandAirKfdEpisode::IsActive() const {
    return mbActive;
}

bool LandAirKfdEpisode::HasLastTimestamp() const {
    return mbHasLastTimestamp;
}

double LandAirKfdEpisode::GetLastTimestamp() const {
    return mLastTimestamp;
}

} // namespace ORB_SLAM3
