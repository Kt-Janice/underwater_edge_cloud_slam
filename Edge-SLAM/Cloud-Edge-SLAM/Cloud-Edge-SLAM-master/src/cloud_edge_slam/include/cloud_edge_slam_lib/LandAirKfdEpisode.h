#ifndef LAND_AIR_KFD_EPISODE_H
#define LAND_AIR_KFD_EPISODE_H

namespace ORB_SLAM3 {

enum class LandAirKfdEpisodeAction {
    NO_ACTION,
    RESET_AND_SKIP,
    RESET_AND_STEP,
    STEP,
    SKIP_DUPLICATE_TIMESTAMP,
    SKIP_OUT_OF_ORDER_TIMESTAMP,
    SKIP_INVALID_TIMESTAMP
};

class LandAirKfdEpisode {
public:
    LandAirKfdEpisode();

    LandAirKfdEpisodeAction Observe(bool isLostSamplingState, double timestamp);
    void CommitStepTimestamp(double timestamp);
    // Keep the current LOST episode active while discarding an invalid baseline.
    void ResetTimestampForRearm();

    bool IsActive() const;
    bool HasLastTimestamp() const;
    double GetLastTimestamp() const;

private:
    bool mbActive;
    bool mbHasLastTimestamp;
    double mLastTimestamp;
};

} // namespace ORB_SLAM3

#endif
