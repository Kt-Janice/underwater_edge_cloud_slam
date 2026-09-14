#ifndef METRIC_EXP_LOGGER_H
#define METRIC_EXP_LOGGER_H

#include <fstream>
#include <limits>
#include <string>

namespace ORB_SLAM3 {

struct MetricExpRecord {
    MetricExpRecord();

    double timestamp;
    unsigned long frameId;
    std::string trackingState;
    long referenceKeyFrameId;
    std::string poseConvention;
    bool poseValid;
    double tx;
    double ty;
    double tz;
    double qx;
    double qy;
    double qz;
    double qw;
    long nMatched;
    long nInlier;
    double inlierRatio;
    double reprojRmsePx;
    double reprojMedianPx;
    double reprojP90Px;
    long parallaxCount;
    double parallaxMedianDeg;
    double parallaxP25Deg;
    long occupiedGridCount;
    double gridOccupancy;
    bool hessianValid;
    long hessianEdgeCount;
    long hessianRank;
    bool hessianRankDeficient;
    bool hessianNegativeEigenvalue;
    double hessianRawMinEig;
    double hessianEig1;
    double hessianEig2;
    double hessianEig3;
    double hessianEig4;
    double hessianEig5;
    double hessianEig6;
    double hessianTrace;
    double hessianMinTraceRatio;
    double hessianMinMaxRatio;
    double hessianLog10Condition;
    bool transObsValid;
    bool transObsNegativeEigenvalue;
    double transObsRawMinEig;
    double transObsEigMin;
    double transObsEigMid;
    double transObsEigMax;
    double transObsTrace;
    double transObsMinTraceRatio;
    double transObsMinMaxRatio;
    double transObsLog10Condition;
    double transObsH00;
    double transObsH01;
    double transObsH02;
    double transObsH11;
    double transObsH12;
    double transObsH22;
};

class MetricExpLogger {
public:
    MetricExpLogger(bool enabled, const std::string &outputPath,
                    bool metricExp2Enabled = false,
                    bool metricExpDirectionEnabled = false);
    ~MetricExpLogger();

    bool IsEnabled() const;
    void Write(const MetricExpRecord &record);

private:
    bool EnsureParentDirectory(const std::string &outputPath) const;
    void WriteHeader();

    bool mEnabled;
    bool mMetricExp2Enabled;
    bool mMetricExpDirectionEnabled;
    std::ofstream mOutput;
};

class MetricExpFrameRecord {
public:
    MetricExpFrameRecord();

    void Begin(double timestamp, unsigned long frameId);
    void Finalize(const MetricExpRecord &snapshot);
    bool Flush(MetricExpLogger &logger);

    bool IsActive() const;
    bool IsFinalized() const;
    bool IsFlushed() const;
    MetricExpRecord &MutableRecord();
    const MetricExpRecord &GetRecord() const;

private:
    MetricExpRecord mRecord;
    bool mActive;
    bool mFinalized;
    bool mFlushed;
};

}  // namespace ORB_SLAM3

#endif  // METRIC_EXP_LOGGER_H
