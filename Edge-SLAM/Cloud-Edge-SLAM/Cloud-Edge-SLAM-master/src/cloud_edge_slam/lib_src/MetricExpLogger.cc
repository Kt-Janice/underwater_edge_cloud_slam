#include "MetricExpLogger.h"

#include <cerrno>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/types.h>

namespace ORB_SLAM3 {
namespace {

double QuietNaN() {
    return std::numeric_limits<double>::quiet_NaN();
}

}  // namespace

MetricExpRecord::MetricExpRecord()
    : timestamp(QuietNaN()),
      frameId(0),
      trackingState("SYSTEM_NOT_READY"),
      referenceKeyFrameId(-1),
      poseConvention("Tcw"),
      poseValid(false),
      tx(QuietNaN()),
      ty(QuietNaN()),
      tz(QuietNaN()),
      qx(QuietNaN()),
      qy(QuietNaN()),
      qz(QuietNaN()),
      qw(QuietNaN()),
      nMatched(0),
      nInlier(0),
      inlierRatio(QuietNaN()),
      reprojRmsePx(QuietNaN()),
      reprojMedianPx(QuietNaN()),
      reprojP90Px(QuietNaN()),
      parallaxCount(0),
      parallaxMedianDeg(QuietNaN()),
      parallaxP25Deg(QuietNaN()),
      occupiedGridCount(0),
      gridOccupancy(QuietNaN()),
      hessianValid(false),
      hessianEdgeCount(0),
      hessianRank(0),
      hessianRankDeficient(true),
      hessianNegativeEigenvalue(false),
      hessianRawMinEig(QuietNaN()),
      hessianEig1(QuietNaN()),
      hessianEig2(QuietNaN()),
      hessianEig3(QuietNaN()),
      hessianEig4(QuietNaN()),
      hessianEig5(QuietNaN()),
      hessianEig6(QuietNaN()),
      hessianTrace(QuietNaN()),
      hessianMinTraceRatio(QuietNaN()),
      hessianMinMaxRatio(QuietNaN()),
      hessianLog10Condition(QuietNaN()),
      transObsValid(false),
      transObsNegativeEigenvalue(false),
      transObsRawMinEig(QuietNaN()),
      transObsEigMin(QuietNaN()),
      transObsEigMid(QuietNaN()),
      transObsEigMax(QuietNaN()),
      transObsTrace(QuietNaN()),
      transObsMinTraceRatio(QuietNaN()),
      transObsMinMaxRatio(QuietNaN()),
      transObsLog10Condition(QuietNaN()),
      transObsH00(QuietNaN()),
      transObsH01(QuietNaN()),
      transObsH02(QuietNaN()),
      transObsH11(QuietNaN()),
      transObsH12(QuietNaN()),
      transObsH22(QuietNaN()) {
}

MetricExpLogger::MetricExpLogger(bool enabled, const std::string &outputPath,
                                 bool metricExp2Enabled,
                                 bool metricExpDirectionEnabled)
    : mEnabled(false),
      mMetricExp2Enabled(metricExp2Enabled),
      mMetricExpDirectionEnabled(metricExp2Enabled && metricExpDirectionEnabled) {
    if (!enabled || outputPath.empty()) {
        return;
    }
    if (!EnsureParentDirectory(outputPath)) {
        return;
    }
    mOutput.open(outputPath.c_str(), std::ios::out | std::ios::trunc);
    if (!mOutput.is_open()) {
        return;
    }
    mEnabled = true;
    WriteHeader();
}

MetricExpLogger::~MetricExpLogger() {
    if (mOutput.is_open()) {
        mOutput.close();
    }
}

bool MetricExpLogger::IsEnabled() const {
    return mEnabled;
}

void MetricExpLogger::Write(const MetricExpRecord &record) {
    if (!mEnabled) {
        return;
    }
    int poseValidValue = 0;
    if (record.poseValid) {
        poseValidValue = 1;
    }
    int hessianValidValue = 0;
    int hessianRankDeficientValue = 0;
    int hessianNegativeEigenvalueValue = 0;
    int transObsValidValue = 0;
    int transObsNegativeEigenvalueValue = 0;
    if (record.hessianValid) {
        hessianValidValue = 1;
    }
    if (record.hessianRankDeficient) {
        hessianRankDeficientValue = 1;
    }
    if (record.hessianNegativeEigenvalue) {
        hessianNegativeEigenvalueValue = 1;
    }
    if (record.transObsValid) {
        transObsValidValue = 1;
    }
    if (record.transObsNegativeEigenvalue) {
        transObsNegativeEigenvalueValue = 1;
    }
    mOutput << std::fixed << std::setprecision(9)
            << record.timestamp << ','
            << record.frameId << ','
            << record.trackingState << ','
            << record.referenceKeyFrameId << ','
            << record.poseConvention << ','
            << poseValidValue << ','
            << record.tx << ',' << record.ty << ',' << record.tz << ','
            << record.qx << ',' << record.qy << ',' << record.qz << ',' << record.qw << ','
            << record.nMatched << ',' << record.nInlier << ',' << record.inlierRatio << ','
            << record.reprojRmsePx << ',' << record.reprojMedianPx << ',' << record.reprojP90Px << ','
            << record.parallaxCount << ',' << record.parallaxMedianDeg << ',' << record.parallaxP25Deg << ','
            << record.occupiedGridCount << ',' << record.gridOccupancy;
    if (mMetricExp2Enabled) {
        mOutput << ',' << hessianValidValue
                << ',' << record.hessianEdgeCount
                << ',' << record.hessianRank
                << ',' << hessianRankDeficientValue
                << ',' << hessianNegativeEigenvalueValue
                << ',' << record.hessianRawMinEig
                << ',' << record.hessianEig1
                << ',' << record.hessianEig2
                << ',' << record.hessianEig3
                << ',' << record.hessianEig4
                << ',' << record.hessianEig5
                << ',' << record.hessianEig6
                << ',' << record.hessianTrace
                << ',' << record.hessianMinTraceRatio
                << ',' << record.hessianMinMaxRatio
                << ',' << record.hessianLog10Condition
                << ',' << transObsValidValue
                << ',' << transObsNegativeEigenvalueValue
                << ',' << record.transObsRawMinEig
                << ',' << record.transObsEigMin
                << ',' << record.transObsEigMid
                << ',' << record.transObsEigMax
                << ',' << record.transObsTrace
                << ',' << record.transObsMinTraceRatio
                << ',' << record.transObsMinMaxRatio
                << ',' << record.transObsLog10Condition;
        if (mMetricExpDirectionEnabled) {
            mOutput << ',' << record.transObsH00
                    << ',' << record.transObsH01
                    << ',' << record.transObsH02
                    << ',' << record.transObsH11
                    << ',' << record.transObsH12
                    << ',' << record.transObsH22;
        }
    }
    mOutput << '\n';
    mOutput.flush();
}

bool MetricExpLogger::EnsureParentDirectory(const std::string &outputPath) const {
    const std::string::size_type slash = outputPath.find_last_of('/');
    if (slash == std::string::npos || slash == 0) {
        return true;
    }
    const std::string directory = outputPath.substr(0, slash);
    std::string current;
    std::string::size_type begin = 0;
    if (directory[0] == '/') {
        current = "/";
        begin = 1;
    }
    while (begin < directory.size()) {
        const std::string::size_type end = directory.find('/', begin);
        std::string component;
        if (end == std::string::npos) {
            component = directory.substr(begin);
        } else {
            component = directory.substr(begin, end - begin);
        }
        if (!component.empty()) {
            if (current.size() > 1 && current[current.size() - 1] != '/') {
                current += '/';
            }
            current += component;
            if (mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
                return false;
            }
        }
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return true;
}

void MetricExpLogger::WriteHeader() {
    mOutput << "timestamp,frame_id,tracking_state,reference_kf_id,pose_convention,pose_valid,"
            << "tx,ty,tz,qx,qy,qz,qw,"
            << "n_matched,n_inlier,inlier_ratio,"
            << "reproj_rmse_px,reproj_median_px,reproj_p90_px,"
            << "parallax_count,parallax_median_deg,parallax_p25_deg,"
            << "occupied_grid_count,grid_occupancy";
    if (mMetricExp2Enabled) {
        mOutput << ",hessian_valid,hessian_edge_count,hessian_rank,"
                << "hessian_rank_deficient,hessian_negative_eigenvalue,"
                << "hessian_raw_min_eig,hessian_eig_1,hessian_eig_2,"
                << "hessian_eig_3,hessian_eig_4,hessian_eig_5,hessian_eig_6,"
                << "hessian_trace,hessian_min_trace_ratio,hessian_min_max_ratio,"
                << "hessian_log10_condition,trans_obs_valid,"
                << "trans_obs_negative_eigenvalue,trans_obs_raw_min_eig,"
                << "trans_obs_eig_min,trans_obs_eig_mid,trans_obs_eig_max,"
                << "trans_obs_trace,trans_obs_min_trace_ratio,"
                << "trans_obs_min_max_ratio,trans_obs_log10_condition";
        if (mMetricExpDirectionEnabled) {
            mOutput << ",trans_obs_h00,trans_obs_h01,trans_obs_h02,"
                    << "trans_obs_h11,trans_obs_h12,trans_obs_h22";
        }
    }
    mOutput << '\n';
}

MetricExpFrameRecord::MetricExpFrameRecord()
    : mRecord(), mActive(false), mFinalized(false), mFlushed(false) {
}

void MetricExpFrameRecord::Begin(double timestamp, unsigned long frameId) {
    if (mActive && !mFlushed) {
        throw std::logic_error("metricexp frame record overwritten before flush");
    }
    mRecord = MetricExpRecord();
    mRecord.timestamp = timestamp;
    mRecord.frameId = frameId;
    mActive = true;
    mFinalized = false;
    mFlushed = false;
}

void MetricExpFrameRecord::Finalize(const MetricExpRecord &snapshot) {
    if (!mActive) {
        return;
    }
    if (mFinalized) {
        return;
    }
    const double inputTimestamp = mRecord.timestamp;
    const unsigned long inputFrameId = mRecord.frameId;
    mRecord = snapshot;
    mRecord.timestamp = inputTimestamp;
    mRecord.frameId = inputFrameId;
    mFinalized = true;
}

bool MetricExpFrameRecord::Flush(MetricExpLogger &logger) {
    if (!mActive || mFlushed) {
        return false;
    }
    if (!mFinalized) {
        throw std::logic_error("metricexp frame record flushed before finalize");
    }
    logger.Write(mRecord);
    mFlushed = true;
    mActive = false;
    return true;
}

bool MetricExpFrameRecord::IsActive() const {
    return mActive;
}

bool MetricExpFrameRecord::IsFinalized() const {
    return mFinalized;
}

bool MetricExpFrameRecord::IsFlushed() const {
    return mFlushed;
}

MetricExpRecord &MetricExpFrameRecord::MutableRecord() {
    if (!mActive || mFinalized) {
        throw std::logic_error("metricexp frame record is not mutable");
    }
    return mRecord;
}

const MetricExpRecord &MetricExpFrameRecord::GetRecord() const {
    return mRecord;
}

}  // namespace ORB_SLAM3
