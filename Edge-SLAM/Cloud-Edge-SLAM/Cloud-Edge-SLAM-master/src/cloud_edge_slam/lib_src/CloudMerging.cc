/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 * Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with ORB-SLAM3.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "CloudMerging.h"

#include "KeyFrame.h"
#include "MapPoint.h"
#include "Sim3Solver.h"
#include "UmeyamaSim3Solver.h"
#include "Converter.h"
#include "Optimizer.h"
#include "ORBmatcher.h"
#include "G2oTypes.h"
#include "Thirdparty/g2o/g2o/types/sim3.h"
#include "opencv2/core/persistence.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"
#include "sophus/sim3.hpp"
#include "SVIn2ORBWrapper.h" // [Sim3合并实验] 查询 OKVIS / SVIn2 前端位姿缓存

#include <Eigen/src/Core/ArithmeticSequence.h>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/SVD>
#include <algorithm>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <sstream>
#include <utility>
#include <limits>
#include <cmath>
#include <map>
#include <vector>

//cap-udf
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/time.h> 

// [Sim3合并实验] 全局 wrapper 指针定义在 SVIn2ORBWrapper.cpp 中。
// CloudMerging 通过该指针按时间戳查询 OKVIS / SVIn2 前端位姿，用于 Sim(3) 点对构造。
extern SVIn2ORBWrapper* pSVIn2ORBWrapper;

namespace ORB_SLAM3 {

namespace {

constexpr double kCloudCorrectionDebugZWarningThreshold = 0.005;
constexpr double kCloudCorrectionDebugBoundaryGapThreshold = 0.02;
// [CloudMap校正诊断] Sim(3) 诊断只使用前端位姿缓存做虚拟尺度对齐，不写回 KeyFrame。
constexpr double kSim3FrontendPoseQueryTolerance = 0.05;
constexpr int kSim3MinAlignmentPairs = 5;

// [Sim3合并实验] CloudMap 校正模式。
// SE3_TIME_LINEAR_TWO_ANCHOR：旧的严格双锚点 SE(3) TIME-LINEAR 残差插值。
// SIM3_ONLY：使用 CloudMap 与 frontend/edge 对应点估计 Sim(3)，只应用 Sim(3)，不再做 two-anchor 残差插值。
// SIM3_PLUS_RESIDUAL_RESERVED：预留模式，后续用于 Sim(3) 后的小残差修正。
enum class CloudMapCorrectionMode {
    SE3_TIME_LINEAR_TWO_ANCHOR = 0,
    SIM3_ONLY = 1,
    SIM3_PLUS_RESIDUAL_RESERVED = 2
};

// [Sim3合并实验] 当前默认切换为 SIM3_ONLY，用于验证尺度感知对齐是否能改善 CloudMap Z 轴问题。
constexpr CloudMapCorrectionMode kCloudMapCorrectionMode =
    CloudMapCorrectionMode::SIM3_ONLY;

// [Sim3合并实验] 用于查询 frontend / OKVIS 位姿的时间容差。
constexpr double kCloudMergeSim3FrontendPoseQueryTolerance = 0.05;
// [Sim3合并实验] 估计 Sim(3) 所需的最少匹配点数量。
constexpr size_t kCloudMergeSim3MinPairs = 10;
// [Sim3合并实验] Sim(3) 成功后的最大允许匹配 RMSE。当前先设宽松一点，避免误杀有效实验。
constexpr double kCloudMergeSim3MaxPairRmse = 0.05;
// [Sim3合并实验] Sim(3) 成功后的最大允许边界 gap。当前根据前期诊断，1-2 cm 可接受，先设为 0.05 m。
constexpr double kCloudMergeSim3MaxBoundaryGap = 0.05;
// [Sim3合并实验] 防止数值异常的尺度范围。
constexpr double kCloudMergeSim3MinScale = 0.01;
constexpr double kCloudMergeSim3MaxScale = 100.0;

// [SE3基线] 统一过滤无效 CloudMap KeyFrame，避免补偿阶段触碰空指针或 bad keyframe。
bool IsValidCloudMergeKeyFrame(KeyFrame *pKF) {
    if (pKF == nullptr) {
        return false;
    }

    if (pKF->isBad()) {
        return false;
    }

    if (!std::isfinite(pKF->mTimeStamp)) {
        return false;
    }

    return true;
}

// [CloudMap校正诊断] 缓存一个 CloudMap KeyFrame 在 raw/head/after 三个阶段的 Twc 位姿。
struct CloudCorrectionDebugPose {
    KeyFrame *pKF;
    double timestamp;
    unsigned long keyFrameId;
    Sophus::SE3f TwcRaw;
    Sophus::SE3f TwcHeadAligned;
    Sophus::SE3f TwcSim3Aligned;
    Sophus::SE3f TwcAfter;
    bool bHasSim3Aligned;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

using CloudCorrectionDebugPoseVector =
    std::vector<CloudCorrectionDebugPose, Eigen::aligned_allocator<CloudCorrectionDebugPose>>;

// [CloudMap校正诊断] 记录一个 CloudMap raw 位置到 OKVIS / SVIn2 前端位置的 Sim(3) 估计匹配点。
struct CloudSim3AlignmentPair {
    double timestamp;
    unsigned long keyFrameId;
    Eigen::Vector3d cloudRawPosition;
    Eigen::Vector3d edgePosition;
    Eigen::Vector3d sim3Position;
    double rawToEdgeError;
    double sim3ToEdgeError;
    double timeGap;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

using CloudSim3AlignmentPairVector =
    std::vector<CloudSim3AlignmentPair, Eigen::aligned_allocator<CloudSim3AlignmentPair>>;

// [CloudMap校正诊断] Sim(3) 位置对齐结果，仅用于诊断虚拟轨迹导出。
struct CloudSim3AlignmentResult {
    bool bOk;
    std::string status;
    int numPairs;
    double scale;
    Eigen::Matrix3d rotation;
    Eigen::Vector3d translation;
    double rotationAngleDeg;
    double rmseBeforeSim3;
    double rmseAfterSim3;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// [Sim3合并实验] Sim(3) 估计结果，方向为 p_frontend ≈ scale * R * p_cloud + t。
struct CloudMergeSim3Result {
    bool success;
    double scale;
    Eigen::Matrix3f R;
    Eigen::Vector3f t;
    double rmseBefore;
    double rmseAfter;
    size_t numPairs;
    std::string failureReason;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// [Sim3合并实验] 返回当前 CloudMap 校正模式名称，用于日志和 summary。
const char *GetCloudMapCorrectionModeName(const CloudMapCorrectionMode mode) {
    if (mode == CloudMapCorrectionMode::SE3_TIME_LINEAR_TWO_ANCHOR) {
        return "SE3_TIME_LINEAR_TWO_ANCHOR";
    }

    if (mode == CloudMapCorrectionMode::SIM3_ONLY) {
        return "SIM3_ONLY";
    }

    if (mode == CloudMapCorrectionMode::SIM3_PLUS_RESIDUAL_RESERVED) {
        return "SIM3_PLUS_RESIDUAL_RESERVED";
    }

    return "UNKNOWN";
}

// [Sim3合并实验] 创建默认失败状态，避免未初始化数值进入日志。
CloudMergeSim3Result MakeCloudMergeSim3Result() {
    CloudMergeSim3Result result;
    result.success = false;
    result.scale = 1.0;
    result.R = Eigen::Matrix3f::Identity();
    result.t = Eigen::Vector3f::Zero();
    result.rmseBefore = 0.0;
    result.rmseAfter = 0.0;
    result.numPairs = 0;
    result.failureReason = "not_requested";
    return result;
}

// [CloudMap校正诊断] 按时间过滤并排序当前 CloudMap 关键帧集合。
std::vector<KeyFrame *> BuildSortedCloudDebugKeyFrames(const std::vector<KeyFrame *> &vCloudKeyFrames) {
    std::vector<KeyFrame *> vSortedCloudKeyFrames;
    vSortedCloudKeyFrames.reserve(vCloudKeyFrames.size());

    for (size_t i = 0; i < vCloudKeyFrames.size(); i++) {
        KeyFrame *pKF = vCloudKeyFrames[i];
        if (!IsValidCloudMergeKeyFrame(pKF)) {
            continue;
        }

        vSortedCloudKeyFrames.push_back(pKF);
    }

    std::sort(vSortedCloudKeyFrames.begin(), vSortedCloudKeyFrames.end(), [](KeyFrame *pLeft, KeyFrame *pRight) {
        return pLeft->mTimeStamp < pRight->mTimeStamp;
    });

    return vSortedCloudKeyFrames;
}

// [CloudMap校正诊断] 将 Sim3 单位尺度位姿转成 SE3f，仅用于调试轨迹输出。
Sophus::SE3f Sim3ToSE3f(const Sophus::Sim3d &sim3Pose) {
    Sophus::SE3f pose(sim3Pose.rotationMatrix().cast<float>(), sim3Pose.translation().cast<float>());
    return pose;
}

// [CloudMap校正诊断] 构造 raw 和 head-aligned-only 调试快照，不写回 KeyFrame。
CloudCorrectionDebugPoseVector BuildCloudCorrectionDebugSnapshots(
    const std::vector<KeyFrame *> &vSortedCloudKeyFrames,
    const Sophus::Sim3d &SHead) {
    CloudCorrectionDebugPoseVector vSnapshots;
    vSnapshots.reserve(vSortedCloudKeyFrames.size());

    for (size_t i = 0; i < vSortedCloudKeyFrames.size(); i++) {
        KeyFrame *pKF = vSortedCloudKeyFrames[i];
        if (!IsValidCloudMergeKeyFrame(pKF)) {
            continue;
        }

        Sophus::SE3f TwcRaw = pKF->GetPoseInverse();
        Sophus::Sim3d rawSim3(TwcRaw.unit_quaternion().cast<double>(), TwcRaw.translation().cast<double>());
        rawSim3.setScale(1.0);
        Sophus::SE3f TwcHeadAligned = Sim3ToSE3f(SHead * rawSim3);

        CloudCorrectionDebugPose snapshot;
        snapshot.pKF = pKF;
        snapshot.timestamp = pKF->mTimeStamp;
        snapshot.keyFrameId = pKF->mnId;
        snapshot.TwcRaw = TwcRaw;
        snapshot.TwcHeadAligned = TwcHeadAligned;
        snapshot.TwcSim3Aligned = TwcHeadAligned;
        snapshot.TwcAfter = TwcRaw;
        snapshot.bHasSim3Aligned = false;
        vSnapshots.push_back(snapshot);
    }

    return vSnapshots;
}

// [CloudMap校正诊断] 校正完成后读取 KeyFrame 当前实际位姿。
void FillCloudCorrectionDebugAfterPose(CloudCorrectionDebugPoseVector &vSnapshots) {
    for (size_t i = 0; i < vSnapshots.size(); i++) {
        if (!IsValidCloudMergeKeyFrame(vSnapshots[i].pKF)) {
            continue;
        }

        vSnapshots[i].TwcAfter = vSnapshots[i].pKF->GetPoseInverse();
    }
}

// [CloudMap校正诊断] 创建 Sim(3) 诊断默认结果。
CloudSim3AlignmentResult MakeCloudSim3AlignmentResult(const std::string &status) {
    CloudSim3AlignmentResult result;
    result.bOk = false;
    result.status = status;
    result.numPairs = 0;
    result.scale = std::numeric_limits<double>::quiet_NaN();
    result.rotation = Eigen::Matrix3d::Identity();
    result.translation = Eigen::Vector3d::Zero();
    result.rotationAngleDeg = std::numeric_limits<double>::quiet_NaN();
    result.rmseBeforeSim3 = std::numeric_limits<double>::quiet_NaN();
    result.rmseAfterSim3 = std::numeric_limits<double>::quiet_NaN();
    return result;
}

// [CloudMap校正诊断] 收集 CloudMap raw 位置与最近 OKVIS / SVIn2 前端位置的时间匹配点。
CloudSim3AlignmentPairVector BuildCloudSim3AlignmentPairs(
    const CloudCorrectionDebugPoseVector &vSnapshots) {
    CloudSim3AlignmentPairVector vPairs;

    if (::pSVIn2ORBWrapper == nullptr) {
        return vPairs;
    }

    vPairs.reserve(vSnapshots.size());
    for (size_t i = 0; i < vSnapshots.size(); i++) {
        if (!IsValidCloudMergeKeyFrame(vSnapshots[i].pKF)) {
            continue;
        }

        Sophus::SE3f TwcFrontend;
        double timeGap = 0.0;
        const bool bFoundFrontendPose = ::pSVIn2ORBWrapper->GetNearestFrontendPoseWithTimeGap(
            vSnapshots[i].timestamp,
            kSim3FrontendPoseQueryTolerance,
            TwcFrontend,
            timeGap);

        if (!bFoundFrontendPose) {
            continue;
        }

        const Eigen::Vector3d cloudRawPosition = vSnapshots[i].TwcRaw.translation().cast<double>();
        const Eigen::Vector3d edgePosition = TwcFrontend.translation().cast<double>();
        if (!cloudRawPosition.allFinite()) {
            continue;
        }

        if (!edgePosition.allFinite()) {
            continue;
        }

        CloudSim3AlignmentPair pair;
        pair.timestamp = vSnapshots[i].timestamp;
        pair.keyFrameId = vSnapshots[i].keyFrameId;
        pair.cloudRawPosition = cloudRawPosition;
        pair.edgePosition = edgePosition;
        pair.sim3Position = cloudRawPosition;
        pair.rawToEdgeError = (cloudRawPosition - edgePosition).norm();
        pair.sim3ToEdgeError = std::numeric_limits<double>::quiet_NaN();
        pair.timeGap = timeGap;
        vPairs.push_back(pair);
    }

    return vPairs;
}

// [CloudMap校正诊断] 使用 Umeyama similarity alignment 估计 p_edge ≈ s * R * p_cloud + t。
CloudSim3AlignmentResult EstimateCloudSim3Alignment(
    CloudSim3AlignmentPairVector &vPairs) {
    CloudSim3AlignmentResult result = MakeCloudSim3AlignmentResult("failed_numeric");
    result.numPairs = static_cast<int>(vPairs.size());

    if (vPairs.size() < static_cast<size_t>(kSim3MinAlignmentPairs)) {
        result.status = "failed_insufficient_pairs";
        return result;
    }

    Eigen::Vector3d sourceMean = Eigen::Vector3d::Zero();
    Eigen::Vector3d targetMean = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < vPairs.size(); i++) {
        sourceMean += vPairs[i].cloudRawPosition;
        targetMean += vPairs[i].edgePosition;
    }

    const double invCount = 1.0 / static_cast<double>(vPairs.size());
    sourceMean *= invCount;
    targetMean *= invCount;

    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    double sourceVariance = 0.0;
    for (size_t i = 0; i < vPairs.size(); i++) {
        const Eigen::Vector3d sourceCentered = vPairs[i].cloudRawPosition - sourceMean;
        const Eigen::Vector3d targetCentered = vPairs[i].edgePosition - targetMean;
        covariance += targetCentered * sourceCentered.transpose();
        sourceVariance += sourceCentered.squaredNorm();
    }

    covariance *= invCount;
    sourceVariance *= invCount;

    if (!covariance.allFinite()) {
        return result;
    }

    if (!std::isfinite(sourceVariance)) {
        return result;
    }

    if (sourceVariance <= std::numeric_limits<double>::epsilon()) {
        return result;
    }

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Matrix3d U = svd.matrixU();
    const Eigen::Matrix3d V = svd.matrixV();
    if (!U.allFinite() || !V.allFinite()) {
        return result;
    }

    Eigen::Matrix3d reflectionGuard = Eigen::Matrix3d::Identity();
    if ((U * V.transpose()).determinant() < 0.0) {
        reflectionGuard(2, 2) = -1.0;
    }

    const Eigen::Matrix3d rotation = U * reflectionGuard * V.transpose();
    if (!rotation.allFinite()) {
        return result;
    }

    if (rotation.determinant() <= 0.0) {
        return result;
    }

    const Eigen::Vector3d singularValues = svd.singularValues();
    const Eigen::Vector3d reflectionSigns = reflectionGuard.diagonal();
    const double scaleNumerator = singularValues.dot(reflectionSigns);
    const double scale = scaleNumerator / sourceVariance;
    if (!std::isfinite(scale)) {
        return result;
    }

    if (scale <= 0.0) {
        return result;
    }

    const Eigen::Vector3d translation = targetMean - scale * rotation * sourceMean;
    if (!translation.allFinite()) {
        return result;
    }

    double sumBeforeSquared = 0.0;
    double sumAfterSquared = 0.0;
    for (size_t i = 0; i < vPairs.size(); i++) {
        const Eigen::Vector3d sim3Position = scale * rotation * vPairs[i].cloudRawPosition + translation;
        if (!sim3Position.allFinite()) {
            return result;
        }

        vPairs[i].sim3Position = sim3Position;
        vPairs[i].rawToEdgeError = (vPairs[i].cloudRawPosition - vPairs[i].edgePosition).norm();
        vPairs[i].sim3ToEdgeError = (sim3Position - vPairs[i].edgePosition).norm();
        sumBeforeSquared += vPairs[i].rawToEdgeError * vPairs[i].rawToEdgeError;
        sumAfterSquared += vPairs[i].sim3ToEdgeError * vPairs[i].sim3ToEdgeError;
    }

    const double pi = std::acos(-1.0);
    Eigen::AngleAxisd angleAxis(rotation);

    result.bOk = true;
    result.status = "ok";
    result.scale = scale;
    result.rotation = rotation;
    result.translation = translation;
    result.rotationAngleDeg = std::abs(angleAxis.angle()) * 180.0 / pi;
    result.rmseBeforeSim3 = std::sqrt(sumBeforeSquared / static_cast<double>(vPairs.size()));
    result.rmseAfterSim3 = std::sqrt(sumAfterSquared / static_cast<double>(vPairs.size()));
    return result;
}

// [Sim3合并实验] 使用 CloudMap KeyFrame 与 frontend/OKVIS 位姿缓存估计 Sim(3)。
// 输入 cloud KeyFrame 的校正前 Twc.translation() 作为 cloud 点；
// 输入同时间 frontend Twc.translation() 作为 edge/frontend 点；
// 估计 p_frontend = s * R * p_cloud + t。
bool EstimateCloudToFrontendSim3(
    const std::vector<ORB_SLAM3::KeyFrame *> &vCloudKeyFrames,
    CloudMergeSim3Result &sim3Result,
    std::vector<double> &matchedTimestamps,
    std::vector<unsigned long> &matchedKeyFrameIds,
    std::vector<Eigen::Vector3f> &cloudPoints,
    std::vector<Eigen::Vector3f> &frontendPoints,
    std::vector<double> &timeGaps) {
    sim3Result = MakeCloudMergeSim3Result();
    sim3Result.failureReason = "failed_numeric";

    matchedTimestamps.clear();
    matchedKeyFrameIds.clear();
    cloudPoints.clear();
    frontendPoints.clear();
    timeGaps.clear();

    if (::pSVIn2ORBWrapper == nullptr) {
        sim3Result.failureReason = "frontend_wrapper_unavailable";
        return false;
    }

    for (size_t i = 0; i < vCloudKeyFrames.size(); i++) {
        KeyFrame *pKF = vCloudKeyFrames[i];
        if (!IsValidCloudMergeKeyFrame(pKF)) {
            continue;
        }

        const Sophus::SE3f TwcCloud = pKF->GetPoseInverse();
        const Eigen::Vector3f pCloud = TwcCloud.translation();
        if (!pCloud.allFinite()) {
            continue;
        }

        Sophus::SE3f TwcFrontend;
        double timeGap = 0.0;
        const bool bFoundFrontendPose = ::pSVIn2ORBWrapper->GetNearestFrontendPoseWithTimeGap(
            pKF->mTimeStamp,
            kCloudMergeSim3FrontendPoseQueryTolerance,
            TwcFrontend,
            timeGap);

        if (!bFoundFrontendPose) {
            continue;
        }

        const Eigen::Vector3f pFrontend = TwcFrontend.translation();
        if (!pFrontend.allFinite()) {
            continue;
        }

        matchedTimestamps.push_back(pKF->mTimeStamp);
        matchedKeyFrameIds.push_back(pKF->mnId);
        cloudPoints.push_back(pCloud);
        frontendPoints.push_back(pFrontend);
        timeGaps.push_back(timeGap);
    }

    sim3Result.numPairs = cloudPoints.size();
    if (cloudPoints.size() < kCloudMergeSim3MinPairs) {
        sim3Result.failureReason = "insufficient_pairs";
        return false;
    }

    Eigen::Vector3d cloudMean = Eigen::Vector3d::Zero();
    Eigen::Vector3d frontendMean = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < cloudPoints.size(); i++) {
        cloudMean += cloudPoints[i].cast<double>();
        frontendMean += frontendPoints[i].cast<double>();
    }

    const double invCount = 1.0 / static_cast<double>(cloudPoints.size());
    cloudMean *= invCount;
    frontendMean *= invCount;

    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    double cloudVariance = 0.0;
    double sumBeforeSquared = 0.0;
    for (size_t i = 0; i < cloudPoints.size(); i++) {
        const Eigen::Vector3d pCloud = cloudPoints[i].cast<double>();
        const Eigen::Vector3d pFrontend = frontendPoints[i].cast<double>();
        const Eigen::Vector3d cloudCentered = pCloud - cloudMean;
        const Eigen::Vector3d frontendCentered = pFrontend - frontendMean;

        covariance += frontendCentered * cloudCentered.transpose();
        cloudVariance += cloudCentered.squaredNorm();
        sumBeforeSquared += (pCloud - pFrontend).squaredNorm();
    }

    covariance *= invCount;
    cloudVariance *= invCount;

    if (!covariance.allFinite()) {
        sim3Result.failureReason = "invalid_covariance";
        return false;
    }

    if (!std::isfinite(cloudVariance)) {
        sim3Result.failureReason = "invalid_cloud_variance";
        return false;
    }

    if (cloudVariance <= std::numeric_limits<double>::epsilon()) {
        sim3Result.failureReason = "degenerate_cloud_variance";
        return false;
    }

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Matrix3d U = svd.matrixU();
    const Eigen::Matrix3d V = svd.matrixV();
    if (!U.allFinite() || !V.allFinite()) {
        sim3Result.failureReason = "invalid_svd";
        return false;
    }

    Eigen::Matrix3d signCorrection = Eigen::Matrix3d::Identity();
    if ((U * V.transpose()).determinant() < 0.0) {
        signCorrection(2, 2) = -1.0;
    }

    const Eigen::Matrix3d R = U * signCorrection * V.transpose();
    if (!R.allFinite()) {
        sim3Result.failureReason = "invalid_rotation";
        return false;
    }

    if (R.determinant() <= 0.0) {
        sim3Result.failureReason = "reflection_rotation";
        return false;
    }

    const Eigen::Vector3d singularValues = svd.singularValues();
    const Eigen::Vector3d signValues = signCorrection.diagonal();
    const double scale = singularValues.dot(signValues) / cloudVariance;
    if (!std::isfinite(scale)) {
        sim3Result.failureReason = "invalid_scale";
        return false;
    }

    if (scale < kCloudMergeSim3MinScale || scale > kCloudMergeSim3MaxScale) {
        sim3Result.failureReason = "scale_out_of_range";
        return false;
    }

    const Eigen::Vector3d t = frontendMean - scale * R * cloudMean;
    if (!t.allFinite()) {
        sim3Result.failureReason = "invalid_translation";
        return false;
    }

    double sumAfterSquared = 0.0;
    for (size_t i = 0; i < cloudPoints.size(); i++) {
        const Eigen::Vector3d pCloud = cloudPoints[i].cast<double>();
        const Eigen::Vector3d pFrontend = frontendPoints[i].cast<double>();
        const Eigen::Vector3d pSim3 = scale * R * pCloud + t;
        if (!pSim3.allFinite()) {
            sim3Result.failureReason = "invalid_transformed_point";
            return false;
        }

        sumAfterSquared += (pSim3 - pFrontend).squaredNorm();
    }

    sim3Result.rmseBefore = std::sqrt(sumBeforeSquared / static_cast<double>(cloudPoints.size()));
    sim3Result.rmseAfter = std::sqrt(sumAfterSquared / static_cast<double>(cloudPoints.size()));
    if (!std::isfinite(sim3Result.rmseAfter)) {
        sim3Result.failureReason = "invalid_rmse_after";
        return false;
    }

    if (sim3Result.rmseAfter > kCloudMergeSim3MaxPairRmse) {
        sim3Result.failureReason = "rmse_after_too_large";
        return false;
    }

    sim3Result.success = true;
    sim3Result.scale = scale;
    sim3Result.R = R.cast<float>();
    sim3Result.t = t.cast<float>();
    sim3Result.failureReason = "";
    return true;
}

// [Sim3合并实验] 将估计得到的 Sim(3) 应用到 CloudMap KeyFrame 和 MapPoint。
bool ApplySim3ToCloudMap(
    ORB_SLAM3::Map *pCloudMap,
    const std::vector<ORB_SLAM3::KeyFrame *> &vCloudKeyFrames,
    const CloudMergeSim3Result &sim3Result) {
    if (pCloudMap == nullptr) {
        return false;
    }

    if (!sim3Result.success) {
        return false;
    }

    for (size_t i = 0; i < vCloudKeyFrames.size(); i++) {
        KeyFrame *pKF = vCloudKeyFrames[i];
        if (!IsValidCloudMergeKeyFrame(pKF)) {
            continue;
        }

        const Sophus::SE3f TwcOld = pKF->GetPoseInverse();
        const Eigen::Matrix3f RwcOld = TwcOld.rotationMatrix();
        const Eigen::Vector3f twcOld = TwcOld.translation();

        Eigen::Matrix3f RwcNew = sim3Result.R * RwcOld;
        Eigen::Quaternionf qwcNew(RwcNew);
        qwcNew.normalize();
        RwcNew = qwcNew.toRotationMatrix();

        const Eigen::Vector3f twcNew = static_cast<float>(sim3Result.scale) * sim3Result.R * twcOld + sim3Result.t;

        Sophus::SE3f TwcNew(RwcNew, twcNew);
        pKF->SetPose(TwcNew.inverse());
    }

    const std::vector<MapPoint *> vpCloudMapPoints = pCloudMap->GetAllMapPoints();
    for (size_t i = 0; i < vpCloudMapPoints.size(); i++) {
        MapPoint *pMP = vpCloudMapPoints[i];
        if (pMP == nullptr) {
            continue;
        }

        if (pMP->isBad()) {
            continue;
        }

        const Eigen::Vector3f PwOld = pMP->GetWorldPos();
        if (!PwOld.allFinite()) {
            continue;
        }

        const Eigen::Vector3f PwNew = static_cast<float>(sim3Result.scale) * sim3Result.R * PwOld + sim3Result.t;
        if (!PwNew.allFinite()) {
            continue;
        }

        pMP->SetWorldPos(PwNew);
        pMP->UpdateNormalAndDepth();
    }

    return true;
}

// [CloudMap校正诊断] 将估计的 Sim(3) 应用到调试快照中的虚拟轨迹，不写回 KeyFrame。
void FillCloudSim3AlignedDebugPose(
    CloudCorrectionDebugPoseVector &vSnapshots,
    const CloudSim3AlignmentResult &sim3Result) {
    if (!sim3Result.bOk) {
        return;
    }

    const Eigen::Matrix3f rotation = sim3Result.rotation.cast<float>();
    const Eigen::Vector3f translation = sim3Result.translation.cast<float>();
    const float scale = static_cast<float>(sim3Result.scale);

    for (size_t i = 0; i < vSnapshots.size(); i++) {
        const Eigen::Vector3f rawPosition = vSnapshots[i].TwcRaw.translation();
        const Eigen::Vector3f sim3Position = scale * rotation * rawPosition + translation;
        const Eigen::Matrix3f sim3Rotation = rotation * vSnapshots[i].TwcRaw.rotationMatrix();
        Eigen::Quaternionf sim3Quaternion(sim3Rotation);
        sim3Quaternion.normalize();

        vSnapshots[i].TwcSim3Aligned = Sophus::SE3f(sim3Quaternion.toRotationMatrix(), sim3Position);
        vSnapshots[i].bHasSim3Aligned = true;
    }
}

// [CloudMap校正诊断] 计算轨迹长度。
double ComputeCloudCorrectionDebugPathLength(
    const CloudCorrectionDebugPoseVector &vSnapshots,
    const std::string &poseStage) {
    if (vSnapshots.size() < 2) {
        return 0.0;
    }

    double pathLength = 0.0;
    for (size_t i = 1; i < vSnapshots.size(); i++) {
        const Sophus::SE3f *pPrevPose = nullptr;
        const Sophus::SE3f *pCurrPose = nullptr;

        if (poseStage == "raw") {
            pPrevPose = &vSnapshots[i - 1].TwcRaw;
            pCurrPose = &vSnapshots[i].TwcRaw;
        } else if (poseStage == "head") {
            pPrevPose = &vSnapshots[i - 1].TwcHeadAligned;
            pCurrPose = &vSnapshots[i].TwcHeadAligned;
        } else if (poseStage == "sim3") {
            if (vSnapshots[i - 1].bHasSim3Aligned && vSnapshots[i].bHasSim3Aligned) {
                pPrevPose = &vSnapshots[i - 1].TwcSim3Aligned;
                pCurrPose = &vSnapshots[i].TwcSim3Aligned;
            }
        } else if (poseStage == "after") {
            pPrevPose = &vSnapshots[i - 1].TwcAfter;
            pCurrPose = &vSnapshots[i].TwcAfter;
        }

        if (pPrevPose == nullptr || pCurrPose == nullptr) {
            continue;
        }

        pathLength += static_cast<double>((pCurrPose->translation() - pPrevPose->translation()).norm());
    }

    return pathLength;
}

// [CloudMap校正诊断] 统计 Z 范围与均值。
void ComputeCloudCorrectionDebugZStats(
    const CloudCorrectionDebugPoseVector &vSnapshots,
    const std::string &poseStage,
    double &zMin,
    double &zMax,
    double &zMean) {
    zMin = std::numeric_limits<double>::quiet_NaN();
    zMax = std::numeric_limits<double>::quiet_NaN();
    zMean = std::numeric_limits<double>::quiet_NaN();

    if (vSnapshots.empty()) {
        return;
    }

    double zSum = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < vSnapshots.size(); i++) {
        const Sophus::SE3f *pPose = nullptr;
        if (poseStage == "raw") {
            pPose = &vSnapshots[i].TwcRaw;
        } else if (poseStage == "head") {
            pPose = &vSnapshots[i].TwcHeadAligned;
        } else if (poseStage == "sim3") {
            if (vSnapshots[i].bHasSim3Aligned) {
                pPose = &vSnapshots[i].TwcSim3Aligned;
            }
        } else if (poseStage == "after") {
            pPose = &vSnapshots[i].TwcAfter;
        }

        if (pPose == nullptr) {
            continue;
        }

        const double z = static_cast<double>(pPose->translation().z());
        if (count == 0) {
            zMin = z;
            zMax = z;
        } else {
            if (z < zMin) {
                zMin = z;
            }

            if (z > zMax) {
                zMax = z;
            }
        }

        zSum += z;
        count++;
    }

    if (count > 0) {
        zMean = zSum / static_cast<double>(count);
    }
}

// [CloudMap校正诊断] 计算两点平移距离。
double ComputeCloudCorrectionDebugGap(const Sophus::SE3f &left, const Sophus::SE3f &right) {
    return static_cast<double>((left.translation() - right.translation()).norm());
}

// [CloudMap校正诊断] 生成固定三位 merge index 文件前缀。
std::string BuildCloudCorrectionDebugPrefix(const std::string &outputDir, const int mergeIndex) {
    std::ostringstream oss;
    oss << outputDir << "/cloud_merge_"
        << std::setw(3) << std::setfill('0') << mergeIndex;
    return oss.str();
}

// [CloudMap校正诊断] 写 Sim(3) 尺度对齐诊断 summary。
void WriteCloudSim3AlignmentDebugSummary(
    const std::string &path,
    const int mergeIndex,
    const CloudCorrectionDebugPoseVector &vSnapshots,
    const CloudSim3AlignmentPairVector &vPairs,
    const CloudSim3AlignmentResult &sim3Result,
    KeyFrame *pKFEdgeFrontAnchor,
    KeyFrame *pKFEdgeBackAnchor) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        cerr << "\033[1;31m[CloudMap Sim3 Debug] Failed to open summary: "
             << path << "\033[0m" << endl;
        return;
    }

    double rawStartToEdgeFrontGap = std::numeric_limits<double>::quiet_NaN();
    double rawEndToEdgeBackGap = std::numeric_limits<double>::quiet_NaN();
    double headStartToEdgeFrontGap = std::numeric_limits<double>::quiet_NaN();
    double headEndToEdgeBackGap = std::numeric_limits<double>::quiet_NaN();
    double sim3StartToEdgeFrontGap = std::numeric_limits<double>::quiet_NaN();
    double sim3EndToEdgeBackGap = std::numeric_limits<double>::quiet_NaN();
    double afterStartToEdgeFrontGap = std::numeric_limits<double>::quiet_NaN();
    double afterEndToEdgeBackGap = std::numeric_limits<double>::quiet_NaN();

    if (!vSnapshots.empty() && pKFEdgeFrontAnchor != nullptr && pKFEdgeBackAnchor != nullptr) {
        const Sophus::SE3f TwcEdgeFront = pKFEdgeFrontAnchor->GetPoseInverse();
        const Sophus::SE3f TwcEdgeBack = pKFEdgeBackAnchor->GetPoseInverse();

        rawStartToEdgeFrontGap = ComputeCloudCorrectionDebugGap(vSnapshots.front().TwcRaw, TwcEdgeFront);
        rawEndToEdgeBackGap = ComputeCloudCorrectionDebugGap(vSnapshots.back().TwcRaw, TwcEdgeBack);
        headStartToEdgeFrontGap = ComputeCloudCorrectionDebugGap(vSnapshots.front().TwcHeadAligned, TwcEdgeFront);
        headEndToEdgeBackGap = ComputeCloudCorrectionDebugGap(vSnapshots.back().TwcHeadAligned, TwcEdgeBack);
        afterStartToEdgeFrontGap = ComputeCloudCorrectionDebugGap(vSnapshots.front().TwcAfter, TwcEdgeFront);
        afterEndToEdgeBackGap = ComputeCloudCorrectionDebugGap(vSnapshots.back().TwcAfter, TwcEdgeBack);

        if (vSnapshots.front().bHasSim3Aligned && vSnapshots.back().bHasSim3Aligned) {
            sim3StartToEdgeFrontGap = ComputeCloudCorrectionDebugGap(vSnapshots.front().TwcSim3Aligned, TwcEdgeFront);
            sim3EndToEdgeBackGap = ComputeCloudCorrectionDebugGap(vSnapshots.back().TwcSim3Aligned, TwcEdgeBack);
        }
    }

    double rawZMin = 0.0;
    double rawZMax = 0.0;
    double rawZMean = 0.0;
    double headZMin = 0.0;
    double headZMax = 0.0;
    double headZMean = 0.0;
    double sim3ZMin = 0.0;
    double sim3ZMax = 0.0;
    double sim3ZMean = 0.0;
    double afterZMin = 0.0;
    double afterZMax = 0.0;
    double afterZMean = 0.0;

    ComputeCloudCorrectionDebugZStats(vSnapshots, "raw", rawZMin, rawZMax, rawZMean);
    ComputeCloudCorrectionDebugZStats(vSnapshots, "head", headZMin, headZMax, headZMean);
    ComputeCloudCorrectionDebugZStats(vSnapshots, "sim3", sim3ZMin, sim3ZMax, sim3ZMean);
    ComputeCloudCorrectionDebugZStats(vSnapshots, "after", afterZMin, afterZMax, afterZMean);

    const double rawPathLength = ComputeCloudCorrectionDebugPathLength(vSnapshots, "raw");
    const double headPathLength = ComputeCloudCorrectionDebugPathLength(vSnapshots, "head");
    const double sim3PathLength = ComputeCloudCorrectionDebugPathLength(vSnapshots, "sim3");
    const double afterPathLength = ComputeCloudCorrectionDebugPathLength(vSnapshots, "after");

    double sim3MinusHeadZSum = 0.0;
    double afterMinusSim3ZSum = 0.0;
    size_t sim3DeltaCount = 0;
    for (size_t i = 0; i < vSnapshots.size(); i++) {
        if (!vSnapshots[i].bHasSim3Aligned) {
            continue;
        }

        sim3MinusHeadZSum += static_cast<double>(
            vSnapshots[i].TwcSim3Aligned.translation().z() - vSnapshots[i].TwcHeadAligned.translation().z());
        afterMinusSim3ZSum += static_cast<double>(
            vSnapshots[i].TwcAfter.translation().z() - vSnapshots[i].TwcSim3Aligned.translation().z());
        sim3DeltaCount++;
    }

    double sim3MinusHeadZMean = std::numeric_limits<double>::quiet_NaN();
    double afterMinusSim3ZMean = std::numeric_limits<double>::quiet_NaN();
    if (sim3DeltaCount > 0) {
        sim3MinusHeadZMean = sim3MinusHeadZSum / static_cast<double>(sim3DeltaCount);
        afterMinusSim3ZMean = afterMinusSim3ZSum / static_cast<double>(sim3DeltaCount);
    }

    ofs << std::fixed << std::setprecision(9);
    ofs << "[Sim3 Alignment Debug Summary]" << std::endl;
    ofs << "merge_index = " << mergeIndex << std::endl;
    ofs << "sim3_status = " << sim3Result.status << std::endl;
    ofs << "num_pairs = " << sim3Result.numPairs << std::endl;
    ofs << "query_tolerance = " << kSim3FrontendPoseQueryTolerance << std::endl;
    ofs << "scale = " << sim3Result.scale << std::endl;
    ofs << "rotation_angle_deg = " << sim3Result.rotationAngleDeg << std::endl;
    ofs << "translation_x = " << sim3Result.translation.x() << std::endl;
    ofs << "translation_y = " << sim3Result.translation.y() << std::endl;
    ofs << "translation_z = " << sim3Result.translation.z() << std::endl;
    ofs << "rmse_before_sim3 = " << sim3Result.rmseBeforeSim3 << std::endl;
    ofs << "rmse_after_sim3 = " << sim3Result.rmseAfterSim3 << std::endl;
    ofs << std::endl;

    ofs << "[Boundary Gap]" << std::endl;
    ofs << "raw_start_to_edge_front_gap = " << rawStartToEdgeFrontGap << std::endl;
    ofs << "raw_end_to_edge_back_gap = " << rawEndToEdgeBackGap << std::endl;
    ofs << "head_start_to_edge_front_gap = " << headStartToEdgeFrontGap << std::endl;
    ofs << "head_end_to_edge_back_gap = " << headEndToEdgeBackGap << std::endl;
    ofs << "sim3_start_to_edge_front_gap = " << sim3StartToEdgeFrontGap << std::endl;
    ofs << "sim3_end_to_edge_back_gap = " << sim3EndToEdgeBackGap << std::endl;
    ofs << "after_start_to_edge_front_gap = " << afterStartToEdgeFrontGap << std::endl;
    ofs << "after_end_to_edge_back_gap = " << afterEndToEdgeBackGap << std::endl;
    ofs << std::endl;

    ofs << "[Path Length]" << std::endl;
    ofs << "raw_path_length = " << rawPathLength << std::endl;
    ofs << "head_path_length = " << headPathLength << std::endl;
    ofs << "sim3_path_length = " << sim3PathLength << std::endl;
    ofs << "after_path_length = " << afterPathLength << std::endl;
    ofs << std::endl;

    ofs << "[Z Statistics]" << std::endl;
    ofs << "raw_z_mean = " << rawZMean << std::endl;
    ofs << "head_z_mean = " << headZMean << std::endl;
    ofs << "sim3_z_mean = " << sim3ZMean << std::endl;
    ofs << "after_z_mean = " << afterZMean << std::endl;
    ofs << "sim3_minus_head_z_mean = " << sim3MinusHeadZMean << std::endl;
    ofs << "after_minus_sim3_z_mean = " << afterMinusSim3ZMean << std::endl;
    ofs << std::endl;

    ofs << "[Interpretation Hint]" << std::endl;
    if (sim3Result.bOk) {
        if (sim3Result.rmseAfterSim3 < sim3Result.rmseBeforeSim3) {
            ofs << "Sim3 alignment improves CloudMap-to-frontend point alignment." << std::endl;
        }

        if (std::isfinite(sim3EndToEdgeBackGap) && std::isfinite(headEndToEdgeBackGap)) {
            if (headEndToEdgeBackGap - sim3EndToEdgeBackGap > kCloudCorrectionDebugBoundaryGapThreshold) {
                ofs << "Scale-aware alignment reduces tail boundary residual." << std::endl;
            }
        }

        if (std::isfinite(afterPathLength) && std::isfinite(sim3PathLength)) {
            if (sim3PathLength - afterPathLength > kCloudCorrectionDebugBoundaryGapThreshold) {
                if (afterPathLength < sim3PathLength * 0.9) {
                    ofs << "Two-anchor correction compresses CloudMap compared with Sim3-aligned trajectory." << std::endl;
                }
            }
        }

        if (std::isfinite(afterMinusSim3ZMean)) {
            if (std::abs(afterMinusSim3ZMean) > kCloudCorrectionDebugZWarningThreshold) {
                ofs << "Two-anchor correction changes CloudMap Z compared with Sim3-aligned trajectory." << std::endl;
            }
        }
    } else {
        ofs << "Sim3 alignment was not estimated; inspect num_pairs and frontend pose cache coverage." << std::endl;
    }

    (void)vPairs;
    ofs.close();
}

// [Sim3合并实验] 按 key = value 写 double 或 not_executed 字段。
void WriteSelectedCorrectionDoubleOrText(
    std::ofstream &ofs,
    const std::string &key,
    const double value,
    const bool bExecuted) {
    ofs << key << " = ";
    if (bExecuted) {
        ofs << value;
    } else {
        ofs << "not_executed";
    }
    ofs << std::endl;
}

// [Sim3合并实验] 写当前被选中 CloudMap 校正模式的 summary。
void WriteCloudSelectedCorrectionSummary(
    const std::string &path,
    const int mergeIndex,
    const CloudMapCorrectionMode selectedMode,
    const CloudCorrectionDebugPoseVector &vSnapshots,
    const CloudMergeSim3Result &sim3Result,
    KeyFrame *pKFEdgeFrontAnchor,
    KeyFrame *pKFEdgeBackAnchor,
    const bool bTwoAnchorExecuted) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        cerr << "\033[1;31m[CloudMap Selected Correction] Failed to open summary: "
             << path << "\033[0m" << endl;
        return;
    }

    double rawZMin = 0.0;
    double rawZMax = 0.0;
    double rawZMean = 0.0;
    double headZMin = 0.0;
    double headZMax = 0.0;
    double headZMean = 0.0;
    double sim3ZMin = 0.0;
    double sim3ZMax = 0.0;
    double sim3ZMean = 0.0;
    double selectedZMin = 0.0;
    double selectedZMax = 0.0;
    double selectedZMean = 0.0;

    ComputeCloudCorrectionDebugZStats(vSnapshots, "raw", rawZMin, rawZMax, rawZMean);
    ComputeCloudCorrectionDebugZStats(vSnapshots, "head", headZMin, headZMax, headZMean);
    ComputeCloudCorrectionDebugZStats(vSnapshots, "sim3", sim3ZMin, sim3ZMax, sim3ZMean);
    ComputeCloudCorrectionDebugZStats(vSnapshots, "after", selectedZMin, selectedZMax, selectedZMean);

    const double rawPathLength = ComputeCloudCorrectionDebugPathLength(vSnapshots, "raw");
    const double headPathLength = ComputeCloudCorrectionDebugPathLength(vSnapshots, "head");
    const double sim3PathLength = ComputeCloudCorrectionDebugPathLength(vSnapshots, "sim3");
    const double selectedPathLength = ComputeCloudCorrectionDebugPathLength(vSnapshots, "after");
    double twoAnchorPathLength = std::numeric_limits<double>::quiet_NaN();
    if (bTwoAnchorExecuted) {
        twoAnchorPathLength = selectedPathLength;
    }

    double selectedStartToEdgeFrontGap = std::numeric_limits<double>::quiet_NaN();
    double selectedEndToEdgeBackGap = std::numeric_limits<double>::quiet_NaN();
    double headEndToEdgeBackGap = std::numeric_limits<double>::quiet_NaN();
    double twoAnchorEndToEdgeBackGap = std::numeric_limits<double>::quiet_NaN();

    if (!vSnapshots.empty() && pKFEdgeFrontAnchor != nullptr && pKFEdgeBackAnchor != nullptr) {
        const Sophus::SE3f TwcEdgeFront = pKFEdgeFrontAnchor->GetPoseInverse();
        const Sophus::SE3f TwcEdgeBack = pKFEdgeBackAnchor->GetPoseInverse();

        selectedStartToEdgeFrontGap = ComputeCloudCorrectionDebugGap(vSnapshots.front().TwcAfter, TwcEdgeFront);
        selectedEndToEdgeBackGap = ComputeCloudCorrectionDebugGap(vSnapshots.back().TwcAfter, TwcEdgeBack);
        headEndToEdgeBackGap = ComputeCloudCorrectionDebugGap(vSnapshots.back().TwcHeadAligned, TwcEdgeBack);
        if (bTwoAnchorExecuted) {
            twoAnchorEndToEdgeBackGap = selectedEndToEdgeBackGap;
        }
    }

    ofs << std::fixed << std::setprecision(9);
    ofs << "[Selected Cloud Correction Summary]" << std::endl;
    ofs << "merge_index = " << mergeIndex << std::endl;
    ofs << "selected_mode = " << GetCloudMapCorrectionModeName(selectedMode) << std::endl;
    ofs << "legacy_se3_weight_mode = ";
    if (selectedMode == CloudMapCorrectionMode::SE3_TIME_LINEAR_TWO_ANCHOR) {
        ofs << "TIME_LINEAR";
    } else {
        ofs << "not_used";
    }
    ofs << std::endl;
    ofs << "num_cloud_keyframes = " << vSnapshots.size() << std::endl;
    ofs << "num_sim3_pairs = " << sim3Result.numPairs << std::endl;
    ofs << "sim3_scale = " << sim3Result.scale << std::endl;
    ofs << "sim3_rmse_before = " << sim3Result.rmseBefore << std::endl;
    ofs << "sim3_rmse_after = " << sim3Result.rmseAfter << std::endl;
    if (!sim3Result.failureReason.empty()) {
        ofs << "sim3_failure_reason = " << sim3Result.failureReason << std::endl;
    }
    ofs << std::endl;

    ofs << "[Boundary Gap]" << std::endl;
    ofs << "selected_start_to_edge_front_gap = " << selectedStartToEdgeFrontGap << std::endl;
    ofs << "selected_end_to_edge_back_gap = " << selectedEndToEdgeBackGap << std::endl;
    ofs << "head_end_to_edge_back_gap = " << headEndToEdgeBackGap << std::endl;
    WriteSelectedCorrectionDoubleOrText(ofs, "two_anchor_end_to_edge_back_gap", twoAnchorEndToEdgeBackGap, bTwoAnchorExecuted);
    ofs << std::endl;

    ofs << "[Path Length]" << std::endl;
    ofs << "raw_path_length = " << rawPathLength << std::endl;
    ofs << "head_path_length = " << headPathLength << std::endl;
    ofs << "sim3_path_length = " << sim3PathLength << std::endl;
    ofs << "selected_path_length = " << selectedPathLength << std::endl;
    WriteSelectedCorrectionDoubleOrText(ofs, "two_anchor_path_length", twoAnchorPathLength, bTwoAnchorExecuted);
    ofs << std::endl;

    ofs << "[Z Statistics]" << std::endl;
    ofs << "raw_z_mean = " << rawZMean << std::endl;
    ofs << "head_z_mean = " << headZMean << std::endl;
    ofs << "sim3_z_mean = " << sim3ZMean << std::endl;
    ofs << "selected_z_mean = " << selectedZMean << std::endl;
    WriteSelectedCorrectionDoubleOrText(ofs, "two_anchor_z_mean", selectedZMean, bTwoAnchorExecuted);
    ofs << std::endl;

    ofs << "[Interpretation Hint]" << std::endl;
    if (selectedMode == CloudMapCorrectionMode::SIM3_ONLY) {
        ofs << "SIM3_ONLY selected: CloudMap is aligned using scale-aware Sim(3). Two-anchor SE(3) residual interpolation is not applied." << std::endl;
    }

    if (std::isfinite(selectedStartToEdgeFrontGap)) {
        if (selectedStartToEdgeFrontGap > kCloudMergeSim3MaxBoundaryGap) {
            ofs << "WARNING: selected correction start boundary gap is larger than threshold." << std::endl;
        }
    }

    if (std::isfinite(selectedEndToEdgeBackGap)) {
        if (selectedEndToEdgeBackGap > kCloudMergeSim3MaxBoundaryGap) {
            ofs << "WARNING: selected correction end boundary gap is larger than threshold." << std::endl;
        }
    }

    ofs.close();
}


// [SE3基线] 旧 SE(3) 双锚点基线仅保留时间线性残差分摊。
double ComputeTimeLinearCorrectionWeight(const double timestamp, const double tStart, const double tEnd) {
    if (tEnd <= tStart) {
        return 0.0;
    }

    if (timestamp <= tStart) {
        return 0.0;
    }

    if (timestamp >= tEnd) {
        return 1.0;
    }

    double weight = (timestamp - tStart) / (tEnd - tStart);
    if (!std::isfinite(weight)) {
        return 0.0;
    }

    if (weight < 0.0) {
        return 0.0;
    }

    if (weight > 1.0) {
        return 1.0;
    }

    return weight;
}

// [SE3基线] 构造纯 timestamp 排序的 TIME_LINEAR 权重表。
bool BuildTimeLinearCorrectionWeights(
    const std::vector<KeyFrame *> &vCloudKeyFrames,
    const double tStart,
    const double tEnd,
    std::map<KeyFrame *, double> &keyFrameCorrectionWeight) {
    keyFrameCorrectionWeight.clear();

    std::vector<KeyFrame *> vSortedCloudKeyFrames;
    vSortedCloudKeyFrames.reserve(vCloudKeyFrames.size());
    for (size_t i = 0; i < vCloudKeyFrames.size(); i++) {
        KeyFrame *pKF = vCloudKeyFrames[i];
        if (!IsValidCloudMergeKeyFrame(pKF)) {
            continue;
        }

        vSortedCloudKeyFrames.push_back(pKF);
    }

    if (vSortedCloudKeyFrames.empty()) {
        return false;
    }

    std::sort(vSortedCloudKeyFrames.begin(), vSortedCloudKeyFrames.end(), [](KeyFrame *pLeft, KeyFrame *pRight) {
        return pLeft->mTimeStamp < pRight->mTimeStamp;
    });

    for (size_t i = 0; i < vSortedCloudKeyFrames.size(); i++) {
        KeyFrame *pKF = vSortedCloudKeyFrames[i];
        keyFrameCorrectionWeight[pKF] = ComputeTimeLinearCorrectionWeight(pKF->mTimeStamp, tStart, tEnd);
    }

    return true;
}

} // namespace

CloudMerging::CloudMerging(Atlas *pAtlas, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, const bool bFixScale, const bool bActiveLC, const bool bWork, const bool bMergeAnyway, MapDrawer *pMapDrawer, FrameDrawer *pFrameDrawer, const bool bOldUdf, const bool bNewUdf) :
    mbResetRequested(false), mbResetActiveMapRequested(false), mbFinishRequested(false), mbFinished(true), mpAtlas(pAtlas),
    mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpMatchedKF(NULL), mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
    mbStopGBA(false), mpThreadGBA(nullptr), mbFixScale(bFixScale), mnFullBAIdx(0), mnMergeNumCoincidences(0),
    mbMergeDetected(false), mnMergeNumNotFound(0), mbActiveCM(bActiveLC), mbMergeAnyway(bMergeAnyway), mbWork(bWork),
    mpMapDrawer(pMapDrawer), mpFrameDrawer(pFrameDrawer), mbOldUdf(bOldUdf), mbNewUdf(bNewUdf),
    mnCloudMergeDebugOutputIndex(0) {
    
    mnCovisibilityConsistencyTh = 3;
    mpLastCurrentKF = static_cast<KeyFrame *>(NULL);

    mstrFolderSubTraj = "SubTrajectories/";
    mnNumCorrection = 0;
    mnCorrectionGBA = 0;
}

void CloudMerging::SetTracker(Tracking *pTracker) {
    mpTracker = pTracker;
}

void CloudMerging::SetLocalMapper(LocalMapping *pLocalMapper) {
    mpLocalMapper = pLocalMapper;
}

void CloudMerging::SetLoopClosing(LoopClosing *pLoopClosing) {
    mpLoopClosing = pLoopClosing;
}

void CloudMerging::SetCloudMergeDebugOutputDir(const std::string &debugOutputDir) {
    // [CloudMap校正诊断] 保存结果目录，供 CloudMap 校正前后轨迹诊断输出使用。
    std::lock_guard<std::mutex> lock(mMutexCloudMergeDebugOutputDir);
    mCloudMergeDebugOutputDir = debugOutputDir;
}

void CloudMerging::Run(bool bOnline) { 
    mbFinished = false;

    // 限制最大的match KF num，防止Cloud SLAM运行时本地进行了Loop Closure的清空
    int nLimitMaxMatchKFNum = 50;

    while (1) {
        // ***************** 第一步，检查是否有CloudMap *****************
        if (CheckNewCloudMap(bOnline) && mbWork) { 
            static int nCloudMerge = 0;
            nCloudMerge++;

            mbRunning = true;
            // ***************** 第二步，准备数据，包括取出地图、匹配timestamp *****************
            {
                unique_lock<mutex> lock(mMutexCloudQueue);
                if (bOnline) {
                    mpCurrentCloudMap = mlpCloudMapQueue.front(); 
                    mlpCloudMapQueue.pop_front();
                    mpCurrentEdgeFrontMap = mpAtlas->GetSpecifyMap(mpCurrentCloudMap->edgeFrontMapMnId); 
                    
                    if (false) {
                        vector<MapPoint *> EdgeFrontMapPoint = mpCurrentEdgeFrontMap->GetAllMapPoints();
                        std::string file_name = "/home/birl/Udf-Edge/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_point/test.xyz";
                        std::ofstream f(file_name);
                        for (int n = 0; n < EdgeFrontMapPoint.size(); n++) {
                            Eigen::Vector3f mpWorldPos = EdgeFrontMapPoint[n]->GetWorldPos();
                            f << mpWorldPos[0] << " " << mpWorldPos[1] << " " << mpWorldPos[2] << std::endl;
                        }
                        f.close();
                    }
                    
                    mpCurrentEdgeBackMap = mpAtlas->GetSpecifyMap(mpCurrentCloudMap->edgeBackMapMnId); 
                } else {
                    mpCurrentCloudMap = mlpCloudMapQueue.front();
                    mlpCloudMapQueue.pop_front();
                    
                    mpCurrentEdgeFrontMap = mlpCloudMapQueue.front();
                    mlpCloudMapQueue.pop_front();
                    mpCurrentEdgeBackMap = mlpCloudMapQueue.front();
                    mlpCloudMapQueue.pop_front();

                    mpAtlas->InsertNewMap(mpCurrentCloudMap);
                    mpAtlas->InsertNewMap(mpCurrentEdgeBackMap);
                    mpAtlas->InsertNewMap(mpCurrentEdgeFrontMap);
                }
            }

            if (!mpCurrentEdgeFrontMap || !mpCurrentEdgeBackMap) {
                cerr << "\033[1;33m[Warning] can't find the refer submap! Maps likely culled due to edge-cloud delay. Aborting this merge.\033[0m" << endl;
                
                if (mpCurrentCloudMap) {
                    delete mpCurrentCloudMap;
                    mpCurrentCloudMap = nullptr;
                }
                mpCurrentEdgeFrontMap = nullptr;
                mpCurrentEdgeBackMap = nullptr;
                mbRunning = false;
                
                mpEdgeFrontCloudKeyFrameMatch.clear();
                mvpEdgeFrontCloudMatchedKeyPoints.clear();
                mgSwEdgeFrontCloud = g2o::Sim3();
                mpNewEdgeFrontEdgeBackKeyFrameMatch.clear();
                mvpNewEdgeFrontEdgeBackMatchedKeyPoints.clear();
                mgSwNewCloudEdgeBack = g2o::Sim3();
                
                continue; 
            }

            // 【核心修复 1：弹性最近邻锚点匹配】
            double matchToleranceTime = 0.05; 
            const vector<KeyFrame *> &edgeFrontMapKeyFrames = mpCurrentEdgeFrontMap->GetAllKeyFrames();
            const vector<KeyFrame *> &cloudMapKeyFrames = mpCurrentCloudMap->GetAllKeyFrames();
            vector<bool> edgeFrontHaveMatchFlag(cloudMapKeyFrames.size(), false); 

            for (unsigned long edgeFrontKF_i = 0; edgeFrontKF_i < edgeFrontMapKeyFrames.size(); edgeFrontKF_i++) { 
                double minDelta = 1e9;
                int bestCloudIdx = -1;

                for (unsigned long cloudKF_i = 0; cloudKF_i < cloudMapKeyFrames.size(); cloudKF_i++) {
                    if (edgeFrontHaveMatchFlag[cloudKF_i]) { 
                        continue; 
                    }
                    
                    double deltaTime = abs(edgeFrontMapKeyFrames[edgeFrontKF_i]->mTimeStamp - cloudMapKeyFrames[cloudKF_i]->mTimeStamp);
                    
                    if (deltaTime < matchToleranceTime && deltaTime < minDelta) {
                        minDelta = deltaTime;
                        bestCloudIdx = cloudKF_i;
                    }
                }
                
                if (bestCloudIdx != -1) {
                    mpEdgeFrontCloudKeyFrameMatch[edgeFrontKF_i] = bestCloudIdx;
                    edgeFrontHaveMatchFlag[bestCloudIdx] = true;
                }
            }

            if (mpEdgeFrontCloudKeyFrameMatch.size() < 1) {
                cerr << "\033[1;31m匹配EdgeFront和Cloud Map的KeyFrames数量过少\033[0m" << endl;
                
                if (mpCurrentCloudMap) {
                    delete mpCurrentCloudMap;
                    mpCurrentCloudMap = nullptr;
                }
                mpCurrentEdgeFrontMap = nullptr;
                mpCurrentEdgeBackMap = nullptr;
                mbRunning = false;
                continue; 
            }

            // [Sim3合并实验] SIM3_ONLY 失败时不自动 fallback 到旧 two-anchor 逻辑，避免实验结果混淆。
            bool bSkipCloudMergeDueToCorrectionFailure = false;

            if (mbOldUdf || mbNewUdf) {
                if (mpEdgeFrontCloudKeyFrameMatch.size() > 0) {
                    
                    // =================================================================================
                    // 【阶段 1：确立前端定子，计算宏观搬运矩阵 S_head】
                    // 物理意义：CloudMap 的起点坐标是云端任意初始化的，必须计算一个基础变换矩阵，
                    // 将其整个坐标系“吸附”到 EdgeFront 的尾部（第一层红圈对齐）。
                    // =================================================================================
                    KeyFrame* pKF_CloudStart = nullptr;
                    KeyFrame* pKF_EdgeFrontEnd = nullptr;
                    double minDeltaFront = 1e9;

                    for (auto iter : mpEdgeFrontCloudKeyFrameMatch) {
                        KeyFrame* kfFront = edgeFrontMapKeyFrames[iter.first];
                        KeyFrame* kfCloud = cloudMapKeyFrames[iter.second];
                        double dt = std::abs(kfFront->mTimeStamp - kfCloud->mTimeStamp);
                        if (dt < minDeltaFront) {
                            minDeltaFront = dt;
                            pKF_EdgeFrontEnd = kfFront;
                            pKF_CloudStart = kfCloud;
                        }
                    }

                    if (pKF_EdgeFrontEnd != nullptr) {
                        if (pKF_CloudStart != nullptr) {
                            double t_start = pKF_CloudStart->mTimeStamp;

                            Sophus::SE3f T_ef_Twc = pKF_EdgeFrontEnd->GetPoseInverse();
                            Sophus::SE3f T_cloud_head_Twc = pKF_CloudStart->GetPoseInverse();

                            Sophus::Sim3d Sim3_ef(T_ef_Twc.unit_quaternion().cast<double>(), T_ef_Twc.translation().cast<double>());
                            Sim3_ef.setScale(1.0);
                            Sophus::Sim3d Sim3_cloud_head(T_cloud_head_Twc.unit_quaternion().cast<double>(), T_cloud_head_Twc.translation().cast<double>());
                            Sim3_cloud_head.setScale(1.0);

                            // S_head = 目标坐标 * 当前坐标^(-1)
                            Sophus::Sim3d S_head = Sim3_ef * Sim3_cloud_head.inverse();

                            // =================================================================================
                            // 【阶段 2：搜索后端定子，计算绝对位姿残差 Delta_T】
                            // =================================================================================
                            const vector<KeyFrame *> &edgeBackMapKeyFrames = mpCurrentEdgeBackMap->GetAllKeyFrames();
                            int bestCloudTailIdx = -1;
                            int bestEdgeBackHeadIdx = -1;
                            double minDeltaBack = 1e9;
                            double matchToleranceTimeBack = 0.05; 

                            for (unsigned long cloud_i = 0; cloud_i < cloudMapKeyFrames.size(); cloud_i++) {
                                for (unsigned long back_i = 0; back_i < edgeBackMapKeyFrames.size(); back_i++) {
                                    double deltaT = std::abs(cloudMapKeyFrames[cloud_i]->mTimeStamp - edgeBackMapKeyFrames[back_i]->mTimeStamp);
                                    if (deltaT < matchToleranceTimeBack) {
                                        if (deltaT < minDeltaBack) {
                                            minDeltaBack = deltaT;
                                            bestCloudTailIdx = cloud_i;
                                            bestEdgeBackHeadIdx = back_i;
                                        }
                                    }
                                }
                            }

                            if (bestCloudTailIdx != -1) {
                                if (bestEdgeBackHeadIdx != -1) {
                                    KeyFrame* pKF_CloudTail = cloudMapKeyFrames[bestCloudTailIdx];
                                    KeyFrame* pKF_EdgeBackHead = edgeBackMapKeyFrames[bestEdgeBackHeadIdx];
                                    double t_end = pKF_CloudTail->mTimeStamp;

                                    if (t_end > t_start) {
                                        Sophus::SE3f T_eb_Twc = pKF_EdgeBackHead->GetPoseInverse();
                                        Sophus::SE3f T_cloud_tail_Twc = pKF_CloudTail->GetPoseInverse();

                                        Sophus::Sim3d Sim3_eb(T_eb_Twc.unit_quaternion().cast<double>(), T_eb_Twc.translation().cast<double>());
                                        Sim3_eb.setScale(1.0);
                                        Sophus::Sim3d Sim3_cloud_tail(T_cloud_tail_Twc.unit_quaternion().cast<double>(), T_cloud_tail_Twc.translation().cast<double>());
                                        Sim3_cloud_tail.setScale(1.0);

                                        // 【核心修正】：计算残差前，必须先让尾部套用 S_head 基准搬运矩阵！
                                        // 理论位置 = 基准搬运 * 原始坐标
                                        Sophus::Sim3d Sim3_cloud_tail_rigid = S_head * Sim3_cloud_tail;

                                        // 纯残差 = 真值目标(EdgeBack) * 理论位置^(-1)
                                        Sophus::Sim3d delta_T = Sim3_eb * Sim3_cloud_tail_rigid.inverse();
                                        Eigen::Matrix<double, 7, 1> xi = delta_T.log();

                                        // [CloudMap校正诊断] 在真正 SetPose 前缓存 raw 与 head-aligned-only 虚拟轨迹。
                                        std::string cloudCorrectionDebugOutputDir;
                                        int cloudCorrectionDebugMergeIndex = 0;
                                        {
                                            std::lock_guard<std::mutex> lock(mMutexCloudMergeDebugOutputDir);
                                            cloudCorrectionDebugOutputDir = mCloudMergeDebugOutputDir;
                                            if (!cloudCorrectionDebugOutputDir.empty()) {
                                                mnCloudMergeDebugOutputIndex++;
                                                cloudCorrectionDebugMergeIndex = mnCloudMergeDebugOutputIndex;
                                            }
                                        }

                                        CloudCorrectionDebugPoseVector cloudCorrectionDebugSnapshots;
                                        CloudSim3AlignmentPairVector cloudSim3AlignmentPairs;
                                        CloudSim3AlignmentResult cloudSim3AlignmentResult =
                                            MakeCloudSim3AlignmentResult("not_requested");
                                        std::string cloudCorrectionDebugPrefix;
                                        if (!cloudCorrectionDebugOutputDir.empty()) {
                                            std::vector<KeyFrame *> vSortedCloudDebugKeyFrames =
                                                BuildSortedCloudDebugKeyFrames(cloudMapKeyFrames);
                                            cloudCorrectionDebugSnapshots =
                                                BuildCloudCorrectionDebugSnapshots(vSortedCloudDebugKeyFrames, S_head);
                                            cloudSim3AlignmentPairs =
                                                BuildCloudSim3AlignmentPairs(cloudCorrectionDebugSnapshots);
                                            cloudSim3AlignmentResult =
                                                EstimateCloudSim3Alignment(cloudSim3AlignmentPairs);
                                            FillCloudSim3AlignedDebugPose(
                                                cloudCorrectionDebugSnapshots,
                                                cloudSim3AlignmentResult);
                                            cloudCorrectionDebugPrefix =
                                                BuildCloudCorrectionDebugPrefix(cloudCorrectionDebugOutputDir, cloudCorrectionDebugMergeIndex);
                                        }

                                        CloudMergeSim3Result cloudMergeSim3Result = MakeCloudMergeSim3Result();
                                        bool bTwoAnchorExecuted = false;
                                        bool bSelectedCorrectionApplied = false;

                                        if (kCloudMapCorrectionMode == CloudMapCorrectionMode::SIM3_ONLY) {
                                            // [Sim3合并实验] 使用 CloudMap/frontend 对应点估计 Sim(3)，并直接写回 CloudMap。
                                            std::vector<double> matchedTimestamps;
                                            std::vector<unsigned long> matchedKeyFrameIds;
                                            std::vector<Eigen::Vector3f> cloudPoints;
                                            std::vector<Eigen::Vector3f> frontendPoints;
                                            std::vector<double> timeGaps;

                                            const bool bSim3Ok = EstimateCloudToFrontendSim3(
                                                cloudMapKeyFrames,
                                                cloudMergeSim3Result,
                                                matchedTimestamps,
                                                matchedKeyFrameIds,
                                                cloudPoints,
                                                frontendPoints,
                                                timeGaps);

                                            if (bSim3Ok) {
                                                const bool bApplyOk = ApplySim3ToCloudMap(
                                                    mpCurrentCloudMap,
                                                    cloudMapKeyFrames,
                                                    cloudMergeSim3Result);

                                                if (bApplyOk) {
                                                    bSelectedCorrectionApplied = true;

                                                    std::cout << "[CloudMerging][Sim3] SIM3_ONLY correction applied. scale = "
                                                              << cloudMergeSim3Result.scale
                                                              << ", pairs = "
                                                              << cloudMergeSim3Result.numPairs
                                                              << ", rmse before = "
                                                              << cloudMergeSim3Result.rmseBefore
                                                              << ", rmse after = "
                                                              << cloudMergeSim3Result.rmseAfter
                                                              << std::endl;

                                                    const Sophus::SE3f TwcCloudStartAfter = pKF_CloudStart->GetPoseInverse();
                                                    const Sophus::SE3f TwcCloudTailAfter = pKF_CloudTail->GetPoseInverse();
                                                    const double sim3StartToEdgeFrontGap =
                                                        ComputeCloudCorrectionDebugGap(TwcCloudStartAfter, pKF_EdgeFrontEnd->GetPoseInverse());
                                                    const double sim3EndToEdgeBackGap =
                                                        ComputeCloudCorrectionDebugGap(TwcCloudTailAfter, pKF_EdgeBackHead->GetPoseInverse());

                                                    if (sim3StartToEdgeFrontGap > kCloudMergeSim3MaxBoundaryGap) {
                                                        std::cout << "[CloudMerging][Sim3][WARNING] sim3_start_to_edge_front_gap = "
                                                                  << sim3StartToEdgeFrontGap
                                                                  << " exceeds threshold "
                                                                  << kCloudMergeSim3MaxBoundaryGap
                                                                  << std::endl;
                                                    }

                                                    if (sim3EndToEdgeBackGap > kCloudMergeSim3MaxBoundaryGap) {
                                                        std::cout << "[CloudMerging][Sim3][WARNING] sim3_end_to_edge_back_gap = "
                                                                  << sim3EndToEdgeBackGap
                                                                  << " exceeds threshold "
                                                                  << kCloudMergeSim3MaxBoundaryGap
                                                                  << std::endl;
                                                    }
                                                } else {
                                                    cloudMergeSim3Result.failureReason = "apply_failed";
                                                }
                                            }

                                            if (!bSelectedCorrectionApplied) {
                                                std::cout << "[CloudMerging][Sim3][ERROR] SIM3_ONLY correction failed: "
                                                          << cloudMergeSim3Result.failureReason
                                                          << std::endl;
                                                bSkipCloudMergeDueToCorrectionFailure = true;
                                            }
                                        } else if (kCloudMapCorrectionMode == CloudMapCorrectionMode::SE3_TIME_LINEAR_TWO_ANCHOR) {
                                            // [SE3基线] 保留旧逻辑：严格双锚点 + TIME-LINEAR 残差插值。
                                            std::map<KeyFrame *, double> cloudKeyFrameCorrectionWeight;
                                            const bool bBuildCorrectionWeight = BuildTimeLinearCorrectionWeights(
                                                cloudMapKeyFrames,
                                                t_start,
                                                t_end,
                                                cloudKeyFrameCorrectionWeight);

                                            cerr << "\033[1;32m[CloudMerging] Legacy SE3 two-anchor correction uses TIME-LINEAR weights.\033[0m" << endl;

                                            if (!bBuildCorrectionWeight) {
                                                cerr << "\033[1;33m[CloudMerging] UDF correction weight table is empty; per-frame fallback remains time-linear.\033[0m" << endl;
                                            }

                                            // [SE3基线] 查询单个 KeyFrame 的 TIME_LINEAR 权重；缺失时按 timestamp 计算兜底。
                                            auto getCloudCorrectionWeight = [&](KeyFrame *pKF) -> double {
                                                if (pKF == nullptr) {
                                                    return 0.0;
                                                }

                                                auto weightIter = cloudKeyFrameCorrectionWeight.find(pKF);
                                                if (weightIter != cloudKeyFrameCorrectionWeight.end()) {
                                                    return weightIter->second;
                                                }

                                                return ComputeTimeLinearCorrectionWeight(pKF->mTimeStamp, t_start, t_end);
                                            };

                                            // =================================================================================
                                            // 【阶段 3：劫持 CloudMap 关键帧，执行 双锚点复合扭曲】
                                            // =================================================================================
                                            for (size_t i = 0; i < cloudMapKeyFrames.size(); i++) {
                                                KeyFrame* pKFi = cloudMapKeyFrames[i];
                                                if (IsValidCloudMergeKeyFrame(pKFi)) {
                                                    // [SE3基线] CloudMap KeyFrame 使用 TIME_LINEAR 补偿权重。
                                                    const double w_i = getCloudCorrectionWeight(pKFi);

                                                    Eigen::Matrix<double, 7, 1> xi_scaled = xi * w_i;
                                                    Sophus::Sim3d T_scale_i = Sophus::Sim3d::exp(xi_scaled);

                                                    Sophus::SE3f T_cloud_i_Twc = pKFi->GetPoseInverse();
                                                    Sophus::Sim3d Sim3_cloud_i(T_cloud_i_Twc.unit_quaternion().cast<double>(), T_cloud_i_Twc.translation().cast<double>());
                                                    Sim3_cloud_i.setScale(1.0);

                                                    // 【终极方程】：纠正后坐标 = 伸缩扭曲(T_scale_i) * 基准搬运(S_head) * 原始坐标(Sim3_cloud_i)
                                                    Sophus::Sim3d Sim3_corrected = T_scale_i * S_head * Sim3_cloud_i;

                                                    Sophus::SE3f T_corrected_Twc(Sim3_corrected.rotationMatrix().cast<float>(), Sim3_corrected.translation().cast<float>());
                                                    pKFi->SetPose(T_corrected_Twc.inverse());
                                                }
                                            }

                                            // =================================================================================
                                            // 【阶段 4：劫持 UDF 点云，套用完全相同的双锚点方程】
                                            // =================================================================================
                                            const vector<MapPoint*> &cloudMPs = mpCurrentCloudMap->GetAllMapPoints();
                                            for (size_t j = 0; j < cloudMPs.size(); j++) {
                                                MapPoint* pMP = cloudMPs[j];
                                                if (pMP) {
                                                    if (!pMP->isBad()) {
                                                        KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
                                                        if (IsValidCloudMergeKeyFrame(pRefKF)) {
                                                            // [SE3基线] MapPoint / UDF 点沿用参考 KeyFrame 的 TIME_LINEAR 补偿权重。
                                                            const double w_i = getCloudCorrectionWeight(pRefKF);

                                                            Eigen::Matrix<double, 7, 1> xi_scaled = xi * w_i;
                                                            Sophus::Sim3d T_scale_i = Sophus::Sim3d::exp(xi_scaled);

                                                            Eigen::Vector3d P3Dw = pMP->GetWorldPos().cast<double>();

                                                            // 同样必须先进行 S_head 搬运，再进行拉伸扭曲
                                                            Eigen::Vector3d P3D_corrected = T_scale_i * S_head * P3Dw;

                                                            pMP->SetWorldPos(P3D_corrected.cast<float>());
                                                            pMP->UpdateNormalAndDepth();
                                                        }
                                                    }
                                                }
                                            }
                                            bTwoAnchorExecuted = true;
                                            bSelectedCorrectionApplied = true;
                                            cerr << "\033[1;32m[CloudMerging] Sim3 Lie-Algebra Manifold Smoothing (With Base Alignment) Successfully Applied!\033[0m" << endl;
                                        } else if (kCloudMapCorrectionMode == CloudMapCorrectionMode::SIM3_PLUS_RESIDUAL_RESERVED) {
                                            // [Sim3合并实验] 预留模式暂不实现，避免与 SIM3_ONLY 实验结果混淆。
                                            cloudMergeSim3Result.failureReason = "sim3_plus_residual_reserved";
                                            bSkipCloudMergeDueToCorrectionFailure = true;
                                            std::cout << "[CloudMerging][Sim3][ERROR] SIM3_PLUS_RESIDUAL_RESERVED is not implemented." << std::endl;
                                        } else {
                                            // [Sim3合并实验] 未知模式不自动 fallback。
                                            cloudMergeSim3Result.failureReason = "unknown_correction_mode";
                                            bSkipCloudMergeDueToCorrectionFailure = true;
                                            std::cout << "[CloudMerging][Sim3][ERROR] Unknown CloudMap correction mode." << std::endl;
                                        }

	                                        if (!cloudCorrectionDebugOutputDir.empty()) {
	                                            // [CloudMap校正诊断] 读取校正后的实际 KeyFrame 位姿并输出保留的统计 summary，不再写中间 TUM/CSV 文件。
	                                            FillCloudCorrectionDebugAfterPose(cloudCorrectionDebugSnapshots);
	                                            WriteCloudSim3AlignmentDebugSummary(
	                                                cloudCorrectionDebugPrefix + "_sim3_alignment_debug_summary.txt",
	                                                cloudCorrectionDebugMergeIndex,
	                                                cloudCorrectionDebugSnapshots,
                                                cloudSim3AlignmentPairs,
                                                cloudSim3AlignmentResult,
                                                pKF_EdgeFrontEnd,
                                                pKF_EdgeBackHead);
                                            WriteCloudSelectedCorrectionSummary(
                                                cloudCorrectionDebugPrefix + "_selected_correction_summary.txt",
                                                cloudCorrectionDebugMergeIndex,
                                                kCloudMapCorrectionMode,
                                                cloudCorrectionDebugSnapshots,
                                                cloudMergeSim3Result,
                                                pKF_EdgeFrontEnd,
	                                                pKF_EdgeBackHead,
	                                                bTwoAnchorExecuted);

	                                            cerr << "\033[1;32m[CloudMap Correction Debug] Saved merge summaries: "
	                                                 << cloudCorrectionDebugPrefix
	                                                 << "_selected_correction_summary.txt, "
	                                                 << cloudCorrectionDebugPrefix
	                                                 << "_sim3_alignment_debug_summary.txt\033[0m" << endl;
	                                        }
                                    } else {
                                        cerr << "\033[1;33m[CloudMerging] Warning: Time inversion detected, bypass smoothing.\033[0m" << endl;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // ***************** 补偿完毕，向下传递给原有串行合并流程 *****************


            // ***************** 第三步，对齐Cloud Map和Ground Map，计算Sim3 *****************
            std::vector<int> vEdgeFrontMatchIndexes;
            for (auto &iter : mpEdgeFrontCloudKeyFrameMatch) {
                vEdgeFrontMatchIndexes.push_back(iter.first);
            }
            sort(vEdgeFrontMatchIndexes.begin(), vEdgeFrontMatchIndexes.end(), [&](int x, int y) { 
                return edgeFrontMapKeyFrames[x]->mTimeStamp < edgeFrontMapKeyFrames[y]->mTimeStamp; 
            });
            
            std::vector<int> vEdgeFrontMatchPartIndexes(vEdgeFrontMatchIndexes.begin(), vEdgeFrontMatchIndexes.begin() + min(int(mpEdgeFrontCloudKeyFrameMatch.size()), nLimitMaxMatchKFNum));
            std::map<int, int> mEdgeFrontCloudKeyFrameRandSelectMatch;
            for (auto &iter : vEdgeFrontMatchPartIndexes) {
                mEdgeFrontCloudKeyFrameRandSelectMatch[iter] = mpEdgeFrontCloudKeyFrameMatch[iter];
            }

            // cerr << "First Merge EdgeFront Map KF num: " << mpCurrentEdgeFrontMap->GetAllKeyFrames().size() << endl;
            // cerr << "First Merge Cloud Map KF num: " << mpCurrentCloudMap->GetAllKeyFrames().size() << endl;
            // cerr << "First Merge Rand Select KF ratio: " << mEdgeFrontCloudKeyFrameRandSelectMatch.size() << " / " << mpEdgeFrontCloudKeyFrameMatch.size() << endl;
            
            // bool bComputeEdgeFront = CloudMerging::ComputeSubmapSim3(mpCurrentEdgeFrontMap, mpCurrentCloudMap, mpEdgeFrontCloudKeyFrameMatch, mEdgeFrontCloudKeyFrameRandSelectMatch, false, mgSwEdgeFrontCloud, mvpEdgeFrontCloudMatchedKeyPoints, mpMapDrawer, mbOldUdf, mbNewUdf, false);
            bool bComputeEdgeFront = false;

            if (bSkipCloudMergeDueToCorrectionFailure) {
                std::cout << "[CloudMerging][Sim3][ERROR] Skip CloudMergeMap because selected CloudMap correction failed." << std::endl;
            } else if (mbOldUdf || mbNewUdf) {
                // 【核心修复：绝对单位阵注入，锁定平滑后的 CloudMap】
                // 此时 CloudMap 已经在内存中通过 Sim(3) 插值补偿被完美掰弯，首尾已经贴合红圈。
                // 注入单位阵，防止原生的 ComputeSubmapSim3 因为伪造特征点产生错误的拉扯。
                mgSwEdgeFrontCloud = g2o::Sim3(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), 1.0);
                bComputeEdgeFront = true;
                cerr << "\033[1;32m[CloudMerging] Bypass 1st ComputeSubmapSim3. Injected Identity Sim3 to PROTECT CloudMap!\033[0m" << endl;
            } else {
                bComputeEdgeFront = CloudMerging::ComputeSubmapSim3(mpCurrentEdgeFrontMap, mpCurrentCloudMap, mpEdgeFrontCloudKeyFrameMatch, mEdgeFrontCloudKeyFrameRandSelectMatch, false, mgSwEdgeFrontCloud, mvpEdgeFrontCloudMatchedKeyPoints, mpMapDrawer, mbOldUdf, mbNewUdf, false);
            }

            // ***************** 第四步，Merge Map *****************
            if (!bSkipCloudMergeDueToCorrectionFailure && (bComputeEdgeFront || mbMergeAnyway)) {
                Verbose::PrintMess("*Merge detected", Verbose::VERBOSITY_QUIET);

                bool bRelaunchBA = false;

                if (isRunningGBA()) {
                    unique_lock<mutex> lock(mMutexGBA);
                    mbStopGBA = true;
                    mnFullBAIdx++;

                    if (mpThreadGBA) {
                        mpThreadGBA->detach();
                        delete mpThreadGBA;
                    }
                    bRelaunchBA = true;
                }

                // Merge !!!
                CloudMergeMap(mpCurrentEdgeFrontMap, mpCurrentCloudMap, mgSwEdgeFrontCloud, mpEdgeFrontCloudKeyFrameMatch, mvpEdgeFrontCloudMatchedKeyPoints, mpLocalMapper, true, mbOldUdf, mbNewUdf); 

                mpAtlas->RemoveBadMaps();

                if (bRelaunchBA) {
                    mbRunningGBA = false;
                    mbFinishedGBA = true;
                    mbStopGBA = false;
                    mpThreadGBA = new thread(&CloudMerging::RunGlobalBundleAdjustment, this, mpCurrentEdgeFrontMap, mpCurrentEdgeFrontMap->GetOriginKF()->mnId);
                }
                Verbose::PrintMess("Merge finished!", Verbose::VERBOSITY_QUIET);
                
                // ***************** 开始edgeback部分 *****************
                const vector<KeyFrame *> &newEdgeFrontMapKeyFrames = mpCurrentEdgeFrontMap->GetAllKeyFrames();
                const vector<KeyFrame *> &edgeBackMapKeyFrames = mpCurrentEdgeBackMap->GetAllKeyFrames();
                std::vector<bool> edgeBackHaveMatchFlag(edgeBackMapKeyFrames.size(), false);
                
                // 【核心修复 2：后段地图的弹性最近邻匹配】
                double matchToleranceTimeBack = 0.05; 

                for (unsigned long newEdgeFrontKF_i = 0; newEdgeFrontKF_i < newEdgeFrontMapKeyFrames.size(); newEdgeFrontKF_i++) {
                    double minDelta = 1e9;
                    int bestBackIdx = -1;

                    for (unsigned long edgeBackKF_i = 0; edgeBackKF_i < edgeBackMapKeyFrames.size(); edgeBackKF_i++) {
                        if (edgeBackHaveMatchFlag[edgeBackKF_i]) { 
                            continue; 
                        }
                        
                        double deltaTime = abs(edgeBackMapKeyFrames[edgeBackKF_i]->mTimeStamp - newEdgeFrontMapKeyFrames[newEdgeFrontKF_i]->mTimeStamp);
                        
                        if (deltaTime < matchToleranceTimeBack && deltaTime < minDelta) {
                            minDelta = deltaTime;
                            bestBackIdx = edgeBackKF_i;
                        }
                    }
                    
                    if (bestBackIdx != -1) {
                        mpNewEdgeFrontEdgeBackKeyFrameMatch[newEdgeFrontKF_i] = bestBackIdx;
                        edgeBackHaveMatchFlag[bestBackIdx] = true;
                    }
                }

                if (mpNewEdgeFrontEdgeBackKeyFrameMatch.size() < 1) {
                    cerr << "\033[1;31m匹配EdgeBack和New Cloud Map的KeyFrames数量过少\033[0m" << endl;
                    
                    mpCurrentEdgeBackMap->ResetHaveMerged();
                    mbRunning = false;
                    
                    if (mpCurrentCloudMap) {
                        delete mpCurrentCloudMap;
                        mpCurrentCloudMap = nullptr;
                    }
                    mpCurrentEdgeFrontMap = nullptr;
                    mpCurrentEdgeBackMap = nullptr;
                    
                    mpEdgeFrontCloudKeyFrameMatch.clear();
                    mvpEdgeFrontCloudMatchedKeyPoints.clear();
                    mgSwEdgeFrontCloud = g2o::Sim3();
                    mpNewEdgeFrontEdgeBackKeyFrameMatch.clear();
                    mvpNewEdgeFrontEdgeBackMatchedKeyPoints.clear();
                    mgSwNewCloudEdgeBack = g2o::Sim3();
                    continue;
                }

                std::vector<int> vNewEdgeFrontMatchIndexes;
                for (auto &iter : mpNewEdgeFrontEdgeBackKeyFrameMatch) {
                    vNewEdgeFrontMatchIndexes.push_back(iter.first);
                }
                sort(vNewEdgeFrontMatchIndexes.begin(), vNewEdgeFrontMatchIndexes.end(), [&](int x, int y) { 
                    return newEdgeFrontMapKeyFrames[x]->mTimeStamp < newEdgeFrontMapKeyFrames[y]->mTimeStamp; 
                });
                
                std::vector<int> vNewEdgeFrontMatchPartIndexes(vNewEdgeFrontMatchIndexes.end() - min(int(vNewEdgeFrontMatchIndexes.size()), nLimitMaxMatchKFNum), vNewEdgeFrontMatchIndexes.end());
                std::map<int, int> mNewEdgeFrontEdgeBackKeyFrameRandSelectMatch;
                for (auto &iter : vNewEdgeFrontMatchPartIndexes) {
                    mNewEdgeFrontEdgeBackKeyFrameRandSelectMatch[iter] = mpNewEdgeFrontEdgeBackKeyFrameMatch[iter];
                }

                bool bComputeEdgeBack = false;

                if (mbOldUdf || mbNewUdf) {
                    // 【核心修复：绝对单位阵注入，物理锚定 EdgeBack】
                    // 物理前提：OKVIS 维持了全局连续的度量空间，EdgeFront 和 EdgeBack 已经处于同一个完美的物理坐标系下。
                    // 任何非单位阵的 Sim(3) 变换都会导致 EdgeBack 被错误地拉扯产生位移（变成图中黑色的错误轨迹）。
                    // 因此，我们强制注入单位阵，让后续的 CloudMergeMap 仅仅执行“内存指针合并”，绝不修改任何 XYZ 坐标。
                    mgSwNewCloudEdgeBack = g2o::Sim3(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), 1.0);
                    bComputeEdgeBack = true;
                    cerr << "\033[1;32m[CloudMerging] Bypass 2nd ComputeSubmapSim3. Injected Identity Sim3 to PROTECT EdgeBack!\033[0m" << endl;
                } else {
                    // 只有在非 UDF (纯视觉特征) 模式下，才走原生的特征点匹配求解逻辑
                    bool mbedge_merge = false;
                    bComputeEdgeBack = CloudMerging::ComputeSubmapSim3(mpCurrentEdgeFrontMap, mpCurrentEdgeBackMap, mpNewEdgeFrontEdgeBackKeyFrameMatch, mNewEdgeFrontEdgeBackKeyFrameRandSelectMatch, false, mgSwNewCloudEdgeBack, mvpNewEdgeFrontEdgeBackMatchedKeyPoints, mpMapDrawer, false, false, mbedge_merge);
                }

                // std::vector<int> vNewEdgeFrontMatchPartIndexes(vNewEdgeFrontMatchIndexes.end() - min(int(vNewEdgeFrontMatchIndexes.size()), nLimitMaxMatchKFNum), vNewEdgeFrontMatchIndexes.end());
                // std::map<int, int> mNewEdgeFrontEdgeBackKeyFrameRandSelectMatch;
                // for (auto &iter : vNewEdgeFrontMatchPartIndexes) {
                //     mNewEdgeFrontEdgeBackKeyFrameRandSelectMatch[iter] = mpNewEdgeFrontEdgeBackKeyFrameMatch[iter];
                // }

                // cerr << "Second Merge Main Map KF num: " << mpCurrentEdgeFrontMap->GetAllKeyFrames().size() << endl;
                // cerr << "Second Merge EdgeBack Map KF num: " << mpCurrentEdgeBackMap->GetAllKeyFrames().size() << endl;
                // cerr << "Second Merge Rand Select KF ratio: " << mNewEdgeFrontEdgeBackKeyFrameRandSelectMatch.size() << " / " << mpNewEdgeFrontEdgeBackKeyFrameMatch.size() << endl;
                
                // bool mbedge_merge = mbOldUdf || mbNewUdf;
                // bool bComputeEdgeBack = CloudMerging::ComputeSubmapSim3(mpCurrentEdgeFrontMap, mpCurrentEdgeBackMap, mpNewEdgeFrontEdgeBackKeyFrameMatch, mNewEdgeFrontEdgeBackKeyFrameRandSelectMatch, false, mgSwNewCloudEdgeBack, mvpNewEdgeFrontEdgeBackMatchedKeyPoints, mpMapDrawer, false, false, mbedge_merge);
                
                if (bComputeEdgeBack || mbMergeAnyway) {
                    Verbose::PrintMess("*Merge detected", Verbose::VERBOSITY_QUIET);

                    bool bRelaunchBA = false;

                    if (isRunningGBA()) {
                        unique_lock<mutex> lock(mMutexGBA);
                        mbStopGBA = true;
                        mnFullBAIdx++;

                        if (mpThreadGBA) {
                            mpThreadGBA->detach();
                            delete mpThreadGBA;
                        }
                        bRelaunchBA = true;
                    }

                    // Merge !!!
                    CloudMergeMap(mpCurrentEdgeFrontMap, mpCurrentEdgeBackMap, mgSwNewCloudEdgeBack, mpNewEdgeFrontEdgeBackKeyFrameMatch, mvpNewEdgeFrontEdgeBackMatchedKeyPoints, mpLocalMapper, true, mbOldUdf, mbNewUdf); 

                    delete mpCurrentCloudMap; 
                    
                    if (mpAtlas->GetCurrentMap() == mpCurrentEdgeBackMap) {
                        mpAtlas->ChangeMap(mpCurrentEdgeFrontMap);
                    }
                    mpCurrentEdgeFrontMap->ChangeId(mpCurrentEdgeBackMap->GetId());
                    mpCurrentEdgeFrontMap->ResetHaveMerged();

                    mpAtlas->SetMapBad(mpCurrentEdgeBackMap);
                    mpAtlas->RemoveBadMaps();

                    if (bRelaunchBA) {
                        mbRunningGBA = false;
                        mbFinishedGBA = true;
                        mbStopGBA = false;
                        mpThreadGBA = new thread(&CloudMerging::RunGlobalBundleAdjustment, this, mpCurrentEdgeFrontMap, mpCurrentEdgeFrontMap->GetOriginKF()->mnId);
                    }
                    Verbose::PrintMess("Merge finished!", Verbose::VERBOSITY_QUIET);
                } else {
                    mpCurrentEdgeBackMap->ResetHaveMerged();
                }
            } else {
                mpCurrentEdgeBackMap->ResetHaveMerged();
            }
            cerr << "测试：退出此次Merge" << endl;
            mbRunning = false;

            mpCurrentCloudMap = static_cast<Map *>(NULL);
            mpCurrentEdgeFrontMap = static_cast<Map *>(NULL);
            mpCurrentEdgeBackMap = static_cast<Map *>(NULL);

            mpEdgeFrontCloudKeyFrameMatch.clear();
            mvpEdgeFrontCloudMatchedKeyPoints.clear();
            mgSwEdgeFrontCloud = g2o::Sim3();
            mpNewEdgeFrontEdgeBackKeyFrameMatch.clear();
            mvpNewEdgeFrontEdgeBackMatchedKeyPoints.clear();
            mgSwNewCloudEdgeBack = g2o::Sim3();

            cerr << "==================================" << endl;
            cerr << "CloudMerge Time: " << nCloudMerge << endl;
            cerr << "==================================" << endl;
        }

        ResetIfRequested();

        if (CheckFinish()) {
            break;
        }

        usleep(5000);
    }

    SetFinish();
}

void CloudMerging::InsertCloudMap(Map *pMap) {
    unique_lock<mutex> lock(mMutexCloudQueue);
    mlpCloudMapQueue.push_back(pMap);
}

bool CloudMerging::CheckNewCloudMap(bool bOnline) {
    unique_lock<mutex> lock(mMutexCloudQueue);

    if (bOnline) {
        return (!mlpCloudMapQueue.empty());
    } else {
        return (mlpCloudMapQueue.size() == 3);
    }
}

//cap-udf
void split(std::string& string_input, std::vector<std::string>&string_output, std::string& delema1) {
	std::string::size_type start = string_input.find_first_not_of(delema1, 0);
	std::string::size_type pose = string_input.find_first_of(delema1, start);
	while (std::string::npos != start || std::string::npos != pose) {
		string_output.push_back(string_input.substr(start, pose - start));
		start = string_input.find_first_not_of(delema1, pose);
		pose = string_input.find_first_of(delema1, start);
	}
}

bool get_parameter_xyz(std::string path, pcl::PointCloud<pcl::PointXYZ> &cloud) {
	cloud.clear();
	std::ifstream inf(path);
	std::string sline;
	std::vector<std::string>string_output;
	std::string Delema = " ";
	pcl::PointXYZ point3d;
	while (getline(inf, sline)) {
		int i = 0;
		split(sline, string_output, Delema);
		point3d.x = stold(string_output[i]);
		point3d.y = stold(string_output[i+1]);
		point3d.z = stold(string_output[i + 2]);
		string_output.clear();
		cloud.push_back(point3d);
	}
	inf.close();
    return true;
}

bool get_xyz(vector<MapPoint *> point_test, pcl::PointCloud<pcl::PointXYZ> &edge_point) {
    pcl::PointXYZ point_trans;
    for (int n = 0; n < point_test.size(); n++) {
        Eigen::Vector3f mpWorldPos = point_test[n]->GetWorldPos();
        point_trans.x = mpWorldPos[0];
        point_trans.y = mpWorldPos[1];
        point_trans.z = mpWorldPos[2];
        edge_point.push_back(point_trans);
    }
    return true;    
}

bool next_iteration = false;
//cap-udf

/*!
 * @param
 * @return
 */
bool CloudMerging::ComputeSubmapSim3(
    Map *pMap1, Map *pMap2, const std::map<int, int> &mKfMatch12,                  
    const std::map<int, int> &mRandSelectKfMatch12,                                
    bool bFixScale,                                                                
    g2o::Sim3 &gSw1w2,                                                             
    std::map<KeyFrame *, std::vector<std::pair<int, int>>> &mvpMatchedKeyPoints12, 
    MapDrawer *pMapDrawer,                                                         
    bool mbOldUdf,
    bool mbNewUdf,
    bool mbedge_merge
) {
    if (!pMap1 || !pMap2) {
        cerr << "Map Input Error!" << endl;
        return 0;
    }
    float keyPointMatchTolerancePixel = 3;
    float nSolveSim3Inliers = 5;
    float nUmeyamaSolveSim3InliersRatio = 0.3;
    int nUmeyamaSolveSim3KFNum = 15;

    const vector<KeyFrame *> &vpMap1KeyFrames = pMap1->GetAllKeyFrames();
    const vector<KeyFrame *> &vpMap2KeyFrames = pMap2->GetAllKeyFrames();

    int matchMapPointNum = 0; 
    std::map<KeyFrame *, std::vector<MapPoint *>> mvpMatchedPoints12;
    std::map<KeyFrame *, std::vector<std::pair<int, int>>> mvpValidMatchedKeyPoints12; 
    std::map<KeyFrame *, int> mvpMatchedPointsNum12;

    std::chrono::steady_clock::time_point timeStartMatchMapPoints = std::chrono::steady_clock::now();

    for (auto iter = mKfMatch12.begin(); iter != mKfMatch12.end(); iter++) { 
        KeyFrame *pMap1KF = vpMap1KeyFrames[iter->first]; 
        KeyFrame *pMap2KF = vpMap2KeyFrames[iter->second]; 

        const std::vector<MapPoint *> &vpMap1MapPoints = pMap1KF->GetMapPointMatches(); 
        const std::vector<MapPoint *> &vpMap2MapPoints = pMap2KF->GetMapPointMatches();
        std::vector<MapPoint *> vpMatchedPoints12(vpMap1MapPoints.size(), static_cast<MapPoint *>(NULL));
        std::vector<std::pair<int, int>> vpMatchedKeyPoints12;
        std::vector<std::pair<int, int>> vpValidMatchedKeyPoints12;

        int matchNum = 0;
        const std::vector<cv::KeyPoint> &map1KFKPs = pMap1KF->mvKeys; 
        const std::vector<cv::KeyPoint> &map2KFKPs = pMap2KF->mvKeys;
        
        for (unsigned long map1KFKP_i = 0; map1KFKP_i < map1KFKPs.size(); ++map1KFKP_i) { 
            const float map1_u = map1KFKPs[map1KFKP_i].pt.x; 
            const float map1_v = map1KFKPs[map1KFKP_i].pt.y;
            const vector<size_t> vIndices = pMap2KF->GetFeaturesInArea(map1_u, map1_v, keyPointMatchTolerancePixel);

            if (vIndices.empty()) {
                continue;
            }

            float best_dist = keyPointMatchTolerancePixel; 
            int best_idx = -1;
            for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++) { 
                const size_t map2KFKP_i = *vit;
                const float map2_u = map2KFKPs[map2KFKP_i].pt.x;
                const float map2_v = map2KFKPs[map2KFKP_i].pt.y;
                const float delta_pixel = (float)sqrt(pow(map1_u - map2_u, 2) + pow(map1_v - map2_v, 2));
                
                if (delta_pixel < best_dist && vpMap1MapPoints[map1KFKP_i] && vpMap2MapPoints[map2KFKP_i]) {
                    best_idx = map2KFKP_i;
                    best_dist = delta_pixel;
                }
            }
            if (best_idx == -1) {
                continue;
            }

            matchNum++;
            vpMatchedPoints12[map1KFKP_i] = vpMap2MapPoints[best_idx];
            vpMatchedKeyPoints12.push_back(std::pair<int, int>(map1KFKP_i, best_idx)); 
            vpValidMatchedKeyPoints12.push_back(std::pair<int, int>(map1KFKP_i, best_idx));
        }

        matchMapPointNum += matchNum;
        mvpMatchedPointsNum12[vpMap1KeyFrames[iter->first]] = matchNum;
        mvpMatchedPoints12[vpMap1KeyFrames[iter->first]] = vpMatchedPoints12;
        mvpMatchedKeyPoints12[vpMap1KeyFrames[iter->first]] = vpMatchedKeyPoints12;
        mvpValidMatchedKeyPoints12[vpMap1KeyFrames[iter->first]] = vpValidMatchedKeyPoints12;
    }

    cerr << "测试：匹配KeyPoint的数量：" << matchMapPointNum << endl;
    std::chrono::steady_clock::time_point timeEndMatchMapPoints = std::chrono::steady_clock::now();
    double time = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(timeEndMatchMapPoints - timeStartMatchMapPoints).count();
    cerr << "Test: Match Map Point Time " << time << endl;

    std::vector<KeyFrame *> matchMap1KeyFrames;
    std::vector<KeyFrame *> matchMap2KeyFrames;
    std::vector<std::vector<MapPoint *>> avpMatchedPoints;
    std::vector<std::vector<std::pair<int, int>>> avpValidKPMatches;
    matchMap1KeyFrames.reserve(mKfMatch12.size());
    matchMap2KeyFrames.reserve(mKfMatch12.size());
    avpMatchedPoints.reserve(mKfMatch12.size());
    avpValidKPMatches.reserve(mKfMatch12.size());
    
    for (auto iter = mKfMatch12.begin(); iter != mKfMatch12.end(); iter++) {
        matchMap1KeyFrames.push_back(vpMap1KeyFrames[iter->first]);
        matchMap2KeyFrames.push_back(vpMap2KeyFrames[iter->second]);
        avpMatchedPoints.push_back(mvpMatchedPoints12[vpMap1KeyFrames[iter->first]]);
        avpValidKPMatches.push_back(mvpValidMatchedKeyPoints12[vpMap1KeyFrames[iter->first]]);
    }

    float OptimizeInliersRatio = 0.0;
    
    // 【核心修复 1：UDF 模式视觉算法全量旁路 (Bypass)】
    if (mbOldUdf || mbNewUdf || mbedge_merge) {
        auto iter = mRandSelectKfMatch12.begin();
        KeyFrame* pKF1 = vpMap1KeyFrames[iter->first];
        KeyFrame* pKF2 = vpMap2KeyFrames[iter->second];

        Sophus::SE3f Tw1_c = pKF1->GetPoseInverse();
        Sophus::SE3f Tc_w2 = pKF2->GetPose();
        Sophus::SE3f Tw1_w2 = Tw1_c * Tc_w2;

        gSw1w2 = g2o::Sim3(Tw1_w2.rotationMatrix().cast<double>(), Tw1_w2.translation().cast<double>(), 1.0);
        
        cerr << "\033[1;32m[UDF BYPASS] Skipping visual Umeyama. Sim3 initialized from Anchor KF.\033[0m" << endl;
    } 
    else {
        if (mRandSelectKfMatch12.size() >= 3) {
            vector<Eigen::Vector3d> map1MatchKFPoints;
            vector<Eigen::Vector3d> map2MatchKFPoints;
            for (auto &iter : mRandSelectKfMatch12) {
                map1MatchKFPoints.push_back(vpMap1KeyFrames[iter.first]->GetPoseInverse().translation().cast<double>());
                map2MatchKFPoints.push_back(vpMap2KeyFrames[iter.second]->GetPoseInverse().translation().cast<double>());
            }
            Eigen::Matrix4d umeyamaResult = Sim3Solver::umeyamaSolve(map2MatchKFPoints, map1MatchKFPoints);
            Sophus::Sim3d sSc1c2(umeyamaResult.cast<double>());
            gSw1w2 = g2o::Sim3(sSc1c2.rotationMatrix(), sSc1c2.translation(), sSc1c2.scale()); 
            
            Eigen::Matrix<double, 7, 7> mHessian7x7;
            OptimizeInliersRatio = Optimizer::OptimizeCloudSim3(matchMap1KeyFrames, matchMap2KeyFrames, avpMatchedPoints, gSw1w2, 10, true, mHessian7x7, true);
        } else {
            cerr << "\033[1;31m[ERROR] Not enough matched KFs (< 3) for Visual Umeyama!\033[0m" << endl;
            return false;
        }
    }

    //cap-udf
    pcl::PointCloud<pcl::PointXYZ>::Ptr edge_point(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_point_icp(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_point(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr edge_point2(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr edge_point_icp(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_point2(new pcl::PointCloud<pcl::PointXYZ>);    
    typedef pcl::PointXYZ PointT;
    pcl::IterativeClosestPoint<PointT, PointT> icp;
    pcl::IterativeClosestPoint<PointT, PointT> icp2;
    Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity ();
    Eigen::Matrix4d transformation_matrix2 = Eigen::Matrix4d::Identity ();
    int iterations = 1;
    
    if (mbOldUdf) {
        std::string path_cloud = "/zclin/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_point/test2.xyz";
        get_parameter_xyz(path_cloud, *cloud_point); 
        system("rm /zclin/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_point/test2.xyz");               
        vector<MapPoint *> EdgeFrontMapPoint = pMap1->GetAllMapPoints();
        get_xyz(EdgeFrontMapPoint, *edge_point);
        Eigen::Quaterniond q_test = gSw1w2.rotation();
        Eigen::Matrix3d R_test = q_test.normalized().toRotationMatrix();
        
        Eigen::Vector3d T_test = gSw1w2.translation();
        double s_test = gSw1w2.scale();
        
        Eigen::Matrix3d Rs_test = R_test * s_test;
        
        Eigen::Matrix4d Trans_test;
        Trans_test.setIdentity(); 
        Trans_test.block<3,3>(0,0) = Rs_test;
        Trans_test.block<3,1>(0,3) = T_test;
        
        pcl::transformPointCloud (*cloud_point, *cloud_point_icp, Trans_test);        
        icp.setInputSource (cloud_point_icp);
        icp.setInputTarget (edge_point);
        do {
            icp.align (*cloud_point_icp);
            iterations ++;
        } while (iterations < 20);
        
        if (icp.hasConverged ()) {
            std::cout << "\nICP has converged, score is " << icp.getFitnessScore () << std::endl;
        }
        transformation_matrix = icp.getFinalTransformation ().cast<double>();
        cerr << transformation_matrix;
        cerr << "\n" << endl;        
        
        Eigen::Matrix4d trans_out = Trans_test * transformation_matrix;
        Sophus::Sim3d qts_matrix(trans_out.cast<double>()); 
        Eigen::Matrix3d qts_R = qts_matrix.rotationMatrix();
        Eigen::Vector3d qts_T = qts_matrix.translation();
        double qts_s = qts_matrix.scale();
        gSw1w2 = g2o::Sim3(qts_R, qts_T, qts_s);
    }
    
    if (mbNewUdf) {
        vector<MapPoint *> CloudMapPoint = pMap2->GetAllMapPoints();
        get_xyz(CloudMapPoint, *cloud_point); 
        vector<MapPoint *> EdgeFrontMapPoint = pMap1->GetAllMapPoints();
        get_xyz(EdgeFrontMapPoint, *edge_point);
        Eigen::Quaterniond q_test = gSw1w2.rotation();
        Eigen::Matrix3d R_test = q_test.normalized().toRotationMatrix();

        Eigen::Vector3d T_test = gSw1w2.translation();
        double s_test = gSw1w2.scale();

        Eigen::Matrix3d Rs_test = R_test * s_test;

        Eigen::Matrix4d Trans_test;
        Trans_test.setIdentity(); 
        Trans_test.block<3,3>(0,0) = Rs_test;
        Trans_test.block<3,1>(0,3) = T_test;

        pcl::transformPointCloud (*cloud_point, *cloud_point_icp, Trans_test);        
        icp.setInputSource (cloud_point_icp);
        icp.setInputTarget (edge_point);
        do {
            icp.align (*cloud_point_icp);
            iterations ++;
        } while (iterations < 20);
        
        if (icp.hasConverged ()) {
            std::cout << "\nICP has converged, score is " << icp.getFitnessScore () << std::endl;
        }
        transformation_matrix = icp.getFinalTransformation ().cast<double>();
        cerr << transformation_matrix;
        cerr << "\n" << endl;        
        
        Eigen::Matrix4d trans_out = Trans_test * transformation_matrix;
        Sophus::Sim3d qts_matrix(trans_out.cast<double>()); 
        Eigen::Matrix3d qts_R = qts_matrix.rotationMatrix();
        Eigen::Vector3d qts_T = qts_matrix.translation();
        double qts_s = qts_matrix.scale();
        gSw1w2 = g2o::Sim3(qts_R, qts_T, qts_s);
    }

    if (false) {
        vector<MapPoint *> EdgeFrontMapPoint2 = pMap1->GetAllMapPoints();
        get_xyz(EdgeFrontMapPoint2, *cloud_point2);               
        vector<MapPoint *> EdgeBackMapPoint = pMap2->GetAllMapPoints();
        get_xyz(EdgeBackMapPoint, *edge_point2);
        
        Eigen::Quaterniond q_test2 = gSw1w2.rotation();
        Eigen::Matrix3d R_test2 = q_test2.normalized().toRotationMatrix();
        Eigen::Vector3d T_test2 = gSw1w2.translation();
        double s_test2 = gSw1w2.scale();
        Eigen::Matrix3d Rs_test2 = R_test2 * s_test2;
        Eigen::Matrix4d Trans_test2;
        Trans_test2.setIdentity(); 
        Trans_test2.block<3,3>(0,0) = Rs_test2;
        Trans_test2.block<3,1>(0,3) = T_test2;
        
        pcl::transformPointCloud (*edge_point2, *edge_point_icp, Trans_test2);        
        icp2.setInputSource (edge_point_icp);
        icp2.setInputTarget (cloud_point2);
        do {
            icp2.align (*edge_point_icp);
            iterations ++;
        } while (iterations < 20);
        
        if (icp2.hasConverged ()) {
            std::cout << "\nICP has converged, score is " << icp2.getFitnessScore () << std::endl;
        }
        transformation_matrix2 = icp2.getFinalTransformation ().cast<double>();
        Eigen::Matrix4d trans_out2 = Trans_test2 * transformation_matrix2;
        Sophus::Sim3d qts_matrix2(trans_out2.cast<double>()); 
        Eigen::Matrix3d qts_R2 = qts_matrix2.rotationMatrix();
        Eigen::Vector3d qts_T2 = qts_matrix2.translation();
        double qts_s2 = qts_matrix2.scale();
        gSw1w2 = g2o::Sim3(qts_R2, qts_T2, qts_s2);
    }

    std::chrono::steady_clock::time_point timeStartComputeInliers = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point timeEndComputeInliers = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(timeEndComputeInliers - timeStartComputeInliers).count();
    cerr << "Test: Optimize Compute Inliers Time " << time << endl;

    // 【核心修复 2：短路特征点验证，强制释放 ICP 控制权】
    if (mbOldUdf || mbNewUdf || mbedge_merge) {
        cerr << "\033[1;32m[UDF BYPASS] ICP Done. Forcing Return True to execute map merge!\033[0m" << endl;
        return true; 
    }

    if (OptimizeInliersRatio > 0.1) { 
        cerr << "测试：优化Sim3的点匹配数量是否超过阈值： true, 值为 " << OptimizeInliersRatio << endl;
        return true;
    } 
    else {
        cerr << "测试：优化Sim3的点匹配数量是否超过阈值： false, 值为" << OptimizeInliersRatio << endl;
        return false;
    }

    return false;
}

bool CloudMerging::NewDetectCommonRegions() {
    if (!mbActiveCM) {
        return false;
    }

    if (mpTracker->mSensor == System::STEREO && mpLastMap->GetAllKeyFrames().size() < 5) {
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    if (mpLastMap->GetAllKeyFrames().size() < 12) {
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    bool bLoopDetectedInKF = false;
    bool bCheckSpatial = false;

    bool bMergeDetectedInKF = false;
    if (mnMergeNumCoincidences > 0) {
        Sophus::SE3d mTcl = (mpCurrentKF->GetPose() * mpMergeLastCurrentKF->GetPoseInverse()).cast<double>();
        g2o::Sim3 gScl(mTcl.unit_quaternion(), mTcl.translation(), 1.0);

        g2o::Sim3 gScw = gScl * mg2oMergeSlw;
        int numProjMatches = 0;
        vector<MapPoint *> vpMatchedMPs;
        
        bool bCommonRegion = DetectAndReffineSim3FromLastKF(mpCurrentKF, mpMergeMatchedKF, gScw, numProjMatches, mvpMergeMPs, vpMatchedMPs);
        if (bCommonRegion) {
            bMergeDetectedInKF = true;

            mnMergeNumCoincidences++; 
            mpMergeLastCurrentKF->SetErase();
            mpMergeLastCurrentKF = mpCurrentKF;
            mg2oMergeSlw = gScw;
            mvpMergeMatchedMPs = vpMatchedMPs;

            mbMergeDetected = mnMergeNumCoincidences >= 3;
        } else {
            mbMergeDetected = false;
            bMergeDetectedInKF = false;

            mnMergeNumNotFound++;
            if (mnMergeNumNotFound >= 2) {
                mpMergeLastCurrentKF->SetErase();
                mpMergeMatchedKF->SetErase();
                mnMergeNumCoincidences = 0;
                mvpMergeMatchedMPs.clear();
                mvpMergeMPs.clear();
                mnMergeNumNotFound = 0;
            }
        }
    }

    if (mbMergeDetected) {
        mpKeyFrameDB->add(mpCurrentKF);
        return true;
    }

    const vector<KeyFrame *> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();

    vector<KeyFrame *> vpMergeBowCand, vpLoopBowCand;
    if (!bMergeDetectedInKF || !bLoopDetectedInKF) {
        mpKeyFrameDB->DetectNBestCandidates(mpCurrentKF, vpLoopBowCand, vpMergeBowCand, 3);
    }

    if (!bMergeDetectedInKF && !vpMergeBowCand.empty()) {
        mbMergeDetected = DetectCommonRegionsFromBoW(vpMergeBowCand, mpMergeMatchedKF, mpMergeLastCurrentKF, mg2oMergeSlw, mnMergeNumCoincidences, mvpMergeMPs, mvpMergeMatchedMPs);
    }

    mpKeyFrameDB->add(mpCurrentKF);

    if (mbMergeDetected) {
        return true;
    }

    mpCurrentKF->SetErase();
    mpCurrentKF->mbCurrentPlaceRecognition = false;

    return false;
}

bool CloudMerging::DetectAndReffineSim3FromLastKF(KeyFrame *pCurrentKF, KeyFrame *pMatchedKF, g2o::Sim3 &gScw, int &nNumProjMatches,
                                                  std::vector<MapPoint *> &vpMPs, std::vector<MapPoint *> &vpMatchedMPs) {
    set<MapPoint *> spAlreadyMatchedMPs;
    nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);

    int nProjMatches = 30;
    int nProjOptMatches = 50;
    int nProjMatchesRep = 100;

    if (nNumProjMatches >= nProjMatches) {
        Sophus::SE3d mTwm = pMatchedKF->GetPoseInverse().cast<double>();
        g2o::Sim3 gSwm(mTwm.unit_quaternion(), mTwm.translation(), 1.0);
        g2o::Sim3 gScm = gScw * gSwm;
        Eigen::Matrix<double, 7, 7> mHessian7x7;

        bool bFixedScale = mbFixScale; 
        if (mpTracker->mSensor == System::IMU_MONOCULAR && !pCurrentKF->GetMap()->GetIniertialBA2()) {
            bFixedScale = false;
        }
            
        int numOptMatches = Optimizer::OptimizeSim3(mpCurrentKF, pMatchedKF, vpMatchedMPs, gScm, 10, bFixedScale, mHessian7x7, true);

        if (numOptMatches > nProjOptMatches) {
            g2o::Sim3 gScw_estimation(gScw.rotation(), gScw.translation(), 1.0);

            vector<MapPoint *> vpMatchedMP;
            vpMatchedMP.resize(mpCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint *>(NULL));

            nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw_estimation, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);
            if (nNumProjMatches >= nProjMatchesRep) {
                gScw = gScw_estimation;
                return true;
            }
        }
    }
    return false;
}

bool CloudMerging::DetectCommonRegionsFromBoW(std::vector<KeyFrame *> &vpBowCand, KeyFrame *&pMatchedKF2, KeyFrame *&pLastCurrentKF, g2o::Sim3 &g2oScw,
                                              int &nNumCoincidences, std::vector<MapPoint *> &vpMPs, std::vector<MapPoint *> &vpMatchedMPs) {
    int nBoWMatches = 20;
    int nBoWInliers = 15;
    int nSim3Inliers = 20;
    int nProjMatches = 50;
    int nProjOptMatches = 80;

    set<KeyFrame *> spConnectedKeyFrames = mpCurrentKF->GetConnectedKeyFrames();

    int nNumCovisibles = 10;

    ORBmatcher matcherBoW(0.9, true);
    ORBmatcher matcher(0.75, true);

    KeyFrame *pBestMatchedKF;
    int nBestMatchesReproj = 0;
    int nBestNumCoindicendes = 0;
    g2o::Sim3 g2oBestScw;
    std::vector<MapPoint *> vpBestMapPoints;
    std::vector<MapPoint *> vpBestMatchedMapPoints;

    int numCandidates = vpBowCand.size();
    vector<int> vnStage(numCandidates, 0);
    vector<int> vnMatchesStage(numCandidates, 0);

    int index = 0;
    
    for (KeyFrame *pKFi : vpBowCand) {
        if (!pKFi || pKFi->isBad()) {
            continue;
        }

        std::vector<KeyFrame *> vpCovKFi = pKFi->GetBestCovisibilityKeyFrames(nNumCovisibles);
        if (vpCovKFi.empty()) {
            std::cerr << "Covisible list empty" << std::endl;
            vpCovKFi.push_back(pKFi);
        } else {
            vpCovKFi.push_back(vpCovKFi[0]);
            vpCovKFi[0] = pKFi;
        }

        bool bAbortByNearKF = false;
        for (int j = 0; j < vpCovKFi.size(); ++j) {
            if (spConnectedKeyFrames.find(vpCovKFi[j]) != spConnectedKeyFrames.end()) {
                bAbortByNearKF = true;
                break;
            }
        }
        if (bAbortByNearKF) {
            continue;
        }

        std::vector<std::vector<MapPoint *>> vvpMatchedMPs;
        vvpMatchedMPs.resize(vpCovKFi.size());
        std::set<MapPoint *> spMatchedMPi;
        int numBoWMatches = 0;

        KeyFrame *pMostBoWMatchesKF = pKFi;
        int nMostBoWNumMatches = 0;

        std::vector<MapPoint *> vpMatchedPoints = std::vector<MapPoint *>(mpCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint *>(NULL));
        std::vector<KeyFrame *> vpKeyFrameMatchedMP = std::vector<KeyFrame *>(mpCurrentKF->GetMapPointMatches().size(), static_cast<KeyFrame *>(NULL));

        int nIndexMostBoWMatchesKF = 0;
        for (int j = 0; j < vpCovKFi.size(); ++j) {
            if (!vpCovKFi[j] || vpCovKFi[j]->isBad()) {
                continue;
            }

            int num = matcherBoW.SearchByBoW(mpCurrentKF, vpCovKFi[j], vvpMatchedMPs[j]);
            if (num > nMostBoWNumMatches) {
                nMostBoWNumMatches = num;
                nIndexMostBoWMatchesKF = j;
            }
        }

        for (int j = 0; j < vpCovKFi.size(); ++j) {
            for (int k = 0; k < vvpMatchedMPs[j].size(); ++k) {
                MapPoint *pMPi_j = vvpMatchedMPs[j][k];
                if (!pMPi_j || pMPi_j->isBad()) {
                    continue;
                }

                if (spMatchedMPi.find(pMPi_j) == spMatchedMPi.end()) {
                    spMatchedMPi.insert(pMPi_j);
                    numBoWMatches++;

                    vpMatchedPoints[k] = pMPi_j;          
                    vpKeyFrameMatchedMP[k] = vpCovKFi[j]; 
                }
            }
        }

        if (numBoWMatches >= nBoWMatches) {
            bool bFixedScale = mbFixScale;

            Sim3Solver solver = Sim3Solver(mpCurrentKF, pMostBoWMatchesKF, vpMatchedPoints, bFixedScale, vpKeyFrameMatchedMP);
            solver.SetRansacParameters(0.99, nBoWInliers, 300); 

            bool bNoMore = false;
            vector<bool> vbInliers;
            int nInliers;
            bool bConverge = false;
            Eigen::Matrix4f mTcm;
            while (!bConverge && !bNoMore) {
                mTcm = solver.iterate(20, bNoMore, vbInliers, nInliers, bConverge);
            }

            if (bConverge) {
                vpCovKFi.clear();
                vpCovKFi = pMostBoWMatchesKF->GetBestCovisibilityKeyFrames(nNumCovisibles);
                vpCovKFi.push_back(pMostBoWMatchesKF);
                set<KeyFrame *> spCheckKFs(vpCovKFi.begin(), vpCovKFi.end());

                set<MapPoint *> spMapPoints;
                vector<MapPoint *> vpMapPoints;
                vector<KeyFrame *> vpKeyFrames;
                for (KeyFrame *pCovKFi : vpCovKFi) {
                    for (MapPoint *pCovMPij : pCovKFi->GetMapPointMatches()) {
                        if (!pCovMPij || pCovMPij->isBad()) {
                            continue;
                        }

                        if (spMapPoints.find(pCovMPij) == spMapPoints.end()) {
                            spMapPoints.insert(pCovMPij);
                            vpMapPoints.push_back(pCovMPij);
                            vpKeyFrames.push_back(pCovKFi);
                        }
                    }
                }

                g2o::Sim3 gScm(solver.GetEstimatedRotation().cast<double>(), solver.GetEstimatedTranslation().cast<double>(), (double)solver.GetEstimatedScale());
                g2o::Sim3 gSmw(pMostBoWMatchesKF->GetRotation().cast<double>(), pMostBoWMatchesKF->GetTranslation().cast<double>(), 1.0);
                g2o::Sim3 gScw = gScm * gSmw; 
                Sophus::Sim3f mScw = Converter::toSophus(gScw);

                vector<MapPoint *> vpMatchedMP;
                vpMatchedMP.resize(mpCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint *>(NULL));
                vector<KeyFrame *> vpMatchedKF;
                vpMatchedKF.resize(mpCurrentKF->GetMapPointMatches().size(), static_cast<KeyFrame *>(NULL));
                
                int numProjMatches = matcher.SearchByProjection(mpCurrentKF, mScw, vpMapPoints, vpKeyFrames, vpMatchedMP, vpMatchedKF, 8, 1.5);

                if (numProjMatches >= nProjMatches) {
                    Eigen::Matrix<double, 7, 7> mHessian7x7;

                    bool bFixedScale = mbFixScale;
                    if (mpTracker->mSensor == System::IMU_MONOCULAR && !mpCurrentKF->GetMap()->GetIniertialBA2()) {
                        bFixedScale = false;
                    }

                    int numOptMatches = Optimizer::OptimizeSim3(mpCurrentKF, pKFi, vpMatchedMP, gScm, 10, mbFixScale, mHessian7x7, true);

                    if (numOptMatches >= nSim3Inliers) {
                        g2o::Sim3 gSmw(pMostBoWMatchesKF->GetRotation().cast<double>(), pMostBoWMatchesKF->GetTranslation().cast<double>(), 1.0);
                        g2o::Sim3 gScw = gScm * gSmw; 
                        Sophus::Sim3f mScw = Converter::toSophus(gScw);

                        vector<MapPoint *> vpMatchedMP;
                        vpMatchedMP.resize(mpCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint *>(NULL));
                        
                        int numProjOptMatches = matcher.SearchByProjection(mpCurrentKF, mScw, vpMapPoints, vpMatchedMP, 5, 1.0);

                        if (numProjOptMatches >= nProjOptMatches) {
                            int max_x = -1, min_x = 1000000;
                            int max_y = -1, min_y = 1000000;
                            for (MapPoint *pMPi : vpMatchedMP) {
                                if (!pMPi || pMPi->isBad()) {
                                    continue;
                                }

                                tuple<size_t, size_t> indexes = pMPi->GetIndexInKeyFrame(pKFi);
                                int index = get<0>(indexes);
                                if (index >= 0) {
                                    int coord_x = pKFi->mvKeysUn[index].pt.x;
                                    if (coord_x < min_x) {
                                        min_x = coord_x;
                                    }
                                    if (coord_x > max_x) {
                                        max_x = coord_x;
                                    }
                                    int coord_y = pKFi->mvKeysUn[index].pt.y;
                                    if (coord_y < min_y) {
                                        min_y = coord_y;
                                    }
                                    if (coord_y > max_y) {
                                        max_y = coord_y;
                                    }
                                }
                            }

                            int nNumKFs = 0;
                            
                            vector<KeyFrame *> vpCurrentCovKFs = mpCurrentKF->GetBestCovisibilityKeyFrames(nNumCovisibles);

                            int j = 0;
                            while (nNumKFs < 3 && j < vpCurrentCovKFs.size()) {
                                KeyFrame *pKFj = vpCurrentCovKFs[j];
                                Sophus::SE3d mTjc = (pKFj->GetPose() * mpCurrentKF->GetPoseInverse()).cast<double>();
                                g2o::Sim3 gSjc(mTjc.unit_quaternion(), mTjc.translation(), 1.0);
                                g2o::Sim3 gSjw = gSjc * gScw;
                                int numProjMatches_j = 0;
                                vector<MapPoint *> vpMatchedMPs_j;
                                
                                bool bValid = DetectCommonRegionsFromLastKF(pKFj, pMostBoWMatchesKF, gSjw, numProjMatches_j, vpMapPoints, vpMatchedMPs_j);

                                if (bValid) {
                                    Sophus::SE3f Tc_w = mpCurrentKF->GetPose();
                                    Sophus::SE3f Tw_cj = pKFj->GetPoseInverse();
                                    Sophus::SE3f Tc_cj = Tc_w * Tw_cj;
                                    Eigen::Vector3f vector_dist = Tc_cj.translation();
                                    nNumKFs++;
                                }
                                j++;
                            }

                            if (nNumKFs < 3) {
                                vnStage[index] = 8;
                                vnMatchesStage[index] = nNumKFs;
                            }

                            if (nBestMatchesReproj < numProjOptMatches) {
                                nBestMatchesReproj = numProjOptMatches;
                                nBestNumCoindicendes = nNumKFs;
                                pBestMatchedKF = pMostBoWMatchesKF;
                                g2oBestScw = gScw;
                                
                                vpBestMapPoints = vpMapPoints;
                                vpBestMatchedMapPoints = vpMatchedMP;
                            }
                        }
                    }
                }
            }
        }
        index++;
    }
    
    if (nBestMatchesReproj > 0) {
        pLastCurrentKF = mpCurrentKF;
        nNumCoincidences = nBestNumCoindicendes;
        
        pMatchedKF2 = pBestMatchedKF;
        pMatchedKF2->SetNotErase();
        g2oScw = g2oBestScw; 
        
        vpMPs = vpBestMapPoints;
        vpMatchedMPs = vpBestMatchedMapPoints;

        return nNumCoincidences >= 3;
    } else {
        int maxStage = -1;
        int maxMatched;
        for (int i = 0; i < vnStage.size(); ++i) {
            if (vnStage[i] > maxStage) {
                maxStage = vnStage[i];
                maxMatched = vnMatchesStage[i];
            }
        }
    }
    return false;
}

bool CloudMerging::DetectCommonRegionsFromLastKF(KeyFrame *pCurrentKF, KeyFrame *pMatchedKF, g2o::Sim3 &gScw, int &nNumProjMatches,
                                                 std::vector<MapPoint *> &vpMPs, std::vector<MapPoint *> &vpMatchedMPs) {
    set<MapPoint *> spAlreadyMatchedMPs(vpMatchedMPs.begin(), vpMatchedMPs.end());
    nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);

    int nProjMatches = 30;
    if (nNumProjMatches >= nProjMatches) {
        return true;
    }

    return false;
}

int CloudMerging::FindMatchesByProjection(KeyFrame *pCurrentKF, KeyFrame *pMatchedKFw, g2o::Sim3 &g2oScw,
                                          set<MapPoint *> &spMatchedMPinOrigin, vector<MapPoint *> &vpMapPoints,
                                          vector<MapPoint *> &vpMatchedMapPoints) {
    int nNumCovisibles = 10;
    vector<KeyFrame *> vpCovKFm = pMatchedKFw->GetBestCovisibilityKeyFrames(nNumCovisibles);
    int nInitialCov = vpCovKFm.size();
    vpCovKFm.push_back(pMatchedKFw);
    set<KeyFrame *> spCheckKFs(vpCovKFm.begin(), vpCovKFm.end());
    set<KeyFrame *> spCurrentCovisbles = pCurrentKF->GetConnectedKeyFrames();
    
    if (nInitialCov < nNumCovisibles) {
        for (int i = 0; i < nInitialCov; ++i) {
            vector<KeyFrame *> vpKFs = vpCovKFm[i]->GetBestCovisibilityKeyFrames(nNumCovisibles);
            int nInserted = 0;
            int j = 0;
            while (j < vpKFs.size() && nInserted < nNumCovisibles) {
                if (spCheckKFs.find(vpKFs[j]) == spCheckKFs.end() && spCurrentCovisbles.find(vpKFs[j]) == spCurrentCovisbles.end()) {
                    spCheckKFs.insert(vpKFs[j]);
                    ++nInserted;
                }
                ++j;
            }
            vpCovKFm.insert(vpCovKFm.end(), vpKFs.begin(), vpKFs.end());
        }
    }
    set<MapPoint *> spMapPoints;
    vpMapPoints.clear();
    vpMatchedMapPoints.clear();
    for (KeyFrame *pKFi : vpCovKFm) {
        for (MapPoint *pMPij : pKFi->GetMapPointMatches()) {
            if (!pMPij || pMPij->isBad()) {
                continue;
            }

            if (spMapPoints.find(pMPij) == spMapPoints.end()) {
                spMapPoints.insert(pMPij);
                vpMapPoints.push_back(pMPij);
            }
        }
    }

    Sophus::Sim3f mScw = Converter::toSophus(g2oScw);
    ORBmatcher matcher(0.9, true);

    vpMatchedMapPoints.resize(pCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint *>(NULL));
    int num_matches = matcher.SearchByProjection(pCurrentKF, mScw, vpMapPoints, vpMatchedMapPoints, 3, 1.5);

    return num_matches;
}

/*!
 * @param [in] pMainMap 作为Main Map
 * @param [in] pMergedMap  作为Merge Map，将被合并到Main Map
 * @param [in] gSwMainMerged  Sim3变换，Merged Map World -> Main Map World
 * @param [in] pLocalMapper  用于暂停、开启LocalMapper，保证LocalMapper的工作时间
 * @param [in] blockLocalMapper  是否暂停Local Mapper，如果是EdgeFront 和 Cloud，不涉及当前运行的Map，就不需要暂停
 */
void CloudMerging::CloudMergeMap(
    Map *pMainMap, Map *pMergedMap,  
    g2o::Sim3 gSwMainMerged,
    const std::map<int, int> &pMainMergedKeyFrameMatch,
    std::map<KeyFrame *, std::vector<std::pair<int, int>>> &vpMainMergedMapPointsMatch,
    LocalMapping *pLocalMapper, bool blockLocalMapper,
    bool bOldUdf, bool bNewUdf) {
    
    const vector<KeyFrame *> &vpMainMapKeyFrames = pMainMap->GetAllKeyFrames();
    const vector<KeyFrame *> &vpMergedMapKeyFrames = pMergedMap->GetAllKeyFrames();
    auto iter = pMainMergedKeyFrameMatch.begin();
    std::advance(iter, rand() % pMainMergedKeyFrameMatch.size());
    KeyFrame *pRandMainKF = vpMainMapKeyFrames[iter->first];
    KeyFrame *pRandMergedKF = vpMergedMapKeyFrames[iter->second];

    Sophus::SE3d mTCameraMergedWorld = pRandMergedKF->GetPose().cast<double>(); 
    g2o::Sim3 gSCameraMergedWorld(mTCameraMergedWorld.unit_quaternion(), mTCameraMergedWorld.translation(), 1.0); 
    g2o::Sim3 gSMergedCameraMainWorld = gSCameraMergedWorld * gSwMainMerged.inverse(); 

    if (blockLocalMapper) {
        pLocalMapper->RequestStop();
        while (!pLocalMapper->isStopped()) {
            usleep(1000);
        }

        pLocalMapper->EmptyQueue(); 
    }

    pRandMergedKF->UpdateConnections();

    set<KeyFrame *> spMergedMapWindowKFs;
    set<MapPoint *> spMergedMapWindowMPs; 

    set<KeyFrame *> spMainMapWindowKFs;
    set<MapPoint *> spMainMapWindowMPs; 

    int numTemporalKFs = 5;
    for (iter = pMainMergedKeyFrameMatch.begin(); iter != pMainMergedKeyFrameMatch.end(); iter++) {
        KeyFrame *pCurMainKF = vpMainMapKeyFrames[iter->first];
        KeyFrame *pCurMergedKF = vpMergedMapKeyFrames[iter->second];

        spMergedMapWindowKFs.insert(pCurMergedKF);
        vector<KeyFrame *> vpCurMergedCovisibleKFs = pCurMergedKF->GetBestCovisibilityKeyFrames(numTemporalKFs);
        spMergedMapWindowKFs.insert(vpCurMergedCovisibleKFs.begin(), vpCurMergedCovisibleKFs.end());

        spMainMapWindowKFs.insert(pCurMainKF);
        vector<KeyFrame *> vpCurMainCovisibleKFs = pCurMainKF->GetBestCovisibilityKeyFrames(numTemporalKFs);
        spMainMapWindowKFs.insert(vpCurMainCovisibleKFs.begin(), vpCurMainCovisibleKFs.end());
    }

    for (KeyFrame *pKFi : spMergedMapWindowKFs) { 
        if (!pKFi || pKFi->isBad()) {
            continue;
        }

        set<MapPoint *> spMPs = pKFi->GetMapPoints(); 
        spMergedMapWindowMPs.insert(spMPs.begin(), spMPs.end()); 
    }

    for (KeyFrame *pKFi : spMainMapWindowKFs) {
        if (!pKFi || pKFi->isBad()) {
            continue;
        }

        set<MapPoint *> vpMPs = pKFi->GetMapPoints();
        spMainMapWindowMPs.insert(vpMPs.begin(), vpMPs.end());
    }

    int nAddObservation = 0;
    
    // 【核心修复 1：UDF 旁路假视觉约束】使用参数 bOldUdf 和 bNewUdf
    if (!bOldUdf && !bNewUdf) {
        for (iter = pMainMergedKeyFrameMatch.begin(); iter != pMainMergedKeyFrameMatch.end(); iter++) {
            KeyFrame *pCurMainKF = vpMainMapKeyFrames[iter->first];
            KeyFrame *pCurMergedKF = vpMergedMapKeyFrames[iter->second];

            if (!(vpMainMergedMapPointsMatch.count(pCurMainKF))) {
                cerr << "Error! " << endl;
                exit(200000);
            }
            std::vector<std::pair<int, int>> vpMapPointsMatch = vpMainMergedMapPointsMatch[pCurMainKF];
            for (auto &iter_match : vpMapPointsMatch) {
                MapPoint *pCurMainMP = pCurMainKF->GetMapPoint(iter_match.first);
                MapPoint *pCurMergedMP = pCurMergedKF->GetMapPoint(iter_match.second);
                if (pCurMainMP && !pCurMainMP->isBad()) {
                    pCurMainMP->AddObservation(pCurMergedKF, iter_match.second);
                    nAddObservation++;
                }
                if (pCurMergedMP && !pCurMergedMP->isBad()) {
                    pCurMergedMP->AddObservation(pCurMainKF, iter_match.first);
                    nAddObservation++;
                }
            }
        }
    } else {
        cerr << "\033[1;32m[UDF BYPASS] Skipping cross-map visual observations.\033[0m" << endl;
    }
    cerr << "Test: Add observation num: " << nAddObservation << endl;

    Sophus::SE3d Twc = pRandMergedKF->GetPoseInverse().cast<double>();
    g2o::Sim3 g2oNonCorrectedSwc(Twc.unit_quaternion(), Twc.translation(), 1.0);
    g2o::Sim3 g2oNonCorrectedScw = g2oNonCorrectedSwc.inverse();
    g2o::Sim3 g2oCorrectedScw = gSMergedCameraMainWorld; 

    KeyFrameAndPose vCorrectedSim3, vNonCorrectedSim3;
    vCorrectedSim3[pRandMergedKF] = g2oCorrectedScw; 
    vNonCorrectedSim3[pRandMergedKF] = g2oNonCorrectedScw;

    for (KeyFrame *pKFi : spMergedMapWindowKFs) {
        if (!pKFi || pKFi->isBad()) {
            Verbose::PrintMess("Bad KF in correction", Verbose::VERBOSITY_DEBUG);
            continue;
        }

        if (pKFi->GetMap() != pMergedMap) {
            Verbose::PrintMess("Other map KF, this should't happen", Verbose::VERBOSITY_DEBUG);
        }

        g2o::Sim3 g2oCorrectedSiw;

        if (pKFi != pRandMergedKF) {
            Sophus::SE3d Tiw = (pKFi->GetPose()).cast<double>();
            g2o::Sim3 g2oSiw(Tiw.unit_quaternion(), Tiw.translation(), 1.0);
            vNonCorrectedSim3[pKFi] = g2oSiw;

            Sophus::SE3d Tic = Tiw * Twc;                                    
            g2o::Sim3 g2oSic(Tic.unit_quaternion(), Tic.translation(), 1.0); 
            g2oCorrectedSiw = g2oSic * gSMergedCameraMainWorld;
            vCorrectedSim3[pKFi] = g2oCorrectedSiw;
        } else {
            g2oCorrectedSiw = g2oCorrectedScw;
        }
        pKFi->mTcwMerge = pKFi->GetPose();

        double s = g2oCorrectedSiw.scale();
        pKFi->mfScale = s;
        Sophus::SE3d correctedTiw(g2oCorrectedSiw.rotation(), g2oCorrectedSiw.translation() / s);
        pKFi->mTcwMerge = correctedTiw.cast<float>();
    }

    int numPointsWithCorrection = 0;

    set<MapPoint *>::iterator itMP = spMergedMapWindowMPs.begin();
    while (itMP != spMergedMapWindowMPs.end()) {
        MapPoint *pMPi = *itMP;
        if (!pMPi || pMPi->isBad()) {
            itMP = spMergedMapWindowMPs.erase(itMP);
            continue;
        }

        KeyFrame *pKFref = pMPi->GetReferenceKeyFrame();
        if (vCorrectedSim3.find(pKFref) == vCorrectedSim3.end()) {
            itMP = spMergedMapWindowMPs.erase(itMP);
            numPointsWithCorrection++;
            continue;
        }
        g2o::Sim3 g2oCorrectedSwi = vCorrectedSim3[pKFref].inverse();
        g2o::Sim3 g2oNonCorrectedSiw = vNonCorrectedSim3[pKFref];

        Eigen::Vector3d P3Dw = pMPi->GetWorldPos().cast<double>();
        Eigen::Vector3d eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oNonCorrectedSiw.map(P3Dw));
        Eigen::Quaterniond Rcor = g2oCorrectedSwi.rotation() * g2oNonCorrectedSiw.rotation();

        pMPi->mPosMerge = eigCorrectedP3Dw.cast<float>(); 
        pMPi->mNormalVectorMerge = Rcor.cast<float>() * pMPi->GetNormal(); 
        itMP++;
    }

    {
        unique_lock<mutex> currentLock(pMergedMap->mMutexMapUpdate); 
        unique_lock<mutex> mergeLock(pMainMap->mMutexMapUpdate);     

        for (KeyFrame *pKFi : spMergedMapWindowKFs) {
            if (!pKFi || pKFi->isBad()) {
                continue;
            }

            pKFi->mTcwBefMerge = pKFi->GetPose();        
            pKFi->mTwcBefMerge = pKFi->GetPoseInverse(); 
            pKFi->SetPose(pKFi->mTcwMerge);

            pKFi->UpdateMap(pMainMap);
            pKFi->mnMergeCorrectedForKF = pRandMergedKF->mnId;
            pMainMap->AddKeyFrame(pKFi);
            pMergedMap->EraseKeyFrame(pKFi);
        }

        for (MapPoint *pMPi : spMergedMapWindowMPs) {
            if (!pMPi || pMPi->isBad()) {
                continue;
            }

            pMPi->SetWorldPos(pMPi->mPosMerge);
            pMPi->SetNormalVector(pMPi->mNormalVectorMerge);
            pMPi->UpdateMap(pMainMap);
            pMainMap->AddMapPoint(pMPi);
            pMergedMap->EraseMapPoint(pMPi);
        }

        pMainMap->IncreaseChangeIndex();

        std::cerr << "[Merge]: merging maps finished" << std::endl;
        pMainMap->mvMergedMapIds.push_back(pMergedMap->GetId());
    }

    KeyFrame *pNewChild;
    KeyFrame *pNewParent;
    pMergedMap->GetOriginKF()->SetFirstConnection(false);
    pNewChild = pRandMergedKF->GetParent(); 
    pNewParent = pRandMergedKF;             
    pRandMergedKF->ChangeParent(pRandMainKF);
    while (pNewChild) 
    {
        pNewChild->EraseChild(pNewParent); 
        KeyFrame *pOldParent = pNewChild->GetParent();

        pNewChild->ChangeParent(pNewParent);

        pNewParent = pNewChild;
        pNewChild = pOldParent;
    }

    pRandMainKF->UpdateConnections();

    for (KeyFrame *pKFi : spMergedMapWindowKFs) {
        if (!pKFi || pKFi->isBad()) {
            continue;
        }

        pKFi->UpdateConnections();
    }
    for (KeyFrame *pKFi : spMainMapWindowKFs) {
        if (!pKFi || pKFi->isBad()) {
            continue;
        }

        pKFi->UpdateConnections();
    }

    std::cerr << "[Merge]: Start welding bundle adjustment" << std::endl;

    bool bStop = false;
    vector<KeyFrame *> vpMergedMapWindowKFs;
    vector<KeyFrame *> vpMainMapKFs;
    std::copy(spMergedMapWindowKFs.begin(), spMergedMapWindowKFs.end(), std::back_inserter(vpMergedMapWindowKFs));
    std::copy(spMainMapWindowKFs.begin(), spMainMapWindowKFs.end(), std::back_inserter(vpMainMapKFs));
    
    // 【核心修复 2：彻底切断 Welding Local Bundle Adjustment】使用参数 bOldUdf 和 bNewUdf
    if (bOldUdf || bNewUdf) {
        std::cerr << "\033[1;32m[UDF BYPASS] Skipping Welding Local Bundle Adjustment (Visual BA is invalid for UDF).\033[0m" << std::endl;
    } else {
        std::cerr << "[Merge]: Local bundle adjustment, spLocalWindowKFs size: " << spMergedMapWindowKFs.size() << std::endl;
        std::cerr << "[Merge]: Local bundle adjustment, spMergeConnectedKFs size: " << spMainMapWindowKFs.size() << std::endl;
        Optimizer::LocalBundleAdjustment(pRandMergedKF, vpMergedMapWindowKFs, vpMainMapKFs, &bStop);
    }

    std::cerr << "[Merge]: Welding bundle adjustment finished" << std::endl;

    if (blockLocalMapper) {
        pLocalMapper->Release();
    }

    vector<KeyFrame *> vpCurrentMergedMapKFs = pMergedMap->GetAllKeyFrames();
    vector<MapPoint *> vpCurrentMergedMapMPs = pMergedMap->GetAllMapPoints();

    if (vpCurrentMergedMapKFs.size() == 0) {
    } else {
        {
            unique_lock<mutex> currentLock(pMergedMap->mMutexMapUpdate); 

            for (KeyFrame *pKFi : vpCurrentMergedMapKFs) {
                if (!pKFi || pKFi->isBad() || pKFi->GetMap() != pMergedMap) {
                    continue;
                }

                g2o::Sim3 g2oCorrectedSiw;

                Sophus::SE3d Tiw = (pKFi->GetPose()).cast<double>();
                g2o::Sim3 g2oSiw(Tiw.unit_quaternion(), Tiw.translation(), 1.0);
                vNonCorrectedSim3[pKFi] = g2oSiw; 

                Sophus::SE3d Tic = Tiw * Twc;
                g2o::Sim3 g2oSim(Tic.unit_quaternion(), Tic.translation(), 1.0);
                g2oCorrectedSiw = g2oSim * gSMergedCameraMainWorld;
                vCorrectedSim3[pKFi] = g2oCorrectedSiw; 

                double s = g2oCorrectedSiw.scale();

                pKFi->mfScale = s;

                Sophus::SE3d correctedTiw(g2oCorrectedSiw.rotation(), g2oCorrectedSiw.translation() / s);

                pKFi->mTcwBefMerge = pKFi->GetPose();
                pKFi->mTwcBefMerge = pKFi->GetPoseInverse();

                pKFi->SetPose(correctedTiw.cast<float>());
            }
            for (MapPoint *pMPi : vpCurrentMergedMapMPs) {
                if (!pMPi || pMPi->isBad() || pMPi->GetMap() != pMergedMap) {
                    continue;
                }

                KeyFrame *pKFref = pMPi->GetReferenceKeyFrame();
                g2o::Sim3 g2oCorrectedSwi = vCorrectedSim3[pKFref].inverse();
                g2o::Sim3 g2oNonCorrectedSiw = vNonCorrectedSim3[pKFref];

                Eigen::Vector3d P3Dw = pMPi->GetWorldPos().cast<double>();
                Eigen::Vector3d eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oNonCorrectedSiw.map(P3Dw));
                pMPi->SetWorldPos(eigCorrectedP3Dw.cast<float>());

                pMPi->UpdateNormalAndDepth();
            }
        }

        if (blockLocalMapper) {
            pLocalMapper->RequestStop();
            while (!pLocalMapper->isStopped()) {
                usleep(1000);
            }
        }

        {
            unique_lock<mutex> currentLock(pMergedMap->mMutexMapUpdate); 
            unique_lock<mutex> mergeLock(pMainMap->mMutexMapUpdate);     

            for (KeyFrame *pKFi : vpCurrentMergedMapKFs) {
                if (!pKFi || pKFi->isBad() || pKFi->GetMap() != pMergedMap) {
                    continue;
                }

                pKFi->UpdateMap(pMainMap);
                pMainMap->AddKeyFrame(pKFi);
                pMergedMap->EraseKeyFrame(pKFi);
            }

            for (MapPoint *pMPi : vpCurrentMergedMapMPs) {
                if (!pMPi || pMPi->isBad()) {
                    continue;
                }

                pMPi->UpdateMap(pMainMap);
                pMainMap->AddMapPoint(pMPi);
                pMergedMap->EraseMapPoint(pMPi);
            }
        }
    }
    if (blockLocalMapper) {
        pLocalMapper->Release();
    }
}

void CloudMerging::CheckObservations(set<KeyFrame *> &spKFsMap1, set<KeyFrame *> &spKFsMap2) {
    cerr << "----------------------" << endl;
    for (KeyFrame *pKFi1 : spKFsMap1) {
        map<KeyFrame *, int> mMatchedMP;
        set<MapPoint *> spMPs = pKFi1->GetMapPoints();

        for (MapPoint *pMPij : spMPs) {
            if (!pMPij || pMPij->isBad()) {
                continue;
            }

            map<KeyFrame *, tuple<int, int>> mMPijObs = pMPij->GetObservations();
            for (KeyFrame *pKFi2 : spKFsMap2) {
                if (mMPijObs.find(pKFi2) != mMPijObs.end()) {
                    if (mMatchedMP.find(pKFi2) != mMatchedMP.end()) {
                        mMatchedMP[pKFi2] = mMatchedMP[pKFi2] + 1;
                    } else {
                        mMatchedMP[pKFi2] = 1;
                    }
                }
            }
        }

        if (mMatchedMP.size() == 0) {
            cerr << "CHECK-OBS: KF " << pKFi1->mnId << " has not any matched MP with the other map" << endl;
        } else {
            cerr << "CHECK-OBS: KF " << pKFi1->mnId << " has matched MP with " << mMatchedMP.size() << " KF from the other map" << endl;
            for (pair<KeyFrame *, int> matchedKF : mMatchedMP) {
                cerr << "   -KF: " << matchedKF.first->mnId << ", Number of matches: " << matchedKF.second << endl;
            }
        }
    }
    cerr << "----------------------" << endl;
}

void CloudMerging::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap, vector<MapPoint *> &vpMapPoints) {
    ORBmatcher matcher(0.8);

    int total_replaces = 0;

    for (KeyFrameAndPose::const_iterator mit = CorrectedPosesMap.begin(), mend = CorrectedPosesMap.end(); mit != mend; mit++) {
        int num_replaces = 0;
        KeyFrame *pKFi = mit->first;
        Map *pMap = pKFi->GetMap();

        g2o::Sim3 g2oScw = mit->second;
        Sophus::Sim3f Scw = Converter::toSophus(g2oScw);

        vector<MapPoint *> vpReplacePoints(vpMapPoints.size(), static_cast<MapPoint *>(NULL));
        int numFused = matcher.Fuse(pKFi, Scw, vpMapPoints, 4, vpReplacePoints); 

        unique_lock<mutex> lock(pMap->mMutexMapUpdate);
        const int nLP = vpMapPoints.size();
        for (int i = 0; i < nLP; i++) {
            MapPoint *pRep = vpReplacePoints[i];
            if (pRep) {
                num_replaces += 1;
                pRep->Replace(vpMapPoints[i]);
            }
        }

        total_replaces += num_replaces;
    }
}

void CloudMerging::SearchAndFuse(const vector<KeyFrame *> &vConectedKFs, vector<MapPoint *> &vpMapPoints) {
    ORBmatcher matcher(0.8);

    int total_replaces = 0;

    for (auto mit = vConectedKFs.begin(), mend = vConectedKFs.end(); mit != mend; mit++) {
        int num_replaces = 0;
        KeyFrame *pKF = (*mit);
        Map *pMap = pKF->GetMap();
        Sophus::SE3f Tcw = pKF->GetPose();
        Sophus::Sim3f Scw(Tcw.unit_quaternion(), Tcw.translation());
        Scw.setScale(1.f);
        
        vector<MapPoint *> vpReplacePoints(vpMapPoints.size(), static_cast<MapPoint *>(NULL));
        matcher.Fuse(pKF, Scw, vpMapPoints, 4, vpReplacePoints);

        unique_lock<mutex> lock(pMap->mMutexMapUpdate);
        const int nLP = vpMapPoints.size();
        for (int i = 0; i < nLP; i++) {
            MapPoint *pRep = vpReplacePoints[i];
            if (pRep) {
                num_replaces += 1;
                pRep->Replace(vpMapPoints[i]);
            }
        }
    }
}

void CloudMerging::RequestReset() {
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while (1) {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if (!mbResetRequested) {
                break;
            }
        }
        usleep(5000);
    }
}

void CloudMerging::RequestResetActiveMap(Map *pMap) {
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetActiveMapRequested = true;
        mpMapToReset = pMap;
    }

    while (1) {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if (!mbResetActiveMapRequested) {
                break;
            }
        }
        usleep(3000);
    }
}

void CloudMerging::ResetIfRequested() {

}

void CloudMerging::RunGlobalBundleAdjustment(Map *pActiveMap, unsigned long nLoopKF) {
    mbFinishedGBA = true;
    mbRunningGBA = false;
    return; 
    
    Verbose::PrintMess("Starting Global Bundle Adjustment", Verbose::VERBOSITY_NORMAL);

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartFGBA = std::chrono::steady_clock::now();

    nFGBA_exec += 1;

    vnGBAKFs.push_back(pActiveMap->GetAllKeyFrames().size());
    vnGBAMPs.push_back(pActiveMap->GetAllMapPoints().size());
#endif

    const bool bImuInit = pActiveMap->isImuInitialized();

    if (!bImuInit) {
        Optimizer::GlobalBundleAdjustemnt(pActiveMap, 10, &mbStopGBA, nLoopKF, false);
    } else {
        Optimizer::FullInertialBA(pActiveMap, 7, false, nLoopKF, &mbStopGBA);
    }

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndGBA = std::chrono::steady_clock::now();

    double timeGBA = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_EndGBA - time_StartFGBA).count();
    vdGBA_ms.push_back(timeGBA);

    if (mbStopGBA) {
        nFGBA_abort += 1;
    }
#endif

    int idx = mnFullBAIdx;
    
    {
        unique_lock<mutex> lock(mMutexGBA);
        if (idx != mnFullBAIdx) {
            return;
        }

        if (!bImuInit && pActiveMap->isImuInitialized()) {
            return;
        }

        if (!mbStopGBA) {
            Verbose::PrintMess("Global Bundle Adjustment finished", Verbose::VERBOSITY_NORMAL);
            Verbose::PrintMess("Updating map ...", Verbose::VERBOSITY_NORMAL);

            mpLocalMapper->RequestStop();

            while (!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished()) {
                usleep(1000);
            }

            unique_lock<mutex> lock(pActiveMap->mMutexMapUpdate);
            
            list<KeyFrame *> lpKFtoCheck(pActiveMap->mvpKeyFrameOrigins.begin(), pActiveMap->mvpKeyFrameOrigins.end());

            while (!lpKFtoCheck.empty()) {
                KeyFrame *pKF = lpKFtoCheck.front();
                const set<KeyFrame *> sChilds = pKF->GetChilds();
                
                Sophus::SE3f Twc = pKF->GetPoseInverse();
                
                for (set<KeyFrame *>::const_iterator sit = sChilds.begin(); sit != sChilds.end(); sit++) {
                    KeyFrame *pChild = *sit;
                    if (!pChild || pChild->isBad()) {
                        continue;
                    }

                    if (pChild->mnBAGlobalForKF != nLoopKF) {
                        Sophus::SE3f Tchildc = pChild->GetPose() * Twc;
                        
                        pChild->mTcwGBA = Tchildc * pKF->mTcwGBA; 

                        Sophus::SO3f Rcor = pChild->mTcwGBA.so3().inverse() * pChild->GetPose().so3();
                        if (pChild->isVelocitySet()) {
                            pChild->mVwbGBA = Rcor * pChild->GetVelocity();
                        } else {
                            Verbose::PrintMess("Child velocity empty!! ", Verbose::VERBOSITY_NORMAL);
                        }

                        pChild->mBiasGBA = pChild->GetImuBias();

                        pChild->mnBAGlobalForKF = nLoopKF;
                    }
                    lpKFtoCheck.push_back(pChild);
                }

                pKF->mTcwBefGBA = pKF->GetPose();
                
                pKF->SetPose(pKF->mTcwGBA);
                
                if (pKF->bImu) {
                    pKF->mVwbBefGBA = pKF->GetVelocity();
                    
                    pKF->SetVelocity(pKF->mVwbGBA);
                    pKF->SetNewBias(pKF->mBiasGBA);
                }

                lpKFtoCheck.pop_front();
            }

            const vector<MapPoint *> vpMPs = pActiveMap->GetAllMapPoints();

            for (size_t i = 0; i < vpMPs.size(); i++) {
                MapPoint *pMP = vpMPs[i];

                if (pMP->isBad()) {
                    continue;
                }

                if (pMP->mnBAGlobalForKF == nLoopKF) {
                    pMP->SetWorldPos(pMP->mPosGBA);
                } else {
                    KeyFrame *pRefKF = pMP->GetReferenceKeyFrame();

                    if (pRefKF->mnBAGlobalForKF != nLoopKF) {
                        continue;
                    }

                    Eigen::Vector3f Xc = pRefKF->mTcwBefGBA * pMP->GetWorldPos();

                    pMP->SetWorldPos(pRefKF->GetPoseInverse() * Xc);
                }
            }

            pActiveMap->InformNewBigChange();
            pActiveMap->IncreaseChangeIndex();

            mpLocalMapper->Release();

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndUpdateMap = std::chrono::steady_clock::now();

            double timeUpdateMap = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_EndUpdateMap - time_EndGBA).count();
            vdUpdateMap_ms.push_back(timeUpdateMap);

            double timeFGBA = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_EndUpdateMap - time_StartFGBA).count();
            vdFGBATotal_ms.push_back(timeFGBA);
#endif
            Verbose::PrintMess("Map updated!", Verbose::VERBOSITY_NORMAL);
        }

        mbFinishedGBA = true;
        mbRunningGBA = false;
    }
}

void CloudMerging::RequestFinish() {
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool CloudMerging::CheckFinish() {
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void CloudMerging::SetFinish() {
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool CloudMerging::isFinished() {
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

bool CloudMerging::isRunning() {
    return mbRunning;
}

Map *CloudMerging::GetEdgeFrontMap() {
    return mpCurrentEdgeFrontMap;
}

Map *CloudMerging::GetEdgeBackMap() {
    return mpCurrentEdgeBackMap;
}

Map *CloudMerging::GetCloudMap() {
    return mpCurrentCloudMap;
}

}
