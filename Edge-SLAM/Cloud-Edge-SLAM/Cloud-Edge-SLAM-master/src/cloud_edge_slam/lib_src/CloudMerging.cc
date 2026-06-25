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
#include "SVIn2ORBWrapper.h" // [阶段2B修改] 查询 OKVIS / SVIn2 前端位姿缓存

#include <Eigen/src/Core/ArithmeticSequence.h>
#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <exception>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <sstream>
#include <utility>
#include <limits> // [阶段2B修改] motion-arc / 前端一致性权重数值保护
#include <cmath>  // [阶段2B修改] std::isfinite
#include <map>    // [阶段2B修改] CloudMap 关键帧纠正权重表
#include <vector> // [阶段2B修改] CloudMap 关键帧排序缓存

//cap-udf
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/time.h> 

// [阶段2B修改] 全局 wrapper 指针定义在 SVIn2ORBWrapper.cpp 中。
// CloudMerging 通过该指针按时间戳查询 OKVIS / SVIn2 前端位姿。
extern SVIn2ORBWrapper* pSVIn2ORBWrapper;

namespace ORB_SLAM3 {

namespace {

// [阶段2B修改] CloudMap 首尾残差分配与 OKVIS / SVIn2 前端运动一致性权重参数。
constexpr double kCloudMergeRotationMotionWeight = 0.25;
constexpr double kCloudMergeMinMotionArc = 0.03;
constexpr double kFrontendPoseQueryTolerance = 0.05;
constexpr double kFrontendTranslationConsistencyWeight = 1.0;
constexpr double kFrontendRotationConsistencyWeight = 0.5;
constexpr double kFrontendMaxUncertaintyScale = 3.0;

// [阶段2B修改] 统一过滤无效 CloudMap KeyFrame，避免补偿阶段触碰空指针或 bad keyframe。
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

// [阶段2B修改] 退化场景下使用时间线性权重，保持原有首尾插值业务逻辑。
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

// [阶段2B修改] 用 CloudMap 运动弧长分配首尾残差，并在可用时叠加 OKVIS / SVIn2 前端位姿一致性权重。
bool BuildMotionArcCorrectionWeights(
    const std::vector<KeyFrame *> &vCloudKeyFrames,
    const double tStart,
    const double tEnd,
    std::map<KeyFrame *, double> &keyFrameCorrectionWeight,
    bool &bUsedFrontendConsistency) {
    keyFrameCorrectionWeight.clear();
    bUsedFrontendConsistency = false;

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

    if (vSortedCloudKeyFrames.size() < 2) {
        for (size_t i = 0; i < vSortedCloudKeyFrames.size(); i++) {
            KeyFrame *pKF = vSortedCloudKeyFrames[i];
            keyFrameCorrectionWeight[pKF] = ComputeTimeLinearCorrectionWeight(pKF->mTimeStamp, tStart, tEnd);
        }
        return false;
    }

    std::vector<double> vAccumulatedWeightedMotion(vSortedCloudKeyFrames.size(), 0.0);
    double totalWeightedMotion = 0.0;
    int nFrontendConsistencySegments = 0;

    for (size_t i = 1; i < vSortedCloudKeyFrames.size(); i++) {
        KeyFrame *pPrevKF = vSortedCloudKeyFrames[i - 1];
        KeyFrame *pCurrKF = vSortedCloudKeyFrames[i];

        Sophus::SE3f T_prev_Twc = pPrevKF->GetPoseInverse();
        Sophus::SE3f T_curr_Twc = pCurrKF->GetPoseInverse();

        const double translationMotion = static_cast<double>((T_curr_Twc.translation() - T_prev_Twc.translation()).norm());
        const Eigen::Quaternionf qPrev = T_prev_Twc.unit_quaternion();
        const Eigen::Quaternionf qCurr = T_curr_Twc.unit_quaternion();
        const double rotationMotion = static_cast<double>(qPrev.angularDistance(qCurr));

        double segmentMotion = translationMotion + kCloudMergeRotationMotionWeight * rotationMotion;
        if (!std::isfinite(segmentMotion)) {
            segmentMotion = 0.0;
        }

        if (segmentMotion < 0.0) {
            segmentMotion = 0.0;
        }

        double uncertaintyScale = 1.0;

        // [阶段2B修改] OKVIS / SVIn2 前端位姿只用于估计该段 CloudMap 运动的不确定度；
        // 查不到前端位姿时自动退回阶段2A motion-arc 权重。
        if (::pSVIn2ORBWrapper != nullptr) {
            Sophus::SE3f T_prev_frontend_Twc;
            Sophus::SE3f T_curr_frontend_Twc;

            const bool bFindPrevFrontendPose = ::pSVIn2ORBWrapper->GetNearestFrontendPose(
                pPrevKF->mTimeStamp,
                kFrontendPoseQueryTolerance,
                T_prev_frontend_Twc);

            const bool bFindCurrFrontendPose = ::pSVIn2ORBWrapper->GetNearestFrontendPose(
                pCurrKF->mTimeStamp,
                kFrontendPoseQueryTolerance,
                T_curr_frontend_Twc);

            if (bFindPrevFrontendPose && bFindCurrFrontendPose) {
                const Sophus::SE3f T_cloud_rel = T_prev_Twc.inverse() * T_curr_Twc;
                const Sophus::SE3f T_frontend_rel = T_prev_frontend_Twc.inverse() * T_curr_frontend_Twc;
                const Sophus::SE3f T_consistency_error = T_frontend_rel.inverse() * T_cloud_rel;

                const double translationConsistencyError =
                    static_cast<double>(T_consistency_error.translation().norm());
                const double rotationConsistencyError =
                    static_cast<double>(T_consistency_error.so3().log().norm());

                uncertaintyScale = 1.0
                    + kFrontendTranslationConsistencyWeight * translationConsistencyError
                    + kFrontendRotationConsistencyWeight * rotationConsistencyError;

                if (!std::isfinite(uncertaintyScale)) {
                    uncertaintyScale = 1.0;
                }

                if (uncertaintyScale < 1.0) {
                    uncertaintyScale = 1.0;
                }

                if (uncertaintyScale > kFrontendMaxUncertaintyScale) {
                    uncertaintyScale = kFrontendMaxUncertaintyScale;
                }

                nFrontendConsistencySegments++;
            }
        }

        const double segmentWeightedMotion = segmentMotion * uncertaintyScale;
        totalWeightedMotion += segmentWeightedMotion;
        vAccumulatedWeightedMotion[i] = totalWeightedMotion;
    }

    if (totalWeightedMotion <= kCloudMergeMinMotionArc) {
        for (size_t i = 0; i < vSortedCloudKeyFrames.size(); i++) {
            KeyFrame *pKF = vSortedCloudKeyFrames[i];
            keyFrameCorrectionWeight[pKF] = ComputeTimeLinearCorrectionWeight(pKF->mTimeStamp, tStart, tEnd);
        }
        bUsedFrontendConsistency = false;
        return false;
    }

    bUsedFrontendConsistency = nFrontendConsistencySegments > 0;

    for (size_t i = 0; i < vSortedCloudKeyFrames.size(); i++) {
        KeyFrame *pKF = vSortedCloudKeyFrames[i];
        double weight = 0.0;

        if (pKF->mTimeStamp <= tStart) {
            weight = 0.0;
        } else if (pKF->mTimeStamp >= tEnd) {
            weight = 1.0;
        } else {
            weight = vAccumulatedWeightedMotion[i] / totalWeightedMotion;
        }

        if (!std::isfinite(weight)) {
            weight = ComputeTimeLinearCorrectionWeight(pKF->mTimeStamp, tStart, tEnd);
        }

        if (weight < 0.0) {
            weight = 0.0;
        }

        if (weight > 1.0) {
            weight = 1.0;
        }

        keyFrameCorrectionWeight[pKF] = weight;
    }

    if (bUsedFrontendConsistency) {
        cerr << "\033[1;32m[CloudMerging] UDF correction uses frontend consistency on "
             << nFrontendConsistencySegments << " motion segments."
             << " query tolerance = " << kFrontendPoseQueryTolerance
             << ", max uncertainty scale = " << kFrontendMaxUncertaintyScale
             << "\033[0m" << endl;
    } else {
        cerr << "\033[1;33m[CloudMerging] Frontend pose buffer unavailable for this CloudMap; keep stage2A motion-arc weights.\033[0m" << endl;
    }

    return true;
}

} // namespace

CloudMerging::CloudMerging(Atlas *pAtlas, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, const bool bFixScale, const bool bActiveLC, const bool bWork, const bool bMergeAnyway, MapDrawer *pMapDrawer, FrameDrawer *pFrameDrawer, const bool bOldUdf, const bool bNewUdf) :
    mbResetRequested(false), mbResetActiveMapRequested(false), mbFinishRequested(false), mbFinished(true), mpAtlas(pAtlas),
    mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpMatchedKF(NULL), mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
    mbStopGBA(false), mpThreadGBA(nullptr), mbFixScale(bFixScale), mnFullBAIdx(0), mnMergeNumCoincidences(0),
    mbMergeDetected(false), mnMergeNumNotFound(0), mbActiveCM(bActiveLC), mbMergeAnyway(bMergeAnyway), mbWork(bWork),
    mpMapDrawer(pMapDrawer), mpFrameDrawer(pFrameDrawer), mbOldUdf(bOldUdf), mbNewUdf(bNewUdf) {
    
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

                                        // [阶段2B修改] 构造 CloudMap 首尾残差分配权重：
                                        // 优先使用 motion-arc，并在 OKVIS / SVIn2 前端位姿可用时叠加一致性权重。
                                        std::map<KeyFrame *, double> cloudKeyFrameCorrectionWeight;
                                        bool bUsedFrontendConsistency = false;
                                        const bool bUseMotionArcWeight = BuildMotionArcCorrectionWeights(
                                            cloudMapKeyFrames,
                                            t_start,
                                            t_end,
                                            cloudKeyFrameCorrectionWeight,
                                            bUsedFrontendConsistency);

                                        if (bUseMotionArcWeight) {
                                            cerr << "\033[1;32m[CloudMerging] UDF correction uses motion-arc weights. "
                                                 << "rotation weight = " << kCloudMergeRotationMotionWeight;
                                            if (bUsedFrontendConsistency) {
                                                cerr << ", frontend consistency = enabled";
                                            } else {
                                                cerr << ", frontend consistency = disabled";
                                            }
                                            cerr << "\033[0m" << endl;
                                        } else {
                                            cerr << "\033[1;33m[CloudMerging] UDF correction falls back to time-linear weights because motion arc is degenerate.\033[0m" << endl;
                                        }

                                        // [阶段2B修改] 查询单个 KeyFrame 的补偿权重；缺失时退回时间线性权重。
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
                                                // [阶段2B修改] 使用前端位姿一致性加权后的 CloudMap 补偿权重。
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
                                                        // [阶段2B修改] MapPoint 跟随其参考 KeyFrame 使用同一补偿权重。
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
                                        cerr << "\033[1;32m[CloudMerging] Sim3 Lie-Algebra Manifold Smoothing (With Base Alignment) Successfully Applied!\033[0m" << endl;
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

            if (mbOldUdf || mbNewUdf) {
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
            if (bComputeEdgeFront || mbMergeAnyway) {
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
