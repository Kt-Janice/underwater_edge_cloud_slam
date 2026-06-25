#include "SVIn2ORBWrapper.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "System.h"
#include "Atlas.h"
#include "LocalMapping.h"
#include <iostream> 
#include <fstream>
#include <iomanip>
#include <set>
#include <set>
#include <mutex>
#include <limits> // [阶段2B修改] 前端位姿最近邻查询
#include <cmath>  // [阶段2B修改] std::abs / std::isfinite

// 声明全局指针，供 SVIn2(OKVIS) 的 Estimator.cpp 访问
SVIn2ORBWrapper* pSVIn2ORBWrapper = nullptr;

SVIn2ORBWrapper::SVIn2ORBWrapper(ORB_SLAM3::System* pSLAM) {
    mpSLAM = pSLAM;
}

SVIn2ORBWrapper::~SVIn2ORBWrapper() {
}

// [阶段2B修改] 缓存 OKVIS / SVIn2 前端连续 Twc 位姿。
// 注意：该函数应由 fullStateCallback 调用，而不是只在 ExecuteInjection() 中调用。
// 原因是 ORB-SLAM3 后端 LOST 时 ExecuteInjection() 可能不执行，但 OKVIS 前端仍可能连续输出位姿。
void SVIn2ORBWrapper::CacheFrontendPose(const double timestamp, const Sophus::SE3f &Twc) {
    if (!std::isfinite(timestamp)) {
        return;
    }

    const Eigen::Vector3f translation = Twc.translation();
    const Eigen::Matrix3f rotation = Twc.rotationMatrix();

    if (!translation.allFinite()) {
        return;
    }

    if (!rotation.allFinite()) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mFrontendPoseMutex);

        FrontendPoseData data;
        data.timestamp = timestamp;
        data.Twc = Twc;
        mFrontendPoseBuffer.push_back(data);

        while (mFrontendPoseBuffer.size() > mnMaxFrontendPoseBufferSize) {
            mFrontendPoseBuffer.pop_front();
        }
    }

    {
        std::lock_guard<std::mutex> lock(mOkvisFullTrajFileMutex);

        if (mOkvisFullTrajFile.is_open()) {
            Eigen::Quaternionf q_wc(rotation);
            q_wc.normalize();

            mOkvisFullTrajFile << std::fixed << std::setprecision(9)
                               << timestamp << " "
                               << translation.x() << " "
                               << translation.y() << " "
                               << translation.z() << " "
                               << q_wc.x() << " "
                               << q_wc.y() << " "
                               << q_wc.z() << " "
                               << q_wc.w() << std::endl;
        }
    }
}

// [阶段2B修改] 根据时间戳查询最近的 OKVIS / SVIn2 前端 Twc 位姿。
// 如果缓存为空或最近时间差超过 tolerance，则返回 false，CloudMerging 会自动退回阶段2A运动弧长权重。
bool SVIn2ORBWrapper::GetNearestFrontendPose(const double timestamp, const double tolerance, Sophus::SE3f &Twc) {
    std::lock_guard<std::mutex> lock(mFrontendPoseMutex);

    if (mFrontendPoseBuffer.empty()) {
        return false;
    }

    double bestDeltaTime = std::numeric_limits<double>::max();
    int bestIndex = -1;

    for (size_t i = 0; i < mFrontendPoseBuffer.size(); i++) {
        const double deltaTime = std::abs(mFrontendPoseBuffer[i].timestamp - timestamp);
        if (deltaTime < bestDeltaTime) {
            bestDeltaTime = deltaTime;
            bestIndex = static_cast<int>(i);
        }
    }

    if (bestIndex < 0) {
        return false;
    }

    if (bestDeltaTime > tolerance) {
        return false;
    }

    Twc = mFrontendPoseBuffer[bestIndex].Twc;
    return true;
}

// 回调绑定实现
void SVIn2ORBWrapper::RegisterStateCallbacks(std::function<void(double, int)> watchdog_cb, std::function<TrackingState()> get_state_cb) {
    mWatchdogCb = watchdog_cb;
    mGetStateCb = get_state_cb;
}

void SVIn2ORBWrapper::InjectSVIn2MarginalizedData(const MarginalizedData& data) {
    // 异常数据拦截
    if (data.num_landmarks == 0 || data.num_landmarks > 50000) {
        std::cout << "\033[1;33m[SVIn2Wrapper] Warning: 边缘化数据异常 (Size: " << data.num_landmarks << ")，已触发防御机制，丢弃该帧。\033[0m" << std::endl;
        return;
    }
    for (int i = 0; i < 16; i++) {
        if (std::isnan(data.Tcw_mat[i]) || std::isinf(data.Tcw_mat[i]) || std::abs(data.Tcw_mat[i]) > 1e6) {
            std::cout << "\033[1;31m[SVIn2Wrapper] 警告: 边缘化位姿发散或出现 NaN，抛弃该帧！\033[0m" << std::endl;
            return;
        }
    }

    if (mWatchdogCb) mWatchdogCb(data.timestamp, data.num_landmarks);

    TrackingState state = TrackingState::NORMAL;
    if (mGetStateCb) state = mGetStateCb();

    // 记录上一帧的状态，用于判断状态跃变
    static TrackingState last_state = TrackingState::NORMAL; 

    if (state == TrackingState::NORMAL) {
        if (mWarningBuffer.size() > 0) {
            for (size_t i = 0; i < mWarningBuffer.size(); ++i) {
                this->ExecuteInjection(mWarningBuffer[i].toRawData());
            }
            mWarningBuffer.clear();
        }
        this->ExecuteInjection(data);
    } 
    else if (state == TrackingState::WARNING) {
        CachedMarginalizedData safe_data;
        safe_data.timestamp = data.timestamp;
        safe_data.frame_id = data.frame_id;
        for (int i = 0; i < 16; ++i) safe_data.Tcw_mat[i] = data.Tcw_mat[i];
        for (size_t i = 0; i < data.num_landmarks; ++i) safe_data.landmarks_vec.push_back(data.landmarks[i]);
        mWarningBuffer.push_back(safe_data);
    } 
    else if (state == TrackingState::LOST) {
        mWarningBuffer.clear();
        // 仅在从 NORMAL/WARNING 跌入 LOST 的“第一帧”时，才进行切图！
        if (last_state != TrackingState::LOST) {
            mpSLAM->GetAtlas()->CreateNewMap();
            std::cout << "\033[1;35m[SVIn2Wrapper] 已成功切断拓扑并创建新子图！等待系统恢复...\033[0m" << std::endl;
        }
    }

    // 更新历史状态
    last_state = state; 
}

void SVIn2ORBWrapper::ExecuteInjection(const MarginalizedData& data) {
    // 在不申请任何大块内存、绝对不加锁的情况下，极速判断是否需要丢弃该帧！
    Eigen::Map<const Eigen::Matrix4f> Tcw_mat_mapped(data.Tcw_mat);
    Eigen::Matrix3f R_cw = Tcw_mat_mapped.block<3, 3>(0, 0);
    Eigen::Vector3f t_cw = Tcw_mat_mapped.block<3, 1>(0, 3);
    Eigen::Quaternionf q_cw(R_cw);
    q_cw.normalize();
    Sophus::SE3f Tcw_sophus(q_cw, t_cw);

    Sophus::SE3f current_Twc = Tcw_sophus.inverse(); // 当前相机的世界坐标系位姿

    if (mFrontendTrajFile.is_open()) {
        Eigen::Vector3f twc = current_Twc.translation();
        Eigen::Quaternionf q_wc(current_Twc.rotationMatrix());
        
        mFrontendTrajFile << std::fixed << std::setprecision(6) << data.timestamp << " "
                        << twc.x() << " " << twc.y() << " " << twc.z() << " "
                        << q_wc.x() << " " << q_wc.y() << " " << q_wc.z() << " " << q_wc.w() << std::endl;
    }
    
    static Sophus::SE3f last_accepted_Twc; // 记录上一次成功注入的世界坐标系位姿
    static bool is_first_frame = true;
    
    if (!is_first_frame) {
        // 1. 计算平移差值 (米)
        float delta_trans = (current_Twc.translation() - last_accepted_Twc.translation()).norm();
        
        // 2. 计算旋转差值 (使用 Sophus 李代数对数映射，自带防 NaN 保护)
        Sophus::SO3f delta_R = last_accepted_Twc.so3().inverse() * current_Twc.so3();
        float delta_rot_degree = delta_R.log().norm() * 180.0 / M_PI;

        // 设置物理阈值（可根据你的海下运动速度调节，建议：平移 0.1米，旋转 3度）
        const float THRESHOLD_TRANS = 0.1f;  
        const float THRESHOLD_ROT = 3.0f;    

        if (delta_trans < THRESHOLD_TRANS && delta_rot_degree < THRESHOLD_ROT) {
            return; // return，不触碰任何锁，不浪费 CPU 资源
        }
    }
    
    // 更新最后一次接受的帧位姿
    last_accepted_Twc = current_Twc;
    is_first_frame = false;
    // =========================================================================

    // 确认接受该帧后，才开始输出日志并分配内存
    std::cout << "\033[1;32m ----------------------------------------------------------------\033[0m" << std::endl;
    std::cout << "\033[1;32m Received OKVIS edge-shifting keyframe! Timestamp: \033[0m" << data.timestamp << std::endl;

    // 1. 从 C 数组还原 2D 坐标，排爆 NaN
    std::vector<cv::KeyPoint> keypointsUn;
    keypointsUn.reserve(data.num_landmarks); // 提前分配内存，加速 push_back
    for (size_t i = 0; i < data.num_landmarks; i++) {
        float u = data.landmarks[i].kp_u;
        float v = data.landmarks[i].kp_v;
        if (std::isnan(u) || std::isnan(v) || std::isinf(u) || std::isinf(v)) { 
            u = 0.0f; v = 0.0f; 
        }
        keypointsUn.push_back(cv::KeyPoint(u, v, 1.0f));
    }

    // 2. 伪造全 0 描述子
    cv::Mat fake_descriptors = cv::Mat::zeros(data.num_landmarks, 32, CV_8U);

    ORB_SLAM3::Map* pCurrentMap = mpSLAM->GetAtlas()->GetCurrentMap();
    ORB_SLAM3::KeyFrameDatabase* pKFDB = mpSLAM->GetKeyFrameDatabase();
    ORB_SLAM3::GeometricCamera* pCamera = mpSLAM->GetCamera();

    // =========================================================================
    // 直到此时，真正需要修改地图拓扑时，才去抢占互斥锁！
    std::unique_lock<std::mutex> lock(pCurrentMap->mMutexMapUpdate);
    
    auto pVocabulary = mpSLAM->GetVocabulary();
    auto pExtractor = mpSLAM->GetExtractor();
    auto distCoef = mpSLAM->GetDistCoef();
    auto bf = mpSLAM->Getbf();
    auto thDepth = mpSLAM->GetThDepth();

    // 假图像注入
    cv::Mat fake_img = cv::Mat::zeros(260, 346, CV_8UC1);
    cv::rectangle(fake_img, cv::Point(150, 100), cv::Point(200, 150), cv::Scalar(255), -1);

    // 调用标准构造函数，完成所有的底层分配和安全边界计算
    ORB_SLAM3::Frame frame(fake_img, data.timestamp, pExtractor, pVocabulary, pCamera, distCoef, bf, thDepth);
    
    // 把假图像提取出的数据，全部覆盖为我们真实的 3D 点云
    frame.N = data.num_landmarks;
    frame.mvKeys = keypointsUn;
    frame.mvKeysUn = keypointsUn;
    frame.mDescriptors = fake_descriptors;
    frame.mvuRight.assign(data.num_landmarks, -1.0f);
    frame.mvDepth.assign(data.num_landmarks, -1.0f);
    frame.mvpMapPoints.assign(data.num_landmarks, nullptr);
    frame.mvbOutlier.assign(data.num_landmarks, false); 
    
    // 重建空间网格（清除假特征占用的网格，重新分配到 346x260 的安全网格中）
    for(int i=0; i<FRAME_GRID_COLS; i++) {
        for(int j=0; j<FRAME_GRID_ROWS; j++) {
            frame.mGrid[i][j].clear();
        }
    }
    for(int i = 0; i < frame.N; i++) {
        const cv::KeyPoint &kp = frame.mvKeysUn[i];
        int nGridPosX, nGridPosY;
        if(frame.PosInGrid(kp, nGridPosX, nGridPosY)) {
            frame.mGrid[nGridPosX][nGridPosY].push_back(i);
        }
    }

    // 生成 KeyFrame
    ORB_SLAM3::KeyFrame* pKF = new ORB_SLAM3::KeyFrame(frame, pCurrentMap, pKFDB);
    pKF->SetPose(Tcw_sophus);
    
    // 禁止局部图优化进行特征匹配
    pKF->mbIsFakeDescriptor = true; 

    int new_pts = 0;
    int tracked_pts = 0;

    std::vector<ORB_SLAM3::MapPoint*> vpAllMPs = pCurrentMap->GetAllMapPoints();
    std::set<ORB_SLAM3::MapPoint*> safe_mps(vpAllMPs.begin(), vpAllMPs.end());
    
    // ABA 幽灵重分配雷达
    static std::map<uint64_t, long unsigned int> mGlobalMapPoints_mnId;

    // 硬绑定 3D MapPoint
    for (size_t i = 0; i < data.num_landmarks; i++) {
        float px = data.landmarks[i].pt_x;
        float py = data.landmarks[i].pt_y;
        float pz = data.landmarks[i].pt_z;
        
        // 3D NaN 拦截
        if (std::isnan(px) || std::isnan(py) || std::isnan(pz) || 
            std::isinf(px) || std::isinf(py) || std::isinf(pz) ||
            std::abs(px) > 1e6 || std::abs(py) > 1e6 || std::abs(pz) > 1e6) {
            continue; 
        }

        uint64_t lm_id = data.landmarks[i].landmark_id;
        ORB_SLAM3::MapPoint* pMP = nullptr;

        // 查询缓存历史点
        if (mGlobalMapPoints.count(lm_id) > 0) {
            ORB_SLAM3::MapPoint* cached_pMP = mGlobalMapPoints[lm_id];
            long unsigned int expected_mnId = mGlobalMapPoints_mnId[lm_id];
            
            // 双重校验：不仅要在安全名单里，它内部的 ID 还必须没变过！
            if (safe_mps.count(cached_pMP) > 0 && cached_pMP->mnId == expected_mnId && !cached_pMP->isBad()) {
                pMP = cached_pMP;
                tracked_pts++;
            }
        }

        // 如果是全新点，或旧点已经被干掉
        if (pMP == nullptr) {
            Eigen::Vector3f pos3d(px, py, pz);
            pMP = new ORB_SLAM3::MapPoint(pos3d, pKF, pCurrentMap);
            mGlobalMapPoints[lm_id] = pMP; 
            mGlobalMapPoints_mnId[lm_id] = pMP->mnId; // 记录出生 ID，防止借尸还魂
            pCurrentMap->AddMapPoint(pMP);
            new_pts++;
        }

        // 核心拓扑注入
        pMP->AddObservation(pKF, i);
        pKF->AddMapPoint(pMP, i);
        pMP->UpdateNormalAndDepth();
        
        // 强制计算描述子，防图优化死机
        pMP->ComputeDistinctiveDescriptors();
    }

    // 5. 将处理好的伪装帧压入地图与局部建图线程
    pCurrentMap->AddKeyFrame(pKF);
    mpSLAM->GetLocalMapping()->InsertKeyFrame(pKF);

    // 日志输出
    std::cout << "\033[1;32m Successfully injected a fake keyframe into LocalMapping! (ID: " << pKF->mnId << ")\033[0m" << std::endl;
    std::cout << "\033[1;32m Skeleton point cloud statistics -> Total " << data.num_landmarks <<"\n"
              << " | New points:" << new_pts 
              << " | Shared Historical Perspective: " << tracked_pts << "\033[0m" << std::endl;
    std::cout << "\033[1;32m ----------------------------------------------------------------\033[0m\n" << std::endl;
}
void SVIn2ORBWrapper::InitTrajectorySaver(const std::string& path) {
    mFrontendTrajFile.open(path, std::ios::out);
    if (mFrontendTrajFile.is_open()) {
        std::cout << "\033[1;32m[Wrapper] Successfully opened injection candidate trajectory file: "
                  << path << "\033[0m" << std::endl;
    } else {
        std::cout << "\033[1;31m[Wrapper] Failed to open injection candidate trajectory file!\033[0m"
                  << std::endl;
    }
}

// [阶段2B修改] 初始化完整 OKVIS / SVIn2 前端轨迹文件，数据来自 fullStateCallback -> CacheFrontendPose()。
void SVIn2ORBWrapper::InitOkvisFullTrajectorySaver(const std::string& path) {
    std::lock_guard<std::mutex> lock(mOkvisFullTrajFileMutex);

    if (mOkvisFullTrajFile.is_open()) {
        mOkvisFullTrajFile.close();
    }

    mOkvisFullTrajFile.open(path, std::ios::out);

    if (mOkvisFullTrajFile.is_open()) {
        std::cout << "\033[1;32m[Wrapper] Successfully opened OKVIS full trajectory file: "
                  << path << "\033[0m" << std::endl;

        mOkvisFullTrajFile << "# timestamp tx ty tz qx qy qz qw" << std::endl;
    } else {
        std::cout << "\033[1;31m[Wrapper] Failed to open OKVIS full trajectory file!\033[0m"
                  << std::endl;
    }
}

void SVIn2ORBWrapper::CloseTrajectorySaver() {
    if (mFrontendTrajFile.is_open()) {
        mFrontendTrajFile.close();
    }
}

// [阶段2B修改] 关闭完整 OKVIS / SVIn2 前端轨迹文件。
void SVIn2ORBWrapper::CloseOkvisFullTrajectorySaver() {
    std::lock_guard<std::mutex> lock(mOkvisFullTrajFileMutex);

    if (mOkvisFullTrajFile.is_open()) {
        mOkvisFullTrajFile.close();
    }
}
