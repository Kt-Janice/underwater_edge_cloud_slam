#ifndef SVIN2_ORB_WRAPPER_H
#define SVIN2_ORB_WRAPPER_H

#include <Eigen/Dense>
#include <Eigen/StdVector> 
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <cstdint>
#include <functional>
#include <fstream> // [新增] 必须包含此头文件
#include <deque>   // [阶段2B修改] 前端位姿缓存队列
#include <mutex>   // [阶段2B修改] 前端位姿缓存互斥锁
#include <string>  // [阶段2B修改] 轨迹文件路径参数
#include "sophus/se3.hpp" // [阶段2B修改] 缓存 OKVIS / SVIn2 前端 Twc 位姿

// 前置声明 ORB-SLAM3 的核心类
namespace ORB_SLAM3 {
    class KeyFrame;
    class MapPoint;
    class System;
}

// 三态状态机枚举
enum class TrackingState {
    NORMAL,
    WARNING,
    LOST
};

// 承载单个3D点及其2D投影的结构体
struct ExtractedLandmark {
    uint64_t landmark_id;          
    float pt_x, pt_y, pt_z;        // 3D 点的纯净内存块
    float kp_u, kp_v;              // 2D 特征点的纯净内存块
};

// 承载整个被边缘化帧的结构体
struct MarginalizedData {
    double timestamp;              
    uint64_t frame_id;             
    float Tcw_mat[16];             // 4x4矩阵退化为 16 个 float 数组
    
    ExtractedLandmark* landmarks;  // 指向内存首地址的纯 C 指针
    size_t num_landmarks;          // 明确告诉后端数组有多长
};

// 安全深拷贝结构体，彻底切断底层 C 指针生命周期依赖
struct CachedMarginalizedData {
    double timestamp;
    uint64_t frame_id;
    float Tcw_mat[16];
    std::vector<ExtractedLandmark> landmarks_vec;

    MarginalizedData toRawData() {
        MarginalizedData raw_data;
        raw_data.timestamp = timestamp;
        raw_data.frame_id = frame_id;
        for (int i = 0; i < 16; ++i) {
            raw_data.Tcw_mat[i] = Tcw_mat[i];
        }
        raw_data.num_landmarks = landmarks_vec.size();
        raw_data.landmarks = landmarks_vec.data(); 
        return raw_data;
    }
};

// 跨系统数据中转与实例伪装层
class SVIn2ORBWrapper {
public:
    SVIn2ORBWrapper(ORB_SLAM3::System* pSLAM);
    ~SVIn2ORBWrapper();
    void RegisterStateCallbacks(std::function<void(double, int)> watchdog_cb, std::function<TrackingState()> get_state_cb);
    void InjectSVIn2MarginalizedData(const MarginalizedData& data);
    void InitTrajectorySaver(const std::string& path);
    void CloseTrajectorySaver();
    void InitOkvisFullTrajectorySaver(const std::string& path); // [阶段2B修改] 保存 fullStateCallback 输出的完整 OKVIS / SVIn2 前端轨迹
    void CloseOkvisFullTrajectorySaver(); // [阶段2B修改] 关闭完整 OKVIS / SVIn2 前端轨迹文件

    // [阶段2B修改] 缓存 OKVIS / SVIn2 前端连续位姿。
    // 该接口由 cloud_edge_demo.cpp 中的 fullStateCallback 调用，独立于 ORB-SLAM3 后端是否 LOST。
    void CacheFrontendPose(const double timestamp, const Sophus::SE3f &Twc);

    // [阶段2B修改] 按时间戳查询最近的 OKVIS / SVIn2 前端位姿。
    // CloudMerging.cc 用该接口构造 OKVIS-CloudMap 相邻运动一致性权重。
    bool GetNearestFrontendPose(const double timestamp, const double tolerance, Sophus::SE3f &Twc);

    // [CloudMap校正诊断] 按时间戳查询最近前端位姿，并返回匹配时间差，仅用于诊断导出。
    bool GetNearestFrontendPoseWithTimeGap(const double timestamp, const double tolerance, Sophus::SE3f &Twc, double &timeGap);
private:
    ORB_SLAM3::System* mpSLAM;
    std::map<uint64_t, ORB_SLAM3::MapPoint*> mGlobalMapPoints;
    // 回调对象与深拷贝拦截缓存
    std::function<void(double, int)> mWatchdogCb;
    std::function<TrackingState()> mGetStateCb;
    std::vector<CachedMarginalizedData> mWarningBuffer;
    std::ofstream mFrontendTrajFile;
    std::ofstream mOkvisFullTrajFile; // [阶段2B修改] okvis_full_traj.txt，记录 fullStateCallback 连续前端轨迹
    std::mutex mOkvisFullTrajFileMutex; // [阶段2B修改] 独立文件锁，避免与前端位姿缓存锁嵌套持有

    // [阶段2B修改] OKVIS / SVIn2 前端位姿缓存，存 Twc，与 KeyFrame::GetPoseInverse() 的方向保持一致。
    struct FrontendPoseData {
        double timestamp;
        Sophus::SE3f Twc;
    };

    std::deque<FrontendPoseData> mFrontendPoseBuffer;
    std::mutex mFrontendPoseMutex;
    size_t mnMaxFrontendPoseBufferSize = 5000;

    // 分离出的原生注入逻辑层
    void ExecuteInjection(const MarginalizedData& data);
};

#endif // SVIN2_ORB_WRAPPER_H
