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
private:
    ORB_SLAM3::System* mpSLAM;
    std::map<uint64_t, ORB_SLAM3::MapPoint*> mGlobalMapPoints;
    // 回调对象与深拷贝拦截缓存
    std::function<void(double, int)> mWatchdogCb;
    std::function<TrackingState()> mGetStateCb;
    std::vector<CachedMarginalizedData> mWarningBuffer;
    std::ofstream mFrontendTrajFile;
    // 分离出的原生注入逻辑层
    void ExecuteInjection(const MarginalizedData& data);
};

#endif // SVIN2_ORB_WRAPPER_H
