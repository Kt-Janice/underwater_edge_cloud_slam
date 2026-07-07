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
#include "KeyFrame.h"
#include "sensor_msgs/image_encodings.h"
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <opencv2/imgcodecs.hpp>
#include <sys/stat.h> 
#include <sys/types.h> 

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/Int16.h>
#include <std_msgs/Int64.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float64.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Quaternion.h>
#include <actionlib/client/simple_action_client.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/core.hpp>
#include <string>
#include <unistd.h>
#include <vector>

#include "System.h"
#include "Map.h"
#include "Atlas.h"
#include "Converter.h"
#include "ORBVocabulary.h"
#include "CloudImageSampler.h"

#include "cloud_edge_slam/Evo.h"
#include "cloud_edge_slam/CloudSlamAction.h"
#include "cloud_edge_slam/CloudSlamActionFeedback.h"
#include "cloud_edge_slam/CloudSlamActionGoal.h"
#include "cloud_edge_slam/CloudSlamGoal.h"
#include "cloud_edge_slam/CloudSlamResult.h"
#include "cloud_edge_slam/Sequence.h"
#include "cloud_edge_slam/CloudMap.h"
#include "cloud_edge_slam/KeyFrame.h"
#include "cloud_edge_slam/KeyPoint.h"
#include "cloud_edge_slam/MapPoint.h"
#include "cloud_edge_slam/Descriptor.h"
#include "cloud_edge_slam/Observation.h"
#include "ros/init.h"
#include "ros/message_traits.h"
#include "ros/node_handle.h"
#include "ros/publisher.h"
#include "ros/subscriber.h"
#include "ros/time.h"
#include "rosbag/query.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"
//cap-udf
#include<fstream> 
#include "SVIn2ORBWrapper.h"
#include <thread>
#include <memory> // [阶段2B修改] fullStateCallback 中使用 std::shared_ptr

// [新增 OS 级堆栈捕获探针头文件]
#include <signal.h>
#include <execinfo.h>
#include <unistd.h>



// 声明外部全局指针，连接 OKVIS 到本文件内的 ORB-SLAM3 实例
extern SVIn2ORBWrapper* pSVIn2ORBWrapper;

// 引入 OKVIS 依赖头文件及初始化绑定函数
#include <okvis/Publisher.hpp>
#include <okvis/RosParametersReader.hpp>
#include <okvis/Subscriber.hpp>
#include <okvis/ThreadedKFVio.hpp>

bool is_reloc = true;

class Grabber;

namespace okvis {
    void initEstimator(ThreadedKFVio* okvis_estimator, Publisher* publisher, VioParameters& parameters, Grabber* igb);
}

//简化在ROS程序中获取参数的过程，并在获取失败时提供错误处理
#define getRosParam(nh, param_name, output)                                     \
    if (!nh.getParam(param_name, output)) {                                     \
        ROS_ERROR_STREAM("!ERROR! cannot get necessary param: " << param_name); \
        exit(100);                                                              \
    }

namespace bfs = boost::filesystem;
using namespace std;

typedef actionlib::SimpleActionClient<cloud_edge_slam::CloudSlamAction> CloudClient;
// typedef actionlib::SimpleActionClient<actionlib_tutorials::FibonacciAction> CloudClient;

// [轨迹导出修改] 用于统一收集 Atlas 中的 KeyFrame 轨迹记录。
struct TrajectoryRecord {
    double timestamp;
    Eigen::Vector3f position;
    Eigen::Quaternionf quaternion;
    unsigned long keyFrameId;
    unsigned long mapId;
    bool isCloud;
};

// [轨迹导出修改] TUM 轨迹导出过滤模式。
static constexpr int kTrajectoryFilterAll = 0;
static constexpr int kTrajectoryFilterCloudOnly = 1;
static constexpr int kTrajectoryFilterEdgeOnly = 2;

// [轨迹导出修改] 从指定 Map 中收集所有有效 KeyFrame 的 Twc 轨迹记录。
static void CollectTrajectoryRecordsFromMap(
    ORB_SLAM3::Map *pMap,
    std::vector<TrajectoryRecord> &records) {
    records.clear();

    if (pMap == nullptr) {
        return;
    }

    std::vector<ORB_SLAM3::KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    for (size_t keyFrameIndex = 0; keyFrameIndex < vpKFs.size(); keyFrameIndex++) {
        ORB_SLAM3::KeyFrame *pKF = vpKFs[keyFrameIndex];
        if (pKF == nullptr) {
            continue;
        }

        if (pKF->isBad()) {
            continue;
        }

        Sophus::SE3f Twc = pKF->GetPoseInverse();
        TrajectoryRecord record;
        record.timestamp = pKF->mTimeStamp;
        record.position = Twc.translation();
        record.quaternion = Twc.unit_quaternion();
        record.quaternion.normalize();
        record.keyFrameId = pKF->mnId;
        record.mapId = pMap->GetId();
        record.isCloud = pKF->isCloud();
        records.push_back(record);
    }
}

// [轨迹导出修改] 从 Atlas 中收集所有 KeyFrame 的 Twc 轨迹记录。
static void CollectTrajectoryRecordsFromAtlas(
    ORB_SLAM3::Atlas *pAtlas,
    std::vector<TrajectoryRecord> &records) {
    records.clear();

    if (pAtlas == nullptr) {
        return;
    }

    std::vector<TrajectoryRecord> mapRecords;
    std::vector<ORB_SLAM3::Map*> vpMaps = pAtlas->GetAllMaps();
    for (size_t mapIndex = 0; mapIndex < vpMaps.size(); mapIndex++) {
        ORB_SLAM3::Map *pMap = vpMaps[mapIndex];
        if (pMap == nullptr) {
            continue;
        }

        CollectTrajectoryRecordsFromMap(pMap, mapRecords);
        for (size_t recordIndex = 0; recordIndex < mapRecords.size(); recordIndex++) {
            records.push_back(mapRecords[recordIndex]);
        }
    }
}

// [轨迹导出修改] 按 timestamp 升序排序；同 timestamp 使用 KeyFrame id 保证稳定顺序。
static void SortTrajectoryRecords(std::vector<TrajectoryRecord> &records) {
    std::sort(records.begin(), records.end(), [](const TrajectoryRecord &left, const TrajectoryRecord &right) {
        if (left.timestamp == right.timestamp) {
            return left.keyFrameId < right.keyFrameId;
        }

        return left.timestamp < right.timestamp;
    });
}

// [轨迹导出修改] 统计指定过滤模式下会写出的记录数，避免无 cloud KeyFrame 的 Map 生成空文件。
static size_t CountTrajectoryRecords(
    const std::vector<TrajectoryRecord> &records,
    const int filterMode) {
    size_t recordCount = 0;
    for (size_t i = 0; i < records.size(); i++) {
        const TrajectoryRecord &record = records[i];

        if (filterMode == kTrajectoryFilterCloudOnly) {
            if (!record.isCloud) {
                continue;
            }
        }

        if (filterMode == kTrajectoryFilterEdgeOnly) {
            if (record.isCloud) {
                continue;
            }
        }

        recordCount++;
    }

    return recordCount;
}

// [轨迹导出修改] 保存 TUM 格式轨迹，支持 all / cloud-only / edge-only 三种过滤模式。
static size_t SaveTrajectoryRecordsTUM(
    const std::vector<TrajectoryRecord> &records,
    const std::string &filename,
    const int filterMode) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cout << "\033[1;31m[Trajectory Export] Failed to open trajectory file: "
                  << filename << "\033[0m" << std::endl;
        return 0;
    }

    std::vector<TrajectoryRecord> sortedRecords = records;
    SortTrajectoryRecords(sortedRecords);

    size_t savedRecordCount = 0;
    for (size_t i = 0; i < sortedRecords.size(); i++) {
        const TrajectoryRecord &record = sortedRecords[i];
        if (filterMode == kTrajectoryFilterCloudOnly) {
            if (!record.isCloud) {
                continue;
            }
        }

        if (filterMode == kTrajectoryFilterEdgeOnly) {
            if (record.isCloud) {
                continue;
            }
        }

        ofs << std::fixed << std::setprecision(9)
            << record.timestamp << " "
            << record.position.x() << " "
            << record.position.y() << " "
            << record.position.z() << " "
            << record.quaternion.x() << " "
            << record.quaternion.y() << " "
            << record.quaternion.z() << " "
            << record.quaternion.w() << std::endl;
        savedRecordCount++;
    }

    ofs.close();
    return savedRecordCount;
}

class Grabber {
public:
    Grabber(ORB_SLAM3::System *pSLAM) :
        mpSLAM(pSLAM) {
    }
    // **********************************************
    // @note init
    void SetParameters(bool bWaitCloudResult, float nMainLoopSleep, string savePath, bool bSaveCloudBag, bool bOldUdf, bool bNewUdf);
    void SetNodeHandle(ros::NodeHandle *pNodeHandle);
    void SetOrbMapPublisher(ros::Publisher *pPublisher);
    void SetCloudImagesActionClient(CloudClient *pCloudImagesActionClient);
    void SetImageTopicPublisher(ros::Publisher *pPublisher);
    // **********************************************

    // **********************************************
    // @note Main Cloud Interact
    // Pub Cloud Images
    void TrackImage(const cv::Mat &img, const double &timestamp, const float &imageScale);
    // Get Cloud Map
    void ActionFinishCb(const actionlib::SimpleClientGoalState &state, const cloud_edge_slam::CloudSlamResultConstPtr &result);
    // Get Memory
    void MemoryCb(const std_msgs::Float32ConstPtr &msg);
    // **********************************************

    // **********************************************
    // @note For Debug
    // Subscriber Callback: Pub Specify ORB Map
    void PubORBMapCb(const std_msgs::Int16ConstPtr &msg);
    // Subscriber Callback: Save Specify ORB Map
    void SaveORBMapCb(const std_msgs::Int16ConstPtr &msg);
    // Subscriber Callback: Read Cloud Map
    void GrabCloudMapCb(const cloud_edge_slam::CloudMapConstPtr &msg);
    void RawImageCb(const sensor_msgs::ImageConstPtr &msg);
    void TrackingWatchdog(double timestamp, int num_landmarks);
    void UploadLostImages(double start_time, double end_time);
    

    // **********************
    // **********************************************

    // **********************************************
    // @note utils
    static cloud_edge_slam::CloudMap ORBMapToROSMap(ORB_SLAM3::Map *pMap);
    ORB_SLAM3::Map *ROSMapToORBMap(cloud_edge_slam::CloudMapConstPtr pMap);
    static void WriteCloudMapBag(cloud_edge_slam::CloudMap map, const std::string &save_path);
    static void WriteCloudImagesBag(std::vector<ORB_SLAM3::CloudImage> &vImages, const std::string &save_path);
    static void WriteSeqBag(const cloud_edge_slam::Sequence &seq, const std::string &save_path);
    // **********************************************
    TrackingState GetTrackingState() {
        std::lock_guard<std::mutex> lock(mStateMutex);
        return current_state_;
    }

public:
    cloud_edge_slam::CloudSlamGoal goal;
    ORB_SLAM3::System *mpSLAM;

    ros::NodeHandle *mpNodeHandler;
    ros::Publisher *mpOrbMapPub;
    ros::Subscriber *mpMemorySub;
    ros::Publisher *mpImageTopicPub;
    CloudClient *mpCloudImagesActionClient = NULL;

    std::vector<float> mvMemory;
    std::map<double, float> mvBagSize;
    std::map<double, float> mvNoSamplingBagSize;
    std::map<double, cv::Mat> mImageBuffer;
    std::mutex mBufMutex;

    bool mbWaitCloudResult;
    std::string mSaveDir;
    float mnMainLoopSleep;
    bool mbSaveCloudBag;
    bool mbOldUdf; //cap-udf
    bool mbNewUdf; //cap-udf
private:
    // 状态机与滑动窗口私有变量
    std::mutex mStateMutex;
    std::deque<int> inliers_window_;
    TrackingState current_state_ = TrackingState::NORMAL;
    double warning_start_time_ = 0.0;
};


namespace okvis {
    void initEstimator(ThreadedKFVio* okvis_estimator, Publisher* publisher, VioParameters& parameters, Grabber* igb) {
        publisher->setParameters(parameters);
    
        // [阶段2B修改] fullStateCallback 不再只转发给 Publisher，同时缓存 OKVIS / SVIn2 前端连续 Twc 位姿。
        // 该缓存供 CloudMerging 阶段2B构造“OKVIS-CloudMap 相邻运动一致性权重”。
        okvis_estimator->setFullStateCallback(
            [publisher, &parameters](const okvis::Time& t,
                                     const okvis::kinematics::Transformation& T_WS,
                                     const Eigen::Matrix<double, 9, 1>& speedAndBiases,
                                     const Eigen::Matrix<double, 3, 1>& omega_S,
                                     const okvis::kinematics::Transformation& driftCorrected_T_WS) {
                publisher->publishFullStateAsCallback(
                    t,
                    T_WS,
                    speedAndBiases,
                    omega_S,
                    driftCorrected_T_WS);

                if (pSVIn2ORBWrapper != nullptr) {
                    std::shared_ptr<const okvis::kinematics::Transformation> T_SC_ptr =
                        parameters.nCameraSystem.T_SC(0);

                    if (T_SC_ptr != nullptr) {
                        // [阶段2B修改] 与 Estimator.cpp 中伪装 KeyFrame 的坐标链保持一致：T_WC = T_WS * T_SC。
                        okvis::kinematics::Transformation T_WC = T_WS * (*T_SC_ptr);

                        Eigen::Quaterniond q_wc = T_WC.q();
                        q_wc.normalize();
                        Eigen::Vector3d t_wc = T_WC.r();

                        Sophus::SE3f Twc(q_wc.cast<float>(), t_wc.cast<float>());
                        pSVIn2ORBWrapper->CacheFrontendPose(t.toSec(), Twc);
                    }
                }
            });
        
        okvis_estimator->setLandmarksCallback([publisher, igb](const okvis::Time& t, const okvis::MapPointVector& actualLandmarks, const okvis::MapPointVector& marginalizedLandmarks) {
            // 彻底注释掉下面这两行！
            // 因为 actualLandmarks 是跨库传过来的 std::vector，读取它必定引发内存塌方！
            
            // publisher->publishLandmarksAsCallback(t, actualLandmarks, marginalizedLandmarks);
            // if (igb != nullptr) {
            //     igb->TrackingWatchdog(t.toSec(), actualLandmarks.size());
            // }
        }
    );
        
        okvis_estimator->setStateCallback(std::bind(&okvis::Publisher::publishStateAsCallback, publisher, std::placeholders::_1, std::placeholders::_2));
        okvis_estimator->setKeyframeCallback(std::bind(&okvis::Publisher::publishKeyframeAsCallback, publisher, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
        
        if (parameters.visualization.publishDebugImages) {
            okvis_estimator->setDebugImgCallback(std::bind(&okvis::Publisher::publishDebugImageAsCallback, publisher, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
        } else {
        }
    
        if (is_reloc) {
            okvis_estimator->setRelocRelativePoseCallback(std::bind(&okvis::Publisher::publishRelocRelativePoseAsCallback, publisher, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5));
        } else {
        }
    
        okvis_estimator->setImuCsvFile("imu_data.csv");
        for (size_t i = 0; i < 2; ++i) {
            std::stringstream num;
            num << i + 1;
            okvis_estimator->setTracksCsvFile(i, "slave" + num.str() + "_tracks.csv");
        }
    }
    }

std::vector<std::string> SplitPath(std::string path) {
    std::vector<std::string> tokens;
    std::string token;
    std::stringstream ss(path);
    while (getline(ss, token, '/')) {
        tokens.push_back(token);
    }
    return tokens; //将path的内容一个一个存入，如home、sylva、...
}

int id = 0;

void SegfaultHandler(int sig) {
    void *array[20];
    size_t size;
    size = backtrace(array, 20);
    fprintf(stderr, "\n\033[1;31m========================================================\033[0m\n");
    fprintf(stderr, "\033[1;31m[FATAL ERROR] BACKGROUND THREAD SEGFAULT (Exit Code -11)!\033[0m\n");
    fprintf(stderr, "\033[1;31m========================================================\033[0m\n");
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    exit(1);
}

int main(int argc, char **argv) {
    // ros::init(argc, argv, "Mono");
    signal(SIGSEGV, SegfaultHandler);
    ros::init(argc, argv, "Mono-temp");//全局命名空间
    ros::start();
    cerr <<"-----------------------------------------------Cloud-Edge-SLAM RUN STAR!--------------------------------------------------------"<< endl;;
    ros::Time time_begin = ros::Time::now();
    ros::NodeHandle nh("~");//局部命名空间，/cloud_edge_slam，对应launch中的node

    string cloudTopicName;
    getRosParam(nh, "cloud_topic_name", cloudTopicName);//获取launcimageTopicPubh中参数的值 /cloud_slam
    string vocabularyPath;
    string settingPath;
    getRosParam(nh, "vocabulary_path", vocabularyPath);// ORBvoc.txt的路径
    getRosParam(nh, "setting_path", settingPath);//.yaml的路径
    string dataType;
    string dataPath;
    string resultPath;
    getRosParam(nh, "data_type", dataType);//数据类型，bag、txt
    getRosParam(nh, "data_path", dataPath);//数据集路径
    getRosParam(nh, "result_path", resultPath);//结果存放路径
    bool bCloudMerge;
    bool bSaveCloudBag;
    bool bSaveEdgeTraj;
    bool bRealOnline;
    bool bMergeAnyway;
    bool bCloudOnline;
    bool bWaitCloudResult;
    float mainLoopSleep;
    bool bSetImgGray;
    bool bKFCulling;
    getRosParam(nh, "cloud_merge", bCloudMerge);//云端合并功能，true
    getRosParam(nh, "save_cloud_bag", bSaveCloudBag);//false
    getRosParam(nh, "real_online", bRealOnline);//等待云端连接成功后初始化，true
    getRosParam(nh, "merge_anyway", bMergeAnyway);//无论什么都合并，false
    getRosParam(nh, "cloud_online", bCloudOnline);//功能不明,true
    getRosParam(nh, "wait_cloud_result", bWaitCloudResult);//等待Cloud SLAM的结果并Merge完成后再继续运行，true
    getRosParam(nh, "main_loop_sleep_ms", mainLoopSleep);//每帧运行后睡眠的时间,30ms
    getRosParam(nh, "kf_culling", bKFCulling);//功能不明，false
    //cap-udf
    bool bOldUdf;
    bool bNewUdf;
    getRosParam(nh, "old_udf_cloud_edge", bOldUdf);
    getRosParam(nh, "new_udf_cloud_edge", bNewUdf);
    //cap-udf
    int nSamplerEdgeFrontKFNum;
    int nSamplerEdgeBackKFNum;
    float nSamplerEdgeFrontMinTime;
    float nSamplerEdgeBackMinTime;
    float samplerPDKp;
    float samplerPDKd;
    float samplerPDth;
    getRosParam(nh, "sampler_edge_front_kf_num", nSamplerEdgeFrontKFNum);//前端地图最小关键帧数量为40
    getRosParam(nh, "sampler_edge_back_kf_num", nSamplerEdgeBackKFNum);//后端地图最小关键帧数量为40
    getRosParam(nh, "sampler_edge_front_min_time", nSamplerEdgeFrontMinTime);//前端，最小时间为3
    getRosParam(nh, "sampler_edge_back_min_time", nSamplerEdgeBackMinTime);//后端，最小时间为3
    getRosParam(nh, "sampler_pd_kp", samplerPDKp);//光流法相关的一些参数
    getRosParam(nh, "sampler_pd_kd", samplerPDKd);//光流法相关的一些参数
    getRosParam(nh, "sampler_pd_th", samplerPDth);//光流法相关的一些参数

    std::vector<string> splitPaths = SplitPath(dataPath); //顾名思义，分离路径，文字
    string datasetName = splitPaths[splitPaths.size() - 2];//提取存放数据集的文件名

    auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());//获取时间戳
    std::stringstream ss;
    ss << std::put_time(std::localtime(&t), "%m-%d-%H_%M_%S");
    std::string str_time = ss.str();

    boost::filesystem::path parent_dir(resultPath);
    boost::filesystem::path sub_dir(datasetName + "#" + str_time);
    boost::filesystem::path finish_sub_dir("Full#" + datasetName + "#" + str_time);
    boost::filesystem::path full_path_ = parent_dir / sub_dir;
    boost::filesystem::path full_finish_path_ = parent_dir / finish_sub_dir;
    string full_path = full_path_.string();//提取结果路径
    string full_finish_path = full_finish_path_.string();
    int isCreate = mkdir(full_path.c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);//创建结果路径
    cerr << "create path: " << full_path << endl;
    if (!isCreate) {
        cerr << "create path finish" << endl;
    } else {
        cerr << "create path failed! error code: !!!!!!!!!!!!" << isCreate << endl;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(vocabularyPath, settingPath, ORB_SLAM3::System::MONOCULAR, true, bCloudMerge, bCloudOnline, bMergeAnyway, bKFCulling, nSamplerEdgeFrontKFNum, nSamplerEdgeBackKFNum, nSamplerEdgeFrontMinTime, nSamplerEdgeBackMinTime, samplerPDKp, samplerPDKd, samplerPDth, bOldUdf, bNewUdf);//
    //调用system.cc，初始化orbslam3，输入一些参数，如字典路径、yaml路径，相机类型，可视化查看器，launch中的参数

    // [CloudMap校正诊断] 将当前结果目录传给 CloudMerging，用于输出 CloudMap 校正统计 summary。
    ORB_SLAM3::CloudMerging *pCloudMerger = SLAM.GetCloudMerger();
    if (pCloudMerger != nullptr) {
        pCloudMerger->SetCloudMergeDebugOutputDir(full_path);
    }
    
    // [新增] 实例化 Wrapper 层，建立 OKVIS 到 ORB-SLAM3 的数据桥梁
    pSVIn2ORBWrapper = new SVIn2ORBWrapper(&SLAM);

    std::string okvis_full_traj_path = full_path + "/okvis_full_traj.txt";
    pSVIn2ORBWrapper->InitOkvisFullTrajectorySaver(okvis_full_traj_path);

    Grabber igb(&SLAM);

    pSVIn2ORBWrapper->RegisterStateCallbacks(
        std::bind(&Grabber::TrackingWatchdog, &igb, std::placeholders::_1, std::placeholders::_2),
        std::bind(&Grabber::GetTrackingState, &igb)
    );

    igb.SetParameters(bWaitCloudResult, mainLoopSleep, full_path, bSaveCloudBag, bOldUdf, bNewUdf);//设置几个参数，如是否等待云端结果，每帧运行后睡眠时间，保存的路径

    igb.SetNodeHandle(&nh);//270-281

    ros::Publisher orbMapPub = nh.advertise<cloud_edge_slam::CloudMap>("/test_cloud_map", 1);
    igb.SetOrbMapPublisher(&orbMapPub);

    ros::Publisher imageTopicPub = nh.advertise<sensor_msgs::CompressedImage>("/cloud_edge_images", 100);
    igb.SetImageTopicPublisher(&imageTopicPub);
    
    ros::ServiceClient evoClient = nh.serviceClient<cloud_edge_slam::Evo>("/cloud_edge_evo_temp");

    ros::Subscriber memorySub = nh.subscribe("/cloud_edge_memory_temp", 1, &Grabber::MemoryCb, &igb);

    // For Pub Cloud Images
    CloudClient cloudImagesActionClient(cloudTopicName, true);
    if (bRealOnline) {
        ROS_INFO_STREAM("Waiting For Server Begin!");
        cloudImagesActionClient.waitForServer();
        ROS_INFO_STREAM("Waiting For Server End!");
        igb.SetCloudImagesActionClient(&cloudImagesActionClient);
    }

    // For Get Test Publish Map
    ros::Subscriber save_orb_map_sub = nh.subscribe("/test_save_orb_map", 1, &Grabber::SaveORBMapCb, &igb);

    ros::Subscriber cloud_map_sub = nh.subscribe("/test_cloud_map", 1, &Grabber::GrabCloudMapCb, &igb);

    std::chrono::steady_clock::time_point timeStart = std::chrono::steady_clock::now();
    
    // [移植入口流接管] 实例化 OKVIS 追踪器并接管底层传感器流
    okvis::RosParametersReader vio_parameters_reader(settingPath);
    okvis::VioParameters parameters;
    vio_parameters_reader.getParameters(parameters);

    okvis::ThreadedKFVio okvis_estimator(parameters);
    okvis::Publisher publisher(nh);
    
    // 动态打入结果目录！
    publisher.setCsvFile(full_path + "/okvis_estimator_output.csv");
    publisher.setLandmarksCsvFile(full_path + "/okvis_estimator_landmarks.csv");

    okvis::initEstimator(&okvis_estimator, &publisher, parameters, &igb);

    okvis::Subscriber subscriber(nh, &okvis_estimator, vio_parameters_reader);
    
    std::string raw_image_topic;
    nh.param<std::string>("raw_image_topic", raw_image_topic, "/camera/rgb/image_color");
    ros::Subscriber raw_image_sub = nh.subscribe(raw_image_topic, 200, &Grabber::RawImageCb, &igb);
    
    if (bCloudOnline) {
        ROS_INFO_STREAM("System is now ready. Please play rosbag or start sensor drivers.");
        // 使用多线程 Spinner 以保障高频 IMU 回调与图像回调不被阻塞
        ros::MultiThreadedSpinner spinner(4);
        spinner.spin();
    } else {
        // 保持逻辑完整
    }

    std::chrono::steady_clock::time_point timeEnd = std::chrono::steady_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(timeEnd - timeStart).count();
    duration /= 1e3;

    // begin export
    std::string map_point = "/home/birl/Udf-Edge/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/src/cloud_edge_slam/results/point.txt";
    std::ofstream fout_map_point(map_point);
    fout_map_point << SLAM.GetAtlas()->MapPointsInMap() << endl;
    fout_map_point << duration << endl;
    fout_map_point.close();

    // [轨迹导出修改] 直接从 Atlas / Map / KeyFrame 内存对象导出最终保留轨迹，不再生成中间 traj_map 文件。
    std::vector<ORB_SLAM3::Map*> vpMaps = SLAM.GetAtlas()->GetAllMaps();
    int valid_map_count = 0;
    for (size_t i = 0; i < vpMaps.size(); i++) {
        ORB_SLAM3::Map* pMap = vpMaps[i];
        if (pMap == nullptr) {
            continue;
        }

        if (pMap->KeyFramesInMap() > 5) {
            valid_map_count++;
            std::cout << "[System] Receive Submap ID: "
                      << pMap->GetId()
                      << " (包含 "
                      << pMap->KeyFramesInMap()
                      << " FPS)"
                      << std::endl;
        }
    }
    std::cout << "[System] A total of " << valid_map_count << " valid trajectories were processed. " << std::endl;

    std::vector<TrajectoryRecord> atlasTrajectoryRecords;
    CollectTrajectoryRecordsFromAtlas(SLAM.GetAtlas(), atlasTrajectoryRecords);
    const size_t rawRecordCount = atlasTrajectoryRecords.size();
    size_t cloudRecordCount = 0;
    size_t nonCloudRecordCount = 0;
    for (size_t i = 0; i < atlasTrajectoryRecords.size(); i++) {
        if (atlasTrajectoryRecords[i].isCloud) {
            cloudRecordCount++;
        } else {
            nonCloudRecordCount++;
        }
    }

    std::cout << "[Trajectory Export] raw records = "
              << rawRecordCount
              << ", cloud records = "
              << cloudRecordCount
              << ", non-cloud records = "
              << nonCloudRecordCount
              << std::endl;

    std::string whole_map_path = full_path + "/whole_map.txt";
    const size_t wholeMapCount = SaveTrajectoryRecordsTUM(
        atlasTrajectoryRecords,
        whole_map_path,
        kTrajectoryFilterAll);

    std::cout << "[Trajectory Export] Saved whole_map.txt, records = "
              << wholeMapCount
              << std::endl;

    std::string whole_no_cloud_path = full_path + "/whole_map_no_cloud.txt";
    const size_t noCloudCount = SaveTrajectoryRecordsTUM(
        atlasTrajectoryRecords,
        whole_no_cloud_path,
        kTrajectoryFilterEdgeOnly);

    std::cout << "[Trajectory Export] Saved whole_map_no_cloud.txt, records = "
              << noCloudCount
              << std::endl;

    std::string cloud_only_path = full_path + "/whole_map_cloud_only.txt";
    const size_t cloudOnlyCount = SaveTrajectoryRecordsTUM(
        atlasTrajectoryRecords,
        cloud_only_path,
        kTrajectoryFilterCloudOnly);

    std::cout << "[Trajectory Export] Saved whole_map_cloud_only.txt, records = "
              << cloudOnlyCount
              << std::endl;

    // [轨迹导出修改] 每个有效 Map 只导出 cloud-only 轨迹；无 cloud KeyFrame 的 Map 跳过空文件生成。
    for (size_t mapIndex = 0; mapIndex < vpMaps.size(); mapIndex++) {
        ORB_SLAM3::Map *pMap = vpMaps[mapIndex];
        if (pMap == nullptr) {
            continue;
        }

        std::vector<TrajectoryRecord> mapTrajectoryRecords;
        CollectTrajectoryRecordsFromMap(pMap, mapTrajectoryRecords);
        std::string map_id_string = std::to_string(pMap->GetId());
        std::string map_cloud_only_file_name = "map_" + map_id_string + "_cloud_only.txt";
        const size_t mapCloudOnlyRecordCount = CountTrajectoryRecords(
            mapTrajectoryRecords,
            kTrajectoryFilterCloudOnly);

        if (mapCloudOnlyRecordCount == 0) {
            std::cout << "[Trajectory Export] Skip "
                      << map_cloud_only_file_name
                      << " because this Map has no cloud KeyFrame."
                      << std::endl;
            continue;
        }

        std::string map_cloud_only_path = full_path + "/" + map_cloud_only_file_name;
        const size_t mapCloudOnlyCount = SaveTrajectoryRecordsTUM(
            mapTrajectoryRecords,
            map_cloud_only_path,
            kTrajectoryFilterCloudOnly);

        std::cout << "[Trajectory Export] Saved "
                  << map_cloud_only_file_name
                  << ", records = "
                  << mapCloudOnlyCount
                  << std::endl;
    }
    
    // rename dir for tag
    boost::filesystem::rename(full_path_, full_finish_path_);



    // @note 直接暂停
    // exit(1000);
    
    // Debug: 等待CloudMap
    ros::spin();
    
    // Stop all threads
    ros::Time time_end = ros::Time::now();
    ros::Duration duration1 = time_end - time_begin;
    ROS_INFO("Cloud-Edge-SLAM系统运行总时长: %lf secs", duration1.toSec());
    SLAM.Shutdown();

    // 退出前的资源清理，销毁跨系统数据通信实例
    if (pSVIn2ORBWrapper != nullptr) {
        pSVIn2ORBWrapper->CloseOkvisFullTrajectorySaver();
        pSVIn2ORBWrapper->CloseTrajectorySaver();

        delete pSVIn2ORBWrapper;
        pSVIn2ORBWrapper = nullptr;
    } else {
        // Wrapper 尚未分配或已回收
    }

    ros::shutdown();
    return 0;
}

inline Sophus::SE3f toSophusPose(const geometry_msgs::Pose rosPose) {
    Eigen::Quaternion<float> quaterniond(rosPose.orientation.w, rosPose.orientation.x, rosPose.orientation.y, rosPose.orientation.z);
    Eigen::Vector3f pos(rosPose.position.x, rosPose.position.y, rosPose.position.z);
    Sophus::SE3f pose(quaterniond, pos);
    return pose;
}

inline sensor_msgs::CameraInfo getCameraInfo(int imageWidth, int imageHeight, Eigen::Matrix3f K) { // extract cameraInfo.
    sensor_msgs::CameraInfo cam;

    boost::array<double, 9> K_array = {
        K(0, 0), K(0, 1), K(0, 2),
        K(1, 0), K(1, 1), K(1, 2),
        K(2, 0), K(2, 1), K(2, 2)};

    cam.width = imageWidth;
    cam.height = imageHeight;
    cam.distortion_model = "plumb_bob";
    cam.K = K_array;
    cam.binning_x = 0;
    cam.binning_y = 0;
    cam.header.frame_id = "camera"; //frame_id为camera，也就是相机名字
    cam.header.stamp = ros::Time::now();
    cam.header.stamp.nsec = 0;
    return cam;
}

void Grabber::SetCloudImagesActionClient(CloudClient *pCloudImagesActionClient) {
    mpCloudImagesActionClient = pCloudImagesActionClient;
}

void Grabber::SetParameters(bool bWaitCloudResult, float nMainLoopSleep, string savePath, bool bSaveCloudBag, bool bOldUdf, bool bNewUdf) {
    mbWaitCloudResult = bWaitCloudResult;
    mnMainLoopSleep = nMainLoopSleep;
    mSaveDir = savePath;
    mbSaveCloudBag = bSaveCloudBag;
    mbOldUdf = bOldUdf;
    mbNewUdf = bNewUdf;
}

void Grabber::SetNodeHandle(ros::NodeHandle *pNodeHandle) {
    mpNodeHandler = pNodeHandle;
}
void Grabber::SetImageTopicPublisher(ros::Publisher *pPublisher) {
    mpImageTopicPub = pPublisher;
}
void Grabber::SetOrbMapPublisher(ros::Publisher *pPublisher) {
    mpOrbMapPub = pPublisher;
}

void Grabber::TrackImage(const cv::Mat &img, const double &timestamp, const float &imageScale) {
    if (imageScale != 1.f) {
        int width = img.cols * imageScale;
        int height = img.rows * imageScale;
        cv::resize(img, img, cv::Size(width, height));
    }

    static int cnt = 0;
    ROS_INFO_STREAM_DELAYED_THROTTLE(1.0, "Deal One Image: " << cnt);
    cnt++;

    // @note input image
    mpSLAM->TrackMonocular(img, timestamp);//转至System.cc的387行，输入图像和时间戳

    // return;

    // 主动式检测Cloud Image
    std::vector<ORB_SLAM3::CloudImage> vCurrentProcessCloudImages;
    std::vector<ORB_SLAM3::CloudImage> vCurrentProcessCloudNoSamplingImages;
    int edgeFrontMapId, edgeBackMapId;
    mpSLAM->GetCloudProcessImages(vCurrentProcessCloudImages, vCurrentProcessCloudNoSamplingImages, edgeFrontMapId, edgeBackMapId);
    if (!vCurrentProcessCloudImages.empty()) {
        // @note print
        cout << "Main: Pub Cloud Images !" << endl;
        // debug: write cloud images
        static int index = 0;
        index++;
        if (mbSaveCloudBag) {
            string curCloudBagPath = (bfs::path(mSaveDir) / "cloud_").string() + to_string(index) + ".bag";
            WriteCloudImagesBag(vCurrentProcessCloudImages, curCloudBagPath);
            string curCloudNoSamplingBagPath = (bfs::path(mSaveDir) / "cloud_nosampling_").string() + to_string(index) + ".bag";
            WriteCloudImagesBag(vCurrentProcessCloudNoSamplingImages, curCloudNoSamplingBagPath);

            double curTimestamp = vCurrentProcessCloudImages[vCurrentProcessCloudImages.size() - 1].timestamp;
            auto bagSize = float(bfs::file_size(curCloudBagPath) / (1024 * 1024));
            mvBagSize[curTimestamp] = bagSize;
            auto noSamplingBagSize = float(bfs::file_size(curCloudNoSamplingBagPath) / (1024 * 1024));
            mvNoSamplingBagSize[curTimestamp] = noSamplingBagSize;
        }


        if (this->mpCloudImagesActionClient) { // 开启云端
            ROS_INFO_STREAM("Action Client Triggered");

            sensor_msgs::CameraInfo cameraInfo = getCameraInfo(mpSLAM->GetSetting()->newImSize().width, mpSLAM->GetSetting()->newImSize().height, mpSLAM->GetCamera()->toK_()); 
            cloud_edge_slam::Sequence imageSeqMsg;
            imageSeqMsg.Header.stamp = ros::Time::now();
            imageSeqMsg.camera = cameraInfo;
            imageSeqMsg.edge_front_map_mnid = edgeFrontMapId;
            imageSeqMsg.edge_back_map_mnid = edgeBackMapId;

            for (auto &img : vCurrentProcessCloudImages) {
                imageSeqMsg.timestamps.push_back(img.timestamp);
            }
        
            int total_imgs = vCurrentProcessCloudImages.size();
            std::vector<sensor_msgs::CompressedImagePtr> msgs_buffer(total_imgs);
            
            #pragma omp parallel for
            for (int i = 0; i < total_imgs; ++i) {
                const auto &image = vCurrentProcessCloudImages[i];
                
                sensor_msgs::CompressedImagePtr msg(new sensor_msgs::CompressedImage());
                msg->header.stamp = ros::Time(image.timestamp);
                msg->header.frame_id = image.type; 
                msg->format = "jpeg";
        
                std::vector<int> params; 
                params.push_back(cv::IMWRITE_JPEG_QUALITY); 
                params.push_back(90); // 压缩质量
        
        
                cv::imencode(".jpg", image.img, msg->data, params);
                msgs_buffer[i] = msg;
            }
        
            goal.sequence = imageSeqMsg;
            goal.total_image_count = total_imgs; 
            mpCloudImagesActionClient->sendGoal(goal,
                                                boost::bind(&Grabber::ActionFinishCb, this, _1, _2),
                                                CloudClient::SimpleActiveCallback(),
                                                CloudClient::SimpleFeedbackCallback());
        
            ros::Duration(1.0).sleep(); 
            ROS_INFO("Start send %d images by Topic...", total_imgs);
            ros::Rate burst_rate(100.0); // 高速发送
            for (auto &msg : msgs_buffer) {
                if (mpImageTopicPub->getNumSubscribers() > 0 || true) {
                    mpImageTopicPub->publish(msg);
                }
                ros::spinOnce();
                burst_rate.sleep();
            }
            
            // @note 等待CloudMerging结束再继续
            if (mbWaitCloudResult) {
                sleep(10);
                mpCloudImagesActionClient->waitForResult();
                usleep(1000 * 1e3);
                while (mpSLAM->GetCloudMerger()->isRunning()) {
                    usleep(100 * 1e3);
                }
            }

            mpSLAM->SetTrackLostTimestamp(timestamp);
        }
        mpSLAM->ResetCloudProcessImages();
    }
}

//cap-udf
inline bool exists_test (const std::string& name ) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}
//cap-udf

// void Grabber::ActionFinishCb(const actionlib::SimpleClientGoalState &state, const cloud_edge_slam::CloudSlamResultConstPtr &result) {
//     //在此接收关键变量，并等待加载bag数据集，调用ROSMapToORBMap函数 //new_udf_cloud-edge
//     if(this->mbNewUdf) {
//         system("rm -f /home/lhf/vc-SLAM_ws/Edge-SLAM/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_bag/test.bag");
//         system("rm -f /home/lhf/vc-SLAM_ws/Edge-SLAM/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_point/test.xyz");
//         cloud_edge_slam::CloudMapConstPtr pMap(new cloud_edge_slam::CloudMap(result->map));
//         std::string pc_name = "/home/lhf/vc-SLAM_ws/Edge-SLAM/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_point/test.xyz";
//         std::ofstream fout_pc_name(pc_name);

//         for (int n = 0; n < pMap->map_points.size(); n++){
//             fout_pc_name << pMap->map_points[n].point.x << " " << pMap->map_points[n].point.y << " " << pMap->map_points[n].point.z << std::endl;
//         }
//         fout_pc_name.close();

//         std::string id_name = "/home/lhf/vc-SLAM_ws/Edge-SLAM/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_others/id.txt";
//         std::ofstream fout_id_name(id_name);

//         fout_id_name << pMap->edge_front_map_mnid << std::endl;
//         fout_id_name << pMap->edge_back_map_mnid << std::endl;
//         fout_id_name.close();

//         bool trans = true;

//         do{
//             bool test1 = exists_test("/home/lhf/vc-SLAM_ws/Edge-SLAM/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_bag/test.bag");//未生成此文件

//             if (test1 == true){
//                 trans = false;
//             }
//         }
//         while(trans);    
//         //new_udf_cloud-edge

//         //加载bag数据集
//         ROS_ERROR_STREAM("Begin read cloud_map_bag");
//         rosbag::Bag test_bag_all;
//         sleep(3);
//         test_bag_all.open("/home/lhf/vc-SLAM_ws/Edge-SLAM/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_bag/test.bag", rosbag::bagmode::Read); //打开bag并读取
//         ROS_ERROR_STREAM("End read cloud_map_bag");
//         std::vector<std::string> topics;
//         const string map_topic_name = "/test";
//         topics.push_back(map_topic_name);
//         rosbag::View view(test_bag_all, rosbag::TopicQuery(topics));
//         for (auto m : view) {
//             cloud_edge_slam::CloudSlamResultConstPtr test_bag = m.instantiate<cloud_edge_slam::CloudSlamResult>();
//             cloud_edge_slam::CloudMapConstPtr mapPtr(new cloud_edge_slam::CloudMap(test_bag->map));
//             mpSLAM->InsertCloudMap(ROSMapToORBMap(mapPtr));
//         }
//         ROS_INFO_STREAM("Cloud Process Finish!");
//         test_bag_all.close();
//         system("rm /home/lhf/vc-SLAM_ws/Edge-SLAM/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_bag/test.bag");
//         system("rm /home/lhf/vc-SLAM_ws/Edge-SLAM/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_point/test.xyz");
//     }
//     //cap-udf-new
//     else {
//         cloud_edge_slam::CloudMapConstPtr mapPtr(new cloud_edge_slam::CloudMap(result->map));
//         mpSLAM->InsertCloudMap(ROSMapToORBMap(mapPtr));
//         ROS_INFO_STREAM("Cloud Process Finish!");
//     }
    
// }

void Grabber::ActionFinishCb(const actionlib::SimpleClientGoalState &state, const cloud_edge_slam::CloudSlamResultConstPtr &result) {
    if (state == actionlib::SimpleClientGoalState::SUCCEEDED) {
        if(this->mbNewUdf) {
            // ROS_INFO("[OUTER PROBE 0] mbNewUdf Triggered. Pre-clearing files.");
            system("rm -f /home/lhf/vc-SLAM_ws/Edge-SLAM/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_bag/test.bag");
            system("rm -f /home/lhf/vc-SLAM_ws/Edge-SLAM/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_point/test.xyz");
            
            cloud_edge_slam::CloudMapConstPtr pMap(new cloud_edge_slam::CloudMap(result->map));
            std::string pc_name = "/home/lhf/vc-SLAM_ws/Edge-SLAM/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_point/test.xyz";
            std::ofstream fout_pc_name(pc_name);

            for (int n = 0; n < pMap->map_points.size(); n++){
                fout_pc_name << pMap->map_points[n].point.x << " " << pMap->map_points[n].point.y << " " << pMap->map_points[n].point.z << std::endl;
            }
            fout_pc_name.close();

            std::string id_name = "/home/lhf/vc-SLAM_ws/Edge-SLAM/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_others/id.txt";
            std::ofstream fout_id_name(id_name);

            fout_id_name << pMap->edge_front_map_mnid << std::endl;
            fout_id_name << pMap->edge_back_map_mnid << std::endl;
            fout_id_name.close();

            sync(); // 强制将操作系统的 I/O 缓冲区数据刷新到物理硬盘
            std::string ready_file = "/home/lhf/vc-SLAM_ws/Edge-SLAM/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_others/data.ready";
            std::ofstream fout_ready(ready_file);
            fout_ready.close();
            // ROS_INFO("\033[1;32m[OUTER PROBE] Data fully written. Sync signal data.ready created.\033[0m");

            bool trans = true;
            // ROS_INFO("[OUTER PROBE 1] Waiting for edge.py to generate test.bag...");
            do{
                bool test1 = exists_test("/home/lhf/vc-SLAM_ws/Edge-SLAM/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_bag/test.bag");
                if (test1 == true){
                    trans = false;
                }
                usleep(50 * 1000); 
            } while(trans);    

            ROS_ERROR_STREAM("Begin read cloud_map_bag");
            rosbag::Bag test_bag_all;
            sleep(3);
            test_bag_all.open("/home/lhf/vc-SLAM_ws/Edge-SLAM/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_bag/test.bag", rosbag::bagmode::Read); 
            ROS_ERROR_STREAM("End read cloud_map_bag");
            
            std::vector<std::string> topics;
            const std::string map_topic_name = "/test";
            topics.push_back(map_topic_name);
            rosbag::View view(test_bag_all, rosbag::TopicQuery(topics));
            
            for (auto m : view) {
                cloud_edge_slam::CloudSlamResultConstPtr test_bag = m.instantiate<cloud_edge_slam::CloudSlamResult>();
                cloud_edge_slam::CloudMapConstPtr mapPtr(new cloud_edge_slam::CloudMap(test_bag->map));
                
                // ROS_INFO("[OUTER PROBE 2] Starting ROSMapToORBMap...");
                ORB_SLAM3::Map* cloudMap = ROSMapToORBMap(mapPtr);
                // ROS_INFO("[OUTER PROBE 3] ROSMapToORBMap completed successfully.");

                // 【探针猜想 A：同步遍历验证内存完整性】
                int valid_kfs = 0, valid_mps = 0;
                for(auto kf : cloudMap->GetAllKeyFrames()) { if(kf) valid_kfs++; }
                for(auto mp : cloudMap->GetAllMapPoints()) { if(mp) valid_mps++; }
                // ROS_INFO("[OUTER PROBE 4] Sync Traversal Check: %d KFs, %d MPs safely readable.", valid_kfs, valid_mps);

                // 【探针猜想 B：OpenCV 描述子类型监测】
                if(valid_kfs > 0 && cloudMap->GetAllKeyFrames()[0]->mDescriptors.rows > 0) {
                    // ROS_INFO("[OUTER PROBE 5] CloudMap Descriptor Type: %d (0=CV_8U, 5=CV_32F)", cloudMap->GetAllKeyFrames()[0]->mDescriptors.type());
                }
                ORB_SLAM3::Map* currentAtlasMap = mpSLAM->GetAtlas()->GetCurrentMap();
                if(currentAtlasMap && currentAtlasMap->GetAllKeyFrames().size() > 0) {
                    // ROS_INFO("[OUTER PROBE 6] Atlas Map Descriptor Type: %d", currentAtlasMap->GetAllKeyFrames()[0]->mDescriptors.type());
                }

                // ROS_INFO("[OUTER PROBE 7] Target FrontMapID: %ld, Target BackMapID: %ld", cloudMap->edgeFrontMapMnId, cloudMap->edgeBackMapMnId);
                
                // ROS_INFO("[OUTER PROBE 8] Calling mpSLAM->InsertCloudMap...");
                mpSLAM->InsertCloudMap(cloudMap);
                // ROS_INFO("[OUTER PROBE 9] mpSLAM->InsertCloudMap returned. Background threads awakened.");
            }
            
            ROS_INFO_STREAM("Cloud Process Finish!");
            test_bag_all.close();
            
            // 【探针猜想 C：异步死亡诱捕器】
            // ROS_INFO("[OUTER PROBE 10] Entering 3-second SLEEP to trap background thread crash...");
            std::this_thread::sleep_for(std::chrono::seconds(3));
            // ROS_INFO("[OUTER PROBE 11] Survived the sleep! Background threads are stable.");
            
            system("rm -f /home/lhf/vc-SLAM_ws/Edge-SLAM/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_bag/test.bag");
            system("rm -f /home/lhf/vc-SLAM_ws/Edge-SLAM/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_point/test.xyz");
            // ROS_INFO("[OUTER PROBE 12] ActionFinishCb fully completed.");
        }
        else {
            cloud_edge_slam::CloudMapConstPtr mapPtr(new cloud_edge_slam::CloudMap(result->map));
            mpSLAM->InsertCloudMap(ROSMapToORBMap(mapPtr));
            ROS_INFO_STREAM("Cloud Process Finish!");
        }
    }
}

void Grabber::MemoryCb(const std_msgs::Float32ConstPtr &msg) {
    mvMemory.push_back(msg->data);
}

void Grabber::GrabCloudMapCb(const cloud_edge_slam::CloudMapConstPtr &msg) {
    ROS_ERROR_STREAM("Read Cloud Map Start!");

    ORB_SLAM3::Map *cloudMap = ROSMapToORBMap(msg);

    mpSLAM->InsertCloudMap(cloudMap);

    ROS_ERROR_STREAM("Read Cloud Map End!");
}

inline geometry_msgs::Pose toRosPose(const Sophus::SE3f pose) {
    Eigen::Matrix4f T = pose.matrix();

    geometry_msgs::Pose rosPose;
    rosPose.position.x = T(0, 3);
    rosPose.position.y = T(1, 3);
    rosPose.position.z = T(2, 3);
    Eigen::Quaternionf q;
    q = T.block<3, 3>(0, 0);
    rosPose.orientation.x = q.x();
    rosPose.orientation.y = q.y();
    rosPose.orientation.z = q.z();
    rosPose.orientation.w = q.w();

    return rosPose;
}

void Grabber::PubORBMapCb(const std_msgs::Int16ConstPtr &msg) {
    int mapMnId = msg->data;
    ORB_SLAM3::Map *pMap;
    if (mapMnId == -1) {
        pMap = mpSLAM->GetAtlas()->GetCurrentMap();
    } else {
        pMap = mpSLAM->GetAtlas()->GetSpecifyMap(mapMnId);
    }
    mpOrbMapPub->publish(ORBMapToROSMap(pMap));
}

cloud_edge_slam::CloudMap Grabber::ORBMapToROSMap(ORB_SLAM3::Map *pMap) {
    cloud_edge_slam::CloudMap pubTestEdgeMap;

    std::vector<ORB_SLAM3::KeyFrame *> allKeyFrames = pMap->GetAllKeyFrames();
    std::vector<ORB_SLAM3::MapPoint *> allMapPoints = pMap->GetAllMapPoints();

    pubTestEdgeMap.header.seq = 0;
    pubTestEdgeMap.edge_front_map_mnid = pMap->GetId();
    pubTestEdgeMap.edge_back_map_mnid = pMap->GetId() + 1;

    // 提前取得指针与index的对应关系
    std::map<ORB_SLAM3::MapPoint *, int> mapPointMap;
    std::map<ORB_SLAM3::KeyFrame *, int> keyFrameMap;
    for (int mapPoint_i = 0; mapPoint_i < allMapPoints.size(); ++mapPoint_i) { mapPointMap[allMapPoints[mapPoint_i]] = mapPoint_i; }
    for (int keyFrame_i = 0; keyFrame_i < allKeyFrames.size(); ++keyFrame_i) { keyFrameMap[allKeyFrames[keyFrame_i]] = keyFrame_i; }

    for (int mapPoint_i = 0; mapPoint_i < allMapPoints.size(); ++mapPoint_i) {
        ORB_SLAM3::MapPoint *mapPoint = allMapPoints[mapPoint_i];
        std::map<ORB_SLAM3::KeyFrame *, std::tuple<int, int>> observations = mapPoint->GetObservations();

        cloud_edge_slam::MapPoint rosMapPoint;
        rosMapPoint.mnId = mapPoint->mnId;
        rosMapPoint.ref_keyframe_id = keyFrameMap[mapPoint->GetReferenceKeyFrame()]; // TODO check是否是这个
        Eigen::Vector3f point = mapPoint->GetWorldPos();
        rosMapPoint.point.x = point(0);
        rosMapPoint.point.y = point(1);
        rosMapPoint.point.z = point(2);

        for (auto observation : observations) {
            cloud_edge_slam::Observation rosObservation;
            if (keyFrameMap.count(observation.first) >= 1) {
                rosObservation.keyframe_id = keyFrameMap[observation.first]; // Bug! 即使keyFrameMap中没有该KeyFrame，std::map依然会返回0
                rosObservation.refer_keypoint_index = (short)std::get<0>(observation.second);
                if ((short)std::get<0>(observation.second) > observation.first->mvKeys.size()) {
                    ROS_ERROR_STREAM("observation keypoint index: " << (short)std::get<0>(observation.second) << "  " << observation.first->mvKeys.size());
                }
                assert((short)std::get<0>(observation.second) < observation.first->mvKeys.size());
                rosMapPoint.observations.push_back(rosObservation);
            }
        }
        rosMapPoint.num_obs = rosMapPoint.observations.size();
        pubTestEdgeMap.map_points.push_back(rosMapPoint);
    }

    // Add All KeyFrames
    for (auto KF : allKeyFrames) {
        Sophus::SE3f pose = KF->GetPose();
        std::vector<cv::KeyPoint> keyPoints = KF->mvKeys;
        std::vector<ORB_SLAM3::MapPoint *> matchMapPoint = KF->GetMapPointMatches();
        std::vector<cv::Mat> descriptors = ORB_SLAM3::Converter::toDescriptorVector(KF->mDescriptors);
        assert(matchMapPoint.size() == keyPoints.size() && keyPoints.size() == descriptors.size());

        cloud_edge_slam::KeyFrame rosKeyFrame;
        rosKeyFrame.mTimeStamp = KF->mTimeStamp;
        rosKeyFrame.mnId = KF->mnId;
        rosKeyFrame.pose_cw = toRosPose(pose);

        for (const auto &descriptor : descriptors) {
            cloud_edge_slam::Descriptor rosDescriptor;
            //            ROS_ERROR_STREAM("descriptor shape: " << descriptor.size);
            std::vector<double> descriptor_vector;
            descriptor.col(0).copyTo(descriptor_vector);
            std::copy(descriptor_vector.begin(), descriptor_vector.end(), rosDescriptor.descriptor.begin());
            rosKeyFrame.descriptors.push_back(rosDescriptor);
        }

        for (const auto &keyPoint : keyPoints) {
            cloud_edge_slam::KeyPoint rosKeyPoint;
            rosKeyPoint.x = keyPoint.pt.x;
            rosKeyPoint.y = keyPoint.pt.y;
            rosKeyFrame.key_points.push_back(rosKeyPoint);
        }

        for (const auto &mapPoint : matchMapPoint) {
            int matchMapPointIndex = -1;
            if (mapPoint) {
                matchMapPointIndex = mapPointMap[mapPoint];
            }
            rosKeyFrame.mvp_map_points_index.push_back(matchMapPointIndex);
        }
        pubTestEdgeMap.key_frames.push_back(rosKeyFrame);
    }

    return pubTestEdgeMap;
}

// ORB_SLAM3::Map *Grabber::ROSMapToORBMap(cloud_edge_slam::CloudMapConstPtr pMap) {
//     ROS_ERROR_STREAM("ROS Map To ORB Map Start!");

//     // Cloud Image
//     std::map<std::string, cv::Mat> dLastCloudProcessImages = mpSLAM->GetCloudImageSampler()->mdLastCloudProcessImages;

//     // hype parameters
//     const bool bIncludeDescriptor = false;

//     // SLAM System Info
//     auto pVocabulary = mpSLAM->GetVocabulary();
//     auto pExtractor = mpSLAM->GetExtractor();
//     auto pCamera = mpSLAM->GetCamera();
//     auto distCoef = mpSLAM->GetDistCoef();
//     auto bf = mpSLAM->Getbf();
//     auto thDepth = mpSLAM->GetThDepth();
//     auto imuCalib = mpSLAM->GetImcCalib();

//     static long unsigned int nCloudMapId = 1000;
//     ORB_SLAM3::Map *cloudMap = new ORB_SLAM3::Map(nCloudMapId++, true); 
    
//     cloudMap->edgeFrontMapMnId = pMap->edge_front_map_mnid;
//     cloudMap->edgeBackMapMnId = pMap->edge_back_map_mnid;

//     std::vector<ORB_SLAM3::KeyFrame *> vKeyFrames;
//     static int gen_start_id = 3565536;
//     gen_start_id -= 300000;
//     if (gen_start_id < 100000) {
//         gen_start_id = 3565536;
//     }
//     long unsigned int gen_mnId = gen_start_id; 
//     for (auto &rosKeyFrame : pMap->key_frames) {
//         // mvKeys
//         std::vector<cv::KeyPoint> vKeyPoints;
//         for (auto &rosKeyPoint : rosKeyFrame.key_points) {
//             cv::KeyPoint keyPoint;
//             keyPoint.pt.x = rosKeyPoint.x;
//             keyPoint.pt.y = rosKeyPoint.y;
//             vKeyPoints.push_back(keyPoint);
//         }

//         // Descriptors
//         cv::Mat descriptors;
//         if (bIncludeDescriptor) { 
//             std::vector<cv::Mat> vDescriptors;
//             for (auto &rosDescriptor : rosKeyFrame.descriptors) {
//                 std::vector<float> tmpDescriptor(rosDescriptor.descriptor.size());
//                 copy(rosDescriptor.descriptor.begin(), rosDescriptor.descriptor.end(), tmpDescriptor.begin());
//                 cv::Mat descriptor(tmpDescriptor);
//                 vDescriptors.push_back(descriptor);
//             }
//             descriptors = vDescriptors[0];
//             for (int descriptor_i = 1; descriptor_i < vDescriptors.size(); ++descriptor_i) {
//                 cv::hconcat(descriptors, vDescriptors[descriptor_i], descriptors);
//             }
//             descriptors = descriptors.t();
//         } else { 
//             descriptors = cv::Mat::zeros(vKeyPoints.size(), 32, CV_32FC1);
//         }

//         // pose
//         Sophus::SE3f pose = toSophusPose(rosKeyFrame.pose_cw);

//         // @note construct frame
//         ORB_SLAM3::Frame frame(vKeyPoints, rosKeyFrame.mTimeStamp, pExtractor, pVocabulary, pCamera, distCoef, bf, thDepth);
//         frame.mnId = gen_mnId--;
//         frame.mDescriptors = descriptors;
//         frame.mvpMapPoints = std::vector<ORB_SLAM3::MapPoint *>(rosKeyFrame.mvp_map_points_index.size(), static_cast<ORB_SLAM3::MapPoint *>(NULL));
//         ORB_SLAM3::KeyFrame *keyFrame = new ORB_SLAM3::KeyFrame(frame, cloudMap, nullptr); 
//         keyFrame->SetPose(pose);
//         keyFrame->SetCloudFlag();

//         // @note add img
//         if (dLastCloudProcessImages.count(std::to_string(frame.mTimeStamp))) {
//             keyFrame->SetImgGray(dLastCloudProcessImages[std::to_string(frame.mTimeStamp)]);
//         }

//         vKeyFrames.push_back(keyFrame); 
//     }
//     std::vector<ORB_SLAM3::MapPoint *> vMapPoints;
//     gen_mnId = gen_start_id; 

//     //cap-udf
//     if (this->mbOldUdf) {
//         std::string pc_name = "/home/birl/Udf-Edge/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_point/test.xyz";
//         std::ofstream fout_pc_name(pc_name);

//         for (int n = 0; n < pMap->map_points.size(); n++){
//             fout_pc_name << pMap->map_points[n].point.x << " " << pMap->map_points[n].point.y << " " << pMap->map_points[n].point.z << std::endl;
//         }
//         fout_pc_name.close();

//         bool trans = true;

//         do{
//             bool test1 = exists_test("/home/birl/Udf-Edge/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_point/test2.xyz");

//             if (test1 == true){
//                 trans = false;
//             }
//         }
//         while(trans);

//     }
//     //cap-udf

//     for (auto &rosMapPoint : pMap->map_points) {
//         if (rosMapPoint.ref_keyframe_id == -1) {
//             // 1：遇到孤儿点，推入 nullptr 占位，保证后续按 index 检索不乱序
//             vMapPoints.push_back(nullptr);
//             continue;
//         }
//         Eigen::Vector3f pos(rosMapPoint.point.x, rosMapPoint.point.y, rosMapPoint.point.z);
//         ORB_SLAM3::KeyFrame *refKeyFrame = vKeyFrames[rosMapPoint.ref_keyframe_id];
//         ORB_SLAM3::MapPoint *mapPoint = new ORB_SLAM3::MapPoint(gen_mnId--, pos, refKeyFrame, cloudMap);
//         mapPoint->isEdge = false;
//         vMapPoints.push_back(mapPoint);
//     }

//     for (int mapPoint_i = 0; mapPoint_i < vMapPoints.size(); ++mapPoint_i) {
//         ORB_SLAM3::MapPoint *mapPoint = vMapPoints[mapPoint_i];
//         if (mapPoint == nullptr) {
//             // 2：跳过空指针
//             continue;
//         }
//         for (auto &observation : pMap->map_points[mapPoint_i].observations) {
//             mapPoint->AddObservation(vKeyFrames[observation.keyframe_id], observation.refer_keypoint_index);
//             if ((short)observation.refer_keypoint_index > vKeyFrames[observation.keyframe_id]->mvKeys.size()) {
//                 ROS_ERROR_STREAM("observation keypoint index: " << (short)observation.refer_keypoint_index << "  " << vKeyFrames[observation.keyframe_id]->mvKeys.size());
//             }
//             assert((short)observation.refer_keypoint_index < vKeyFrames[observation.keyframe_id]->mvKeys.size());
//         }
//         mapPoint->ComputeDistinctiveDescriptors();
//         mapPoint->UpdateNormalAndDepth();
//     }
//     for (int keyFrame_i = 0; keyFrame_i < vKeyFrames.size(); ++keyFrame_i) {
//         ORB_SLAM3::KeyFrame *keyFrame = vKeyFrames[keyFrame_i];
//         keyFrame->ComputeBoW();
//         int i = 0;
//         for (auto &matchMapPointIndex : pMap->key_frames[keyFrame_i].mvp_map_points_index) {
//             if (matchMapPointIndex != -1) {
//                 if (vMapPoints[matchMapPointIndex] != nullptr) {
//                     // 3：必须确保被关联的 MapPoint 真实存在
//                     keyFrame->AddMapPoint(vMapPoints[matchMapPointIndex], i);
//                 }
//             }
//             i++;
//         }
//         keyFrame->UpdateCloudConnections();
//     }

//     // 处理完成后，添加到Map中
//     for (auto &keyFrame : vKeyFrames) {
//         cloudMap->AddKeyFrame(keyFrame);
//     }
//     for (auto &mapPoint : vMapPoints) {
//         if (mapPoint != nullptr) {
//             // 4：决不允许空指针混入底层维护图
//             cloudMap->AddMapPoint(mapPoint);
//         }
//     }

//     ROS_ERROR_STREAM("ROS Map To ORB Map End!");

//     return cloudMap;
// }

ORB_SLAM3::Map *Grabber::ROSMapToORBMap(cloud_edge_slam::CloudMapConstPtr pMap) {
    // ROS_INFO("=== [INNER PROBE 0] ROSMapToORBMap Start ===");

    std::map<std::string, cv::Mat> dLastCloudProcessImages = mpSLAM->GetCloudImageSampler()->mdLastCloudProcessImages;
    const bool bIncludeDescriptor = false;

    auto pVocabulary = mpSLAM->GetVocabulary();
    auto pExtractor = mpSLAM->GetExtractor();
    auto pCamera = mpSLAM->GetCamera();
    auto distCoef = mpSLAM->GetDistCoef();
    auto bf = mpSLAM->Getbf();
    auto thDepth = mpSLAM->GetThDepth();
    auto imuCalib = mpSLAM->GetImcCalib();

    static long unsigned int nCloudMapId = 1000;
    ORB_SLAM3::Map *cloudMap = new ORB_SLAM3::Map(nCloudMapId++, true); 
    
    cloudMap->edgeFrontMapMnId = pMap->edge_front_map_mnid;
    cloudMap->edgeBackMapMnId = pMap->edge_back_map_mnid;

    std::vector<ORB_SLAM3::KeyFrame *> vKeyFrames;
    static int gen_start_id = 3565536;
    gen_start_id -= 300000;
    if (gen_start_id < 100000) {
        gen_start_id = 3565536;
    }
    long unsigned int gen_mnId = gen_start_id; 
    
    ROS_INFO("=== [INNER PROBE 1] Parsing %zu KeyFrames ===", pMap->key_frames.size());
    for (auto &rosKeyFrame : pMap->key_frames) {
        std::vector<cv::KeyPoint> vKeyPoints;
        for (auto &rosKeyPoint : rosKeyFrame.key_points) {
            cv::KeyPoint keyPoint;
            
            if (std::isnan(rosKeyPoint.x)) {
                keyPoint.pt.x = 0.0f;
            } else {
                keyPoint.pt.x = rosKeyPoint.x;
            }
            
            if (std::isnan(rosKeyPoint.y)) {
                keyPoint.pt.y = 0.0f;
            } else {
                keyPoint.pt.y = rosKeyPoint.y;
            }
            
            vKeyPoints.push_back(keyPoint);
        }

        cv::Mat descriptors;
        if (bIncludeDescriptor) { 
            std::vector<cv::Mat> vDescriptors;
            for (auto &rosDescriptor : rosKeyFrame.descriptors) {
                std::vector<float> tmpDescriptor(rosDescriptor.descriptor.size());
                copy(rosDescriptor.descriptor.begin(), rosDescriptor.descriptor.end(), tmpDescriptor.begin());
                cv::Mat descriptor(tmpDescriptor);
                vDescriptors.push_back(descriptor);
            }
            descriptors = vDescriptors[0];
            for (int descriptor_i = 1; descriptor_i < vDescriptors.size(); ++descriptor_i) {
                cv::hconcat(descriptors, vDescriptors[descriptor_i], descriptors);
            }
            descriptors = descriptors.t();
        } else { 
            descriptors = cv::Mat::zeros(vKeyPoints.size(), 32, CV_32FC1);
        }

        geometry_msgs::Pose safe_pose_cw = rosKeyFrame.pose_cw; 
        if (std::isnan(safe_pose_cw.position.x) || std::isnan(safe_pose_cw.position.y) || std::isnan(safe_pose_cw.position.z)) {
            ROS_ERROR_STREAM("[MATH ARMOR] NaN Translation in KF " << rosKeyFrame.mnId);
            safe_pose_cw.position.x = 0.0f; 
            safe_pose_cw.position.y = 0.0f; 
            safe_pose_cw.position.z = 0.0f;
        }
        
        float q_norm = sqrt(safe_pose_cw.orientation.w * safe_pose_cw.orientation.w +
                            safe_pose_cw.orientation.x * safe_pose_cw.orientation.x +
                            safe_pose_cw.orientation.y * safe_pose_cw.orientation.y +
                            safe_pose_cw.orientation.z * safe_pose_cw.orientation.z);
                            
        if (std::isnan(q_norm) || q_norm < 1e-5) {
            ROS_ERROR_STREAM("[MATH ARMOR] Invalid Quaternion in KF " << rosKeyFrame.mnId);
            safe_pose_cw.orientation.w = 1.0f; 
            safe_pose_cw.orientation.x = 0.0f; 
            safe_pose_cw.orientation.y = 0.0f; 
            safe_pose_cw.orientation.z = 0.0f;
        }
        
        Sophus::SE3f pose = toSophusPose(safe_pose_cw);

        ORB_SLAM3::Frame frame(vKeyPoints, rosKeyFrame.mTimeStamp, pExtractor, pVocabulary, pCamera, distCoef, bf, thDepth);
        frame.mnId = gen_mnId--;
        frame.mDescriptors = descriptors;
        frame.mvpMapPoints = std::vector<ORB_SLAM3::MapPoint *>(rosKeyFrame.mvp_map_points_index.size(), static_cast<ORB_SLAM3::MapPoint *>(NULL));
        ORB_SLAM3::KeyFrame *keyFrame = new ORB_SLAM3::KeyFrame(frame, cloudMap, nullptr); 
        keyFrame->SetPose(pose);
        keyFrame->SetCloudFlag();

        if (dLastCloudProcessImages.count(std::to_string(frame.mTimeStamp))) {
            keyFrame->SetImgGray(dLastCloudProcessImages[std::to_string(frame.mTimeStamp)]);
        }
        vKeyFrames.push_back(keyFrame); 
    }

    std::vector<ORB_SLAM3::MapPoint *> vMapPoints;
    gen_mnId = gen_start_id; 

    if (this->mbOldUdf) {
        std::string pc_name = "/home/birl/Udf-Edge/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_point/test.xyz";
        std::ofstream fout_pc_name(pc_name);
        for (int n = 0; n < pMap->map_points.size(); n++) {
            fout_pc_name << pMap->map_points[n].point.x << " " << pMap->map_points[n].point.y << " " << pMap->map_points[n].point.z << std::endl;
        }
        fout_pc_name.close();
        bool trans = true;
        do {
            bool test1 = exists_test("/home/birl/Udf-Edge/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_point/test2.xyz");
            if (test1 == true) {
                trans = false;
            }
        } while(trans);
    }

    ROS_INFO("=== [INNER PROBE 2] Parsing %zu MapPoints ===", pMap->map_points.size());
    for (auto &rosMapPoint : pMap->map_points) {
        if (rosMapPoint.ref_keyframe_id == -1) {
            vMapPoints.push_back(nullptr);
            continue;
        }
        
        Eigen::Vector3f pos(rosMapPoint.point.x, rosMapPoint.point.y, rosMapPoint.point.z);
        if (std::isnan(pos(0)) || std::isnan(pos(1)) || std::isnan(pos(2))) {
            ROS_ERROR_STREAM("[MATH ARMOR] NaN MapPoint! Forcing origin.");
            pos = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
        }

        ORB_SLAM3::KeyFrame *refKeyFrame = vKeyFrames[rosMapPoint.ref_keyframe_id];
        ORB_SLAM3::MapPoint *mapPoint = new ORB_SLAM3::MapPoint(gen_mnId--, pos, refKeyFrame, cloudMap);
        mapPoint->isEdge = false;
        vMapPoints.push_back(mapPoint);
    }

    // ROS_INFO("=== [INNER PROBE 3] Applying Observations & Matches ===");
    for (int mapPoint_i = 0; mapPoint_i < vMapPoints.size(); ++mapPoint_i) {
        ORB_SLAM3::MapPoint *mapPoint = vMapPoints[mapPoint_i];
        if (mapPoint == nullptr) {
            continue;
        }
        for (auto &observation : pMap->map_points[mapPoint_i].observations) {
            mapPoint->AddObservation(vKeyFrames[observation.keyframe_id], observation.refer_keypoint_index);
            assert((short)observation.refer_keypoint_index < vKeyFrames[observation.keyframe_id]->mvKeys.size());
        }
        mapPoint->ComputeDistinctiveDescriptors();
        mapPoint->UpdateNormalAndDepth();
    }
    
    for (int keyFrame_i = 0; keyFrame_i < vKeyFrames.size(); ++keyFrame_i) {
        ORB_SLAM3::KeyFrame *keyFrame = vKeyFrames[keyFrame_i];
        keyFrame->ComputeBoW();
        int i = 0;
        for (auto &matchMapPointIndex : pMap->key_frames[keyFrame_i].mvp_map_points_index) {
            if (matchMapPointIndex != -1) {
                if (vMapPoints[matchMapPointIndex] != nullptr) {
                    keyFrame->AddMapPoint(vMapPoints[matchMapPointIndex], i);
                }
            }
            i++;
        }
        keyFrame->UpdateCloudConnections();
    }

    for (auto &keyFrame : vKeyFrames) {
        cloudMap->AddKeyFrame(keyFrame);
    }
    for (auto &mapPoint : vMapPoints) {
        if (mapPoint != nullptr) {
            cloudMap->AddMapPoint(mapPoint);
        }
    }

    // ROS_INFO("=== [INNER PROBE 4] ROSMapToORBMap End ===");
    return cloudMap;
}

void Grabber::SaveORBMapCb(const std_msgs::Int16ConstPtr &msg) {
    int mapMnId = msg->data;
    ORB_SLAM3::Map *pMap;
    if (mapMnId == -1) {
        pMap = mpSLAM->GetAtlas()->GetCurrentMap();
    } else {
        pMap = mpSLAM->GetAtlas()->GetSpecifyMap(mapMnId);
    }
    WriteCloudMapBag(ORBMapToROSMap(pMap), "/home/birl/Udf-Edge/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/src/cloud_edge_slam/TestData/All_ORB_Test_Offline_Map/" + to_string(mapMnId) + ".bag");
}

void Grabber::WriteCloudMapBag(cloud_edge_slam::CloudMap map, const std::string &save_path) {
    rosbag::Bag bag;
    bag.open(save_path, rosbag::bagmode::Write);
    bag.write("/test_cloud_map", ros::Time::now(), map);
    bag.close();
}

void Grabber::WriteCloudImagesBag(std::vector<ORB_SLAM3::CloudImage> &vImages, const std::string &save_path) {
    rosbag::Bag bag;
    bag.open(save_path, rosbag::bagmode::Write);
    for (auto &image : vImages) {
        std_msgs::Header header;
        header.stamp = ros::Time(image.timestamp);
        header.frame_id = image.type;
        string encode;
        if (image.img.channels() == 3) {
            encode = "bgr8";
        } else if (image.img.channels() == 1) {
            encode = "mono8";
        } else {
        }
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, encode, image.img).toImageMsg();
        bag.write("/test_cloud_images", ros::Time(image.timestamp), msg);
    }
    bag.close();
}

void Grabber::WriteSeqBag(const cloud_edge_slam::Sequence &seq, const std::string &save_path) {
    rosbag::Bag bag;
    bag.open(save_path, rosbag::bagmode::Write);
    auto curTime = ros::Time::now();
    bag.write("/test_seq", curTime, seq);
    bag.close();
}
void Grabber::RawImageCb(const sensor_msgs::ImageConstPtr &msg) {
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    double t = msg->header.stamp.toSec();
    {
        std::lock_guard<std::mutex> lock(mBufMutex);
        mImageBuffer[t] = cv_ptr->image.clone();
        
        // 维持约 15 秒的滑动缓存区 (时间差大于15秒则剔除最旧帧，严格控制内存)
        while (!mImageBuffer.empty() && (t - mImageBuffer.begin()->first > 15.0)) {
            mImageBuffer.erase(mImageBuffer.begin());
        }
    }
}

void Grabber::TrackingWatchdog(double timestamp, int num_landmarks) {
    std::lock_guard<std::mutex> lock(mStateMutex);
    
    inliers_window_.push_back(num_landmarks);
    if (inliers_window_.size() > 5) {
        inliers_window_.pop_front();
    }
    
    int sum = 0;
    for (size_t i = 0; i < inliers_window_.size(); ++i) {
        sum = sum + inliers_window_[i];
    }
    int avg_inliers = sum / inliers_window_.size();

    static double last_dash_time = 0.0;
    if (timestamp - last_dash_time > 0.5) { 
        std::string state_str = "NORMAL";
        if (current_state_ == TrackingState::NORMAL) {
            state_str = "NORMAL";
        } else if (current_state_ == TrackingState::WARNING) {
            state_str = "WARNING";
        } else if (current_state_ == TrackingState::LOST) {
            state_str = "LOST";
        }

        printf("\n\033[1;37m--------------------- SYSTEM MONITOR ---------------------\033[0m\n");
        printf("\033[1;36m[Status] %-8s | [Avg Inliers] %-4d | [Time] %.2f\033[0m\n", 
                state_str.c_str(), avg_inliers, timestamp);
        printf("\033[1;37m----------------------------------------------------------\033[0m\n");
        last_dash_time = timestamp;
    }

    // 针对海下边云协同优化的阶梯阈值与时间墙约束
    const int WARNING_THRESHOLD = 70;     
    const int LOST_THRESHOLD = 50;
    const int RECOVER_THRESHOLD = 90;  
    const double MAX_WARNING_DURATION = 5.0; 

if (current_state_ == TrackingState::NORMAL) {
        if (avg_inliers < WARNING_THRESHOLD) {
            current_state_ = TrackingState::WARNING;
            warning_start_time_ = timestamp; 
            ROS_WARN("Tracking Degrading! Entering WARNING state.");
        }
    } else if (current_state_ == TrackingState::WARNING) {
        if (avg_inliers >= RECOVER_THRESHOLD) {
            current_state_ = TrackingState::NORMAL;
            ROS_INFO("Tracking Recovered from WARNING to NORMAL.");
        } else {
            bool trigger_lost = false;
            
            if (avg_inliers < LOST_THRESHOLD) {
                trigger_lost = true;
                ROS_ERROR("Tracking LOST! Reason: Inliers dropped below physical threshold.");
            } else if ((timestamp - warning_start_time_) > MAX_WARNING_DURATION) {
                trigger_lost = true;
                ROS_ERROR("Tracking LOST! Reason: WARNING state timeout .");
            }

            if (trigger_lost) {
                current_state_ = TrackingState::LOST;
                // 进入 LOST 时只负责切图，绝对禁止在这里“抢跑”发送图像！
                ROS_ERROR("Topology cut. Entering blind spot, waiting for edge recovery...");
            }
        }
    } else if (current_state_ == TrackingState::LOST) {
        if (avg_inliers >= RECOVER_THRESHOLD) {
            current_state_ = TrackingState::NORMAL;
            double lost_end_time = timestamp;
            
            // 只有当边缘端物理上驶出盲区，且新子图彻底建立后，才触发云端求救！
            ROS_INFO("Tracking Recovered! New submap established. Triggering Cloud Recovery.");
            
            // 此时打包的图像序列完美覆盖：旧图截断 -> 盲区穿越 -> 新图建立的全过程
            std::thread(&Grabber::UploadLostImages, this, warning_start_time_, lost_end_time).detach();
        }
    }
}

// void Grabber::UploadLostImages(double start_time, double end_time) {
//     std::vector<ORB_SLAM3::CloudImage> vCurrentProcessCloudImages;

//     {
//         std::lock_guard<std::mutex> lock(mBufMutex);
        
//         double adjusted_start_time = start_time;
//         ORB_SLAM3::Map* pCurrentMap = mpSLAM->GetAtlas()->GetCurrentMap();
//         if (pCurrentMap != nullptr) {
//             std::vector<ORB_SLAM3::KeyFrame*> vpKFs = pCurrentMap->GetAllKeyFrames();
//             double last_kf_ts = 0.0;
//             for (size_t i = 0; i < vpKFs.size(); i++) {
//                 if (vpKFs[i] != nullptr) {
//                     if (vpKFs[i]->mTimeStamp > last_kf_ts) {
//                         last_kf_ts = vpKFs[i]->mTimeStamp;
//                     }
//                 }
//             }
//             // 如果最后一个关键帧的时间早于 warning_start_time_，则将时间窗向左延伸
//             if (last_kf_ts > 0.1 && last_kf_ts < adjusted_start_time) {
//                 adjusted_start_time = last_kf_ts;
//             }
//         }
        
//         auto it_start = mImageBuffer.lower_bound(adjusted_start_time - 0.1); 
//         auto it_end = mImageBuffer.upper_bound(end_time + 0.1);
        
//         for (auto it = it_start; it != mImageBuffer.end() && it != it_end; ++it) {
//             ORB_SLAM3::CloudImage ci(it->second.clone(), it->first, "RGB");
//             vCurrentProcessCloudImages.push_back(ci);
//         }
//     }
    
//     if (vCurrentProcessCloudImages.empty()) {
//         ROS_WARN("No images found in buffer for the lost period!");
//         return;
//     }
    
//     ROS_INFO("Extracted %zu images for Cloud Upload. Action Client Triggered.", vCurrentProcessCloudImages.size());

//     // -------------------------------------------------------------
//     // 以下为系统原有的网络通讯与压缩组装逻辑，无缝复用
//     // int edgeFrontMapId = mpSLAM->GetAtlas()->GetCurrentMap()->GetId(); 
//     // int edgeBackMapId = edgeFrontMapId + 1; // 假设云端会分配新地图
//     int edgeFrontMapId = 0;
//     int edgeBackMapId = 0;
    
//     if (mpSLAM->GetAtlas()->CountMaps() >= 2) {
//         edgeFrontMapId = mpSLAM->GetAtlas()->GetLastMap()->GetId();
//         edgeBackMapId = mpSLAM->GetAtlas()->GetCurrentMap()->GetId();
//     } else {
//         // 极小概率的兜底边界：如果系统一启动就立刻跟丢，只有一张图
//         edgeFrontMapId = mpSLAM->GetAtlas()->GetCurrentMap()->GetId();
//         edgeBackMapId = edgeFrontMapId + 1;
//     }
    
//     ROS_INFO("\033[1;32m[Cloud Trigger] Corrected Topology IDs -> FrontMapID: %d, BackMapID: %d\033[0m", edgeFrontMapId, edgeBackMapId);
//     if (this->mpCloudImagesActionClient) { 
//         sensor_msgs::CameraInfo cameraInfo = getCameraInfo(mpSLAM->GetSetting()->newImSize().width, mpSLAM->GetSetting()->newImSize().height, mpSLAM->GetCamera()->toK_()); 
//         cloud_edge_slam::Sequence imageSeqMsg;
//         imageSeqMsg.Header.stamp = ros::Time::now();
//         imageSeqMsg.camera = cameraInfo;
//         imageSeqMsg.edge_front_map_mnid = edgeFrontMapId;
//         imageSeqMsg.edge_back_map_mnid = edgeBackMapId;

//         for (auto &img : vCurrentProcessCloudImages) {
//             imageSeqMsg.timestamps.push_back(img.timestamp);
//         }
    
//         int total_imgs = vCurrentProcessCloudImages.size();
//         std::vector<sensor_msgs::CompressedImagePtr> msgs_buffer(total_imgs);
        
//         #pragma omp parallel for
//         for (int i = 0; i < total_imgs; ++i) {
//             const auto &image = vCurrentProcessCloudImages[i];
//             sensor_msgs::CompressedImagePtr msg(new sensor_msgs::CompressedImage());
//             msg->header.stamp = ros::Time(image.timestamp);
//             msg->header.frame_id = image.type; 
//             msg->format = "jpeg";
    
//             std::vector<int> params; 
//             params.push_back(cv::IMWRITE_JPEG_QUALITY); 
//             params.push_back(90); 
//             cv::imencode(".jpg", image.img, msg->data, params);
//             msgs_buffer[i] = msg;
//         }
    
//         goal.sequence = imageSeqMsg;
//         goal.total_image_count = total_imgs; 
//         mpCloudImagesActionClient->sendGoal(goal,
//                                             boost::bind(&Grabber::ActionFinishCb, this, _1, _2),
//                                             CloudClient::SimpleActiveCallback(),
//                                             CloudClient::SimpleFeedbackCallback());
    
//         ros::Duration(1.0).sleep(); 
//         ROS_INFO("Start send %d images by Topic...", total_imgs);
//         ros::Rate burst_rate(100.0); 
//         for (auto &msg : msgs_buffer) {
//             if (mpImageTopicPub->getNumSubscribers() > 0 || true) {
//                 mpImageTopicPub->publish(msg);
//             }
//             ros::spinOnce();
//             burst_rate.sleep();
//         }
        
//         if (mbWaitCloudResult) {
//             sleep(10);
//             mpCloudImagesActionClient->waitForResult();
//             usleep(1000 * 1e3);
//             while (mpSLAM->GetCloudMerger()->isRunning()) {
//                 usleep(100 * 1e3);
//             }
//         }
//     }
// }


void Grabber::UploadLostImages(double start_time, double end_time) {
    std::vector<ORB_SLAM3::CloudImage> vCurrentProcessCloudImages;
    
    // 1. 正确提取历史子图 ID
    int edgeFrontMapId = 0;
    int edgeBackMapId = 0;
    ORB_SLAM3::Map* pFrontMap = nullptr;

    if (mpSLAM->GetAtlas()->CountMaps() >= 2) {
        pFrontMap = mpSLAM->GetAtlas()->GetLastMap();
        edgeFrontMapId = pFrontMap->GetId();
        edgeBackMapId = mpSLAM->GetAtlas()->GetCurrentMap()->GetId();
    } else {
        pFrontMap = mpSLAM->GetAtlas()->GetCurrentMap();
        edgeFrontMapId = pFrontMap->GetId();
        edgeBackMapId = edgeFrontMapId + 1;
    }

    // ==========================================================
    // 【架构优化】：将地图查询（内部含地图锁）移出图像池锁的范围，防止 AB-BA 死锁
    // ==========================================================
    double adjusted_start_time = start_time;
    if (pFrontMap != nullptr) {
        std::vector<ORB_SLAM3::KeyFrame*> vpKFs = pFrontMap->GetAllKeyFrames();
        double last_kf_ts = 0.0;
        for (auto pKF : vpKFs) {
            if (pKF) {
                if (!pKF->isBad()) {
                    if (pKF->mTimeStamp > last_kf_ts) {
                        last_kf_ts = pKF->mTimeStamp;
                    }
                }
            }
        }
        // 如果最后一帧时间确实早于 warning，且不是异常值，则左移起始时间
        if (last_kf_ts > 0.1 && last_kf_ts < adjusted_start_time) {
            ROS_INFO("\033[1;33m[Time Sync] Auto-shifting cloud start time to %.3f to match EdgeFrontMap.\033[0m", last_kf_ts);
            adjusted_start_time = last_kf_ts;
        }
    }

    // ==========================================================
    // 2. 仅在物理提取图像时加锁，保证图像池内存安全
    // ==========================================================
    {
        std::lock_guard<std::mutex> lock(mBufMutex);
        
        auto it_start = mImageBuffer.lower_bound(adjusted_start_time - 0.05); 
        auto it_end = mImageBuffer.upper_bound(end_time + 0.05);
        
        double last_extracted_time = -1.0;
        
        // 【30Hz 满帧输出配置】
        // 彻底取消强制降频，以提供最密集的 DROID-SLAM 观测
        const double MIN_EXTRACTION_INTERVAL = 0.0; 

        // 【物理锚点硬绑定】
        // 保证云端 DROID-SLAM 重建的起点与前段地图时间的数学绝对一致性
        auto exact_it = mImageBuffer.lower_bound(adjusted_start_time - 0.005);
        if (exact_it != mImageBuffer.end()) {
            if (std::abs(exact_it->first - adjusted_start_time) < 0.01) {
                ORB_SLAM3::CloudImage ci(exact_it->second.clone(), exact_it->first, "RGB");
                vCurrentProcessCloudImages.push_back(ci);
                last_extracted_time = exact_it->first; 
                ROS_INFO("\033[1;32m[Time Sync] Hard-anchored Exact Frame for ICP: %.3f\033[0m", exact_it->first);
            }
        }

        // 【30Hz 缓冲池全量遍历】
        for (auto it = it_start; it != mImageBuffer.end() && it != it_end; ++it) {
            if (last_extracted_time > 0.0) {
                if ((it->first - last_extracted_time) <= MIN_EXTRACTION_INTERVAL) {
                    continue; 
                }
            }
            last_extracted_time = it->first;

            ORB_SLAM3::CloudImage ci(it->second.clone(), it->first, "RGB");
            vCurrentProcessCloudImages.push_back(ci);
        }
    }
    
    if (vCurrentProcessCloudImages.empty()) {
        ROS_WARN("No images found in buffer for the lost period!");
        return;
    }
    
    ROS_INFO("Extracted %zu images for Cloud Upload. Action Client Triggered.", vCurrentProcessCloudImages.size());

    // 发送动作客户端（保留原有逻辑）
    if (this->mpCloudImagesActionClient) { 
        sensor_msgs::CameraInfo cameraInfo = getCameraInfo(mpSLAM->GetSetting()->newImSize().width, mpSLAM->GetSetting()->newImSize().height, mpSLAM->GetCamera()->toK_()); 
        cloud_edge_slam::Sequence imageSeqMsg;
        imageSeqMsg.Header.stamp = ros::Time::now();
        imageSeqMsg.camera = cameraInfo;
        imageSeqMsg.edge_front_map_mnid = edgeFrontMapId;
        imageSeqMsg.edge_back_map_mnid = edgeBackMapId;

        for (auto &img : vCurrentProcessCloudImages) {
            imageSeqMsg.timestamps.push_back(img.timestamp);
        }
    
        int total_imgs = vCurrentProcessCloudImages.size();
        std::vector<sensor_msgs::CompressedImagePtr> msgs_buffer(total_imgs);
        
        #pragma omp parallel for
        for (int i = 0; i < total_imgs; ++i) {
            const auto &image = vCurrentProcessCloudImages[i];
            sensor_msgs::CompressedImagePtr msg(new sensor_msgs::CompressedImage());
            msg->header.stamp = ros::Time(image.timestamp);
            msg->header.frame_id = image.type; 
            msg->format = "jpeg";
            std::vector<int> params; 
            params.push_back(cv::IMWRITE_JPEG_QUALITY); 
            params.push_back(90); 
            cv::imencode(".jpg", image.img, msg->data, params);
            msgs_buffer[i] = msg;
        }
    
        goal.sequence = imageSeqMsg;
        goal.total_image_count = total_imgs; 
        mpCloudImagesActionClient->sendGoal(goal,
                                            boost::bind(&Grabber::ActionFinishCb, this, _1, _2),
                                            CloudClient::SimpleActiveCallback(),
                                            CloudClient::SimpleFeedbackCallback());
    
        ros::Duration(1.0).sleep(); 
        ros::Rate burst_rate(100.0); 
        for (auto &msg : msgs_buffer) {
            if (mpImageTopicPub->getNumSubscribers() > 0 || true) {
                mpImageTopicPub->publish(msg);
            }
            ros::spinOnce();
            burst_rate.sleep();
        }
        
        if (mbWaitCloudResult) {
            sleep(10);
            mpCloudImagesActionClient->waitForResult();
            usleep(1000 * 1e3);
            while (mpSLAM->GetCloudMerger()->isRunning()) {
                usleep(100 * 1e3);
            }
        }
    }
}
