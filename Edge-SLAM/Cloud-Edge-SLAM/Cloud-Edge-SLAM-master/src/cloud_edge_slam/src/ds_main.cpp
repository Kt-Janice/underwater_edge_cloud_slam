/** 
 * This file is part of ORB-SLAM3
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel, and Juan D. Tardós, University of Zaragoza.
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License.
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY.
 */
#include "KeyFrame.h"  // 引入 KeyFrame 类头文件，用于处理关键帧
#include "sensor_msgs/image_encodings.h"  // 引入ROS消息，处理图像编码
#include <boost/filesystem/operations.hpp>  // 引入Boost库的文件系统操作
#include <boost/filesystem/path.hpp>  // 引入Boost库的路径操作
#include <cstdlib>  // 引入C标准库，包含通用函数，如内存分配
#include <cstring>  // 引入C标准库，字符串操作
#include <iostream>  // 引入输入输出流，用于打印信息
#include <algorithm>  // 引入算法库，用于常用算法，如排序等
#include <fstream>  // 引入文件流，用于读写文件
#include <chrono>  // 引入时间库，用于时间操作
#include <opencv2/imgcodecs.hpp>  // OpenCV库，用于图像编码解码
#include <sys/stat.h>  // 系统库，用于文件状态查询
#include <sys/types.h>  // 系统库，用于系统调用
#include <filesystem>  // C++17文件系统库，用于处理文件和目录操作

// ROS相关头文件，引入ROS的相关功能
#include <ros/ros.h>  // ROS主头文件
#include <rosbag/bag.h>  // ROS包操作
#include <rosbag/view.h>  // ROS包视图操作
#include <cv_bridge/cv_bridge.h>  // 用于ROS图像消息和OpenCV图像转换
#include <std_msgs/Int16.h>  // ROS消息类型，16位整型
#include <std_msgs/Int64.h>  // ROS消息类型，64位整型
#include <std_msgs/Float32.h>  // ROS消息类型，32位浮点型
#include <std_msgs/Float64.h>  // ROS消息类型，64位浮点型
#include <geometry_msgs/Pose.h>  // ROS消息类型，表示位姿
#include <geometry_msgs/PoseStamped.h>  // ROS消息类型，带时间戳的位姿
#include <geometry_msgs/Point.h>  // ROS消息类型，表示三维点
#include <geometry_msgs/Quaternion.h>  // ROS消息类型，表示四元数，用于旋转
#include <actionlib/client/simple_action_client.h>  // ROS Action客户端头文件

#include <Eigen/Core>  // 引入Eigen库，用于矩阵运算
#include <Eigen/Dense>  // 引入Eigen库的密集矩阵运算
#include <opencv2/core/core.hpp>  // OpenCV核心模块
#include <opencv2/core.hpp>  // OpenCV核心模块
#include <string>  // 字符串库
#include <unistd.h>  // UNIX标准函数库
#include <vector>  // 向量容器类

// 自定义头文件，ORB-SLAM3相关功能
#include "System.h"  // ORB-SLAM3的核心系统
#include "Map.h"  // 地图处理类
#include "Atlas.h"  // Atlas类，管理多个地图
#include "Converter.h"  // 数据转换类
#include "ORBVocabulary.h"  // ORB词汇表
#include "CloudImageSampler.h"  // 云图像采样类

#include "cloud_edge_slam/Evo.h"  // 与云边缘SLAM相关的消息类型
#include "cloud_edge_slam/CloudSlamAction.h"  // CloudSLAM Action头文件
#include "cloud_edge_slam/CloudSlamActionFeedback.h"  // CloudSLAM Action反馈
#include "cloud_edge_slam/CloudSlamActionGoal.h"  // CloudSLAM Action目标
#include "cloud_edge_slam/CloudSlamGoal.h"  // CloudSLAM目标消息
#include "cloud_edge_slam/CloudSlamResult.h"  // CloudSLAM结果消息
#include "cloud_edge_slam/Sequence.h"  // 序列消息类型
#include "cloud_edge_slam/CloudMap.h"  // 云地图类型
#include "cloud_edge_slam/KeyFrame.h"  // 关键帧消息类型
#include "cloud_edge_slam/KeyPoint.h"  // 关键点消息类型
#include "cloud_edge_slam/MapPoint.h"  // 地图点消息类型
#include "cloud_edge_slam/Descriptor.h"  // 描述符消息类型
#include "cloud_edge_slam/Observation.h"  // 观测消息类型

// ROS初始化和消息操作
#include "ros/init.h"  // ROS初始化头文件
#include "ros/message_traits.h"  // ROS消息特征
#include "ros/node_handle.h"  // ROS节点句柄
#include "ros/publisher.h"  // ROS发布器实际通过网络发给云端的是内存中的消息对
#include "sensor_msgs/Image.h"  // 图像消息
//cap-udf
#include<fstream>  // 引入文件流，用于文件操作

// ROS参数获取宏定义，简化ROS程序中的参数获取过程，并进行错误处理
#define getRosParam(nh, param_name, output)                                     \
    if (!nh.getParam(param_name, output)) {  // 如果获取参数失败，输出错误信息并处理
        ROS_ERROR_STREAM("!ERROR! cannot get necessary param: " << param_name); 
        exit(100);                                                              
        }
        namespace bfs = boost::filesystem;  // 将boost::filesystem命名空间简化为bfs，方便使用文件系统操作
        using namespace std;  // 使用标准命名空间，简化代码书写
        
        // 定义CloudClient类型，基于actionlib库的SimpleActionClient
        typedef actionlib::SimpleActionClient<cloud_edge_slam::CloudSlamAction> CloudClient;
        // typedef actionlib::SimpleActionClient<actionlib_tutorials::FibonacciAction> CloudClient;  // 示例代码：另一个类型定义，未启用
        
        // Grabber类：处理与SLAM系统、图像、云图等相关的操作
        class Grabber {
        public:
            Grabber(ORB_SLAM3::System *pSLAM) :  // 构造函数，接受ORB_SLAM3系统指针
                mpSLAM(pSLAM)  // 初始化成员变量mpSLAM
                {  

                }
        
            // **********************************************
            // @note init
            void SetParameters(bool bWaitCloudResult, float nMainLoopSleep, string savePath, bool bSaveCloudBag, bool bOldUdf, bool bNewUdf);
            void SetNodeHandle(ros::NodeHandle *pNodeHandle);  // 设置ROS节点句柄
            void SetOrbMapPublisher(ros::Publisher *pPublisher);  // 设置ORB地图发布器
            void SetCloudImagesActionClient(CloudClient *pCloudImagesActionClient);  // 设置云图像Action客户端
            // **********************************************
        
            // **********************************************
            // @note Main Tracking
            void RunBag(const string &bag_path);  // 处理ROS包文件
            void RunTxt(const string &txt_path);  // 处理文本文件
            void GrabImage(const sensor_msgs::ImageConstPtr &msg);  // 获取图像数据
            // **********************************************
        
            // **********************************************
            // @note Main Cloud Interact
            void TrackImage(const cv::Mat &img, const double &timestamp, const float &imageScale);  // 追踪图像
            void ActionFinishCb(const actionlib::SimpleClientGoalState &state, const cloud_edge_slam::CloudSlamResultConstPtr &result);  // 处理Action完成回调
            void MemoryCb(const std_msgs::Float32ConstPtr &msg);  // 处理内存回调
            // **********************************************
        
            // **********************************************
            // @note For Debug
            void PubORBMapCb(const std_msgs::Int16ConstPtr &msg);  // 发布ORB地图回调
            void SaveORBMapCb(const std_msgs::Int16ConstPtr &msg);  // 保存ORB地图回调
            void GrabCloudMapCb(const cloud_edge_slam::CloudMapConstPtr &msg);  // 获取云地图回调
            // **********************************************
        
            // **********************************************
            // @note utils
            static cloud_edge_slam::CloudMap ORBMapToROSMap(ORB_SLAM3::Map *pMap);  // 将ORB地图转换为ROS地图
            ORB_SLAM3::Map *ROSMapToORBMap(cloud_edge_slam::CloudMapConstPtr pMap);  // 将ROS地图转换为ORB地图
            static void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps);  // 加载图像
            static void WriteCloudMapBag(cloud_edge_slam::CloudMap map, const std::string &save_path);  // 保存云地图为ROS包
            static void WriteCloudImagesBag(std::vector<ORB_SLAM3::CloudImage> &vImages, const std::string &save_path);  // 保存云图像为ROS包
            static void WriteSeqBag(const cloud_edge_slam::Sequence &seq, const std::string &save_path);  // 保存序列为ROS包
            // **********************************************
        
        public:
            cloud_edge_slam::CloudSlamGoal goal;  // CloudSLAM目标
            ORB_SLAM3::System *mpSLAM;  // 指向ORB_SLAM3系统的指针
            ros::NodeHandle *mpNodeHandler;  // ROS节点句柄
            ros::Publisher *mpOrbMapPub;  // ROS发布器指针
            ros::Subscriber *mpMemorySub;
            CloudClient *mpCloudImagesActionClient = NULL;
            Grabber
            std::vector<float> mvMemory;
            std::map<double, float> mvBagSize;
            std::map<double, float> mvNoSamplingBagSize;
    
            bool mbWaitCloudResult;
            std::string mSaveDir;
            float mnMainLoopSleep;
            bool mbSaveCloudBag;
            bool mbOldUdf; //cap-udf
            bool mbNewUdf; //cap-udf
    };
    
    std::vector<std::string> SplitPath(std::string path) {
        std::vector<std::string> tokens;
        std::string token;
        std::stringstream ss(path);
        while (getline(ss, token, '/')) 
        {
            tokens.push_back(token);
        }
        return tokens; //将path的内容一个一个存入，如home、sylva、...
    }
    
    int id = 0;
// main函数：Cloud-Edge-SLAM系统的主入口点
int main(int argc, char **argv) 
{
    // **********************************************
    // @note ROS系统初始化
    // ros::init(argc, argv, "Mono");
    //auto start = std::chrono::high_resolution_clock::now();
    ros::init(argc, argv, "Mono-temp");  // 初始化ROS节点，使用全局命名空间
    ros::start();  // 启动ROS节点
    cerr <<"-----------------------------------------------Cloud-Edge-SLAM RUN STAR!--------------------------------------------------------"<< endl;//cerr不经过缓冲而直接输出，一般用于迅速输出出错信息
    ros::Time time_begin = ros::Time::now();  // 记录开始时间
    ros::NodeHandle nh("~");  // 创建局部命名空间节点句柄，对应launch文件中的node
    // **********************************************

    // **********************************************
    // @note 参数声明和获取
    // Debug: Fix param
    // string vocabularyPath = "/home/red0orange/github_projects/Cloud-Edge-SLAM/src/cloud_edge_slam/Vocabulary/ORBvoc.txt";
    // string settingPath = "/home/red0orange/github_projects/Cloud-Edge-SLAM/src/cloud_edge_slam/config/TUM2.yaml";
    // string dataType = "bag";
    // string dataPath = "/media/red0orange/Data/数据集/CloudEdge/rgbd_dataset_freiburg2_pioneer_360_cut_0-24s.bag";
    // bool bRealOnline = true;
    // bool bCloudOnline = true;
    // bool bSetImgGray = false;

    string cloudTopicName;
    getRosParam(nh, "cloud_topic_name", cloudTopicName);  // 获取云端话题名称参数
    string vocabularyPath;
    string settingPath;
    getRosParam(nh, "vocabulary_path", vocabularyPath);  // 获取ORB词汇表文件路径参数
    getRosParam(nh, "setting_path", settingPath);  // 获取配置文件路径参数
    string dataType;
    string dataPath;
    string resultPath;
    getRosParam(nh, "data_type", dataType);  // 获取数据类型参数(bag、txt)
    getRosParam(nh, "data_path", dataPath);  // 获取数据集路径参数
    getRosParam(nh, "result_path", resultPath);  // 获取结果保存路径参数
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
    getRosParam(nh, "cloud_merge", bCloudMerge);  // 获取云端合并功能开关参数
    getRosParam(nh, "save_cloud_bag", bSaveCloudBag);  // 获取保存云端数据包开关参数
    getRosParam(nh, "real_online", bRealOnline);  // 获取等待云端连接成功后初始化开关参数
    getRosParam(nh, "merge_anyway", bMergeAnyway);  // 获取强制合并开关参数
    getRosParam(nh, "cloud_online", bCloudOnline);  // 获取云端在线模式开关参数
    getRosParam(nh, "wait_cloud_result", bWaitCloudResult);  // 获取等待云端结果开关参数
    getRosParam(nh, "main_loop_sleep_ms", mainLoopSleep);  // 获取主循环睡眠时间参数
    getRosParam(nh, "kf_culling", bKFCulling);  // 获取关键帧筛选开关参数
    //cap-udf
    bool bOldUdf;
    bool bNewUdf;
    getRosParam(nh, "old_udf_cloud_edge", bOldUdf);  // 获取旧版UDF参数
    getRosParam(nh, "new_udf_cloud_edge", bNewUdf);  // 获取新版UDF参数
    //cap-udf
    int nSamplerEdgeFrontKFNum;
    int nSamplerEdgeBackKFNum;
    float nSamplerEdgeFrontMinTime;
    float nSamplerEdgeBackMinTime;
    float samplerPDKp;
    float samplerPDKd;
    float samplerPDth;
    getRosParam(nh, "sampler_edge_front_kf_num", nSamplerEdgeFrontKFNum);  // 获取前端关键帧采样数量参数
    getRosParam(nh, "sampler_edge_back_kf_num", nSamplerEdgeBackKFNum);  // 获取后端关键帧采样数量参数
    getRosParam(nh, "sampler_edge_front_min_time", nSamplerEdgeFrontMinTime);  // 获取前端最小时间间隔参数
    getRosParam(nh, "sampler_edge_back_min_time", nSamplerEdgeBackMinTime);  // 获取后端最小时间间隔参数
    getRosParam(nh, "sampler_pd_kp", samplerPDKp);  // 获取光流法Kp参数
    getRosParam(nh, "sampler_pd_kd", samplerPDKd);  // 获取光流法Kd参数
    getRosParam(nh, "sampler_pd_th", samplerPDth);  // 获取光流法阈值参数
    // **********************************************

    // **********************************************
    // @note 路径处理和结果目录创建
    std::vector<string> splitPaths = SplitPath(dataPath);  // 分离数据集路径
    string datasetName = splitPaths[splitPaths.size() - 2];  // 提取数据集名称，eg:splitPaths = ["", "home", "user", "datasets", "kitti", "sequence00"]，splitPaths[kitti]

    auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());  // 获取当前时间戳,to_time_t()将时间点转换为C风格的time_t类型,auto自动类型推导，t的类型是time_t
    std::stringstream ss;
    ss << std::put_time(std::localtime(&t), "%m-%d-%H_%M_%S");  // 格式化时间字符串,eg:"07-15-14_30_25",std::localtime(&t)将time_t转换为本地时间的tm结构,std::put_time()时间格式化函数
    std::string str_time = ss.str();

    boost::filesystem::path parent_dir(resultPath);
    boost::filesystem::path sub_dir(datasetName + "#" + str_time);  // 创建子目录名
    boost::filesystem::path finish_sub_dir("Full#" + datasetName + "#" + str_time);  // 创建完成目录名
    boost::filesystem::path full_path_ = parent_dir / sub_dir;  // 组合完整路径
    boost::filesystem::path full_finish_path_ = parent_dir / finish_sub_dir;  // 组合完成路径
    string full_path = full_path_.string();  // 转换为字符串路径
    string full_finish_path = full_finish_path_.string();
    int isCreate = mkdir(full_path.c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);  // 创建结果目录,权限：S_IRUSR读权限;S_IWUSR写权限;S_IXUSR执行权限;S_IRWXG组权限（读、写、执行);S_IRWXO其他用户权限（读、写、执行）
    cerr << "create path: " << full_path << endl;
    if (!isCreate)//mkdir()函数通常返回 int 类型：成功时返回0,失败时返回-1
        cerr << "create path finish" << endl;
    else
        cerr << "create path failed! error code: !!!!!!!!!!!!" << isCreate << endl;
    // **********************************************

    // **********************************************
    // @note SLAM系统初始化
    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(vocabularyPath, settingPath, ORB_SLAM3::System::MONOCULAR, true, bCloudMerge, bCloudOnline, bMergeAnyway, bKFCulling, nSamplerEdgeFrontKFNum, nSamplerEdgeBackKFNum, nSamplerEdgeFrontMinTime, nSamplerEdgeBackMinTime, samplerPDKp, samplerPDKd, samplerPDth, bOldUdf, bNewUdf);  // 初始化ORB-SLAM3系统
    Grabber igb(&SLAM);  // 创建Grabber实例，igb变量名，可能是image_grabber缩写

    igb.SetParameters(bWaitCloudResult, mainLoopSleep, full_path, bSaveCloudBag, bOldUdf, bNewUdf);  // 设置Grabber参数

    igb.SetNodeHandle(&nh);  // 设置节点句柄
    // **********************************************

    // **********************************************
    // @note ROS发布器和订阅器设置
    ros::Publisher orbMapPub = nh.advertise<cloud_edge_slam::CloudMap>("/test_cloud_map", 1);  // 创建ORB地图发布器,参数2:队列中最大保存的消息数，超出此阀值时，先进的先销毁
    igb.SetOrbMapPublisher(&orbMapPub);  // 设置ORB地图发布器

    ros::ServiceClient evoClient = nh.serviceClient<cloud_edge_slam::Evo>("/cloud_edge_evo_temp");  // 创建评估服务客户端

    ros::Subscriber memorySub = nh.subscribe("/cloud_edge_memory_temp", 1, &Grabber::MemoryCb, &igb);  // 创建内存订阅器
    //&Grabber::MemoryCb: 回调函数指针，指向Grabber类的成员函数MemoryCb。当收到消息时，会调用这个函数;&igb: 回调函数所属的对象指针，即Grabber类的一个实例（igb）。当收到消息时，会调用该实例的MemoryCb成员函数。
    
    // For Pub Cloud Images
    // CloudClient cloudImagesActionClient("/test_pub_cloud_images", true);
    CloudClient cloudImagesActionClient(cloudTopicName, true);  // 创建云端图像Action客户端
    if (bRealOnline) 
    {
        ROS_INFO_STREAM("Waiting For Server Begin!");  // 等待服务器连接开始
        cloudImagesActionClient.waitForServer();  // 等待Action服务器
        ROS_INFO_STREAM("Waiting For Server End!");  // 等待服务器连接结束
        igb.SetCloudImagesActionClient(&cloudImagesActionClient);  // 设置云端图像Action客户端
    }

    // For Get Test Publish Map
    ros::Subscriber save_orb_map_sub = nh.subscribe("/test_save_orb_map", 1, &Grabber::SaveORBMapCb, &igb);  // 创建保存ORB地图订阅器
    // // For Get Test Publish Map
    // ros::Subscriber pub_orb_map_sub = nodeHandler.subscribe("/test_pub_orb_map", 1, &Grabber::PubORBMap, &igb);

    // // For rosbag online play
    // ros::Subscriber image_sub = nodeHandler.subscribe("/camera/rgb/image_color", 1, &Grabber::GrabImage, &igb);

    ros::Subscriber cloud_map_sub = nh.subscribe("/test_cloud_map", 1, &Grabber::GrabCloudMapCb, &igb);  // 创建云地图订阅器
    // **********************************************

    // **********************************************
    // @note 主运行逻辑
    std::chrono::steady_clock::time_point timeStart = std::chrono::steady_clock::now();  // 记录运行开始时间
    if (bCloudOnline) {
        if (dataType == "txt") {
            igb.RunTxt(dataPath);  // 运行文本数据模式
        } else if (dataType == "bag") {
            igb.RunBag(dataPath);  // 运行ROS包数据模式
        } else {
            ROS_ERROR_STREAM("Error Data Type!");  // 数据类型错误
            exit(1000);  // 退出程序
        }
    }
    std::chrono::steady_clock::time_point timeEnd = std::chrono::steady_clock::now();  // 记录运行结束时间
    double duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(timeEnd - timeStart).count();  // 计算运行时长
    //std::chrono::duration_cast将时间差转换为以毫秒为单位（std::milli）的duration，并且用double表示
    //std::chrono::duration_cast持续时间类型转换函数,std::chrono::duration<double, std::milli>为目标类型
    duration /= 1e3;  // 转换为秒

    /*
    测量SLAM处理一帧的时间
    auto start = std::chrono::high_resolution_clock::now();
    slam->ProcessFrame(frame);
    auto end = std::chrono::high_resolution_clock::now();
    double process_time = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
    std::cout << "帧处理时间: " << process_time << " ms" << std::endl;
    */
    
    // @note 结果导出和评估
    // begin export
    std::string map_point = "/home/birl/Udf-Edge/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/src/cloud_edge_slam/results/point.txt";  // 地图点结果文件路径
    std::ofstream fout_map_point(map_point);  // 打开结果文件
    fout_map_point << SLAM.GetAtlas()->MapPointsInMap() << endl;  // 写入地图点数量
    fout_map_point << duration << endl;  // 写入运行时长
    fout_map_point.close();  // 关闭文件

    ORB_SLAM3::CloudSaveKeyFrameTrajectoryTUM(SLAM.GetAtlas()->GetDurationLongestMap(), (full_path_ / "whole_map.txt").string());  // 保存关键帧轨迹
    std::vector<double> vTimestammp;
    std::vector<Eigen::Vector3f> vPosition;
    std::vector<Eigen::Quaternionf> vQuaternion;
    ORB_SLAM3::Map *pLongestMap = SLAM.GetAtlas()->GetDurationLongestMap();  // 获取最长持续时间的地图
    std::vector<ORB_SLAM3::KeyFrame *> vpKFs = pLongestMap->GetSortedKeyFrames();  // 获取排序后的关键帧
    ORB_SLAM3::CloudExportKeyFrameTrajectoryTUM(pLongestMap, vTimestammp, vPosition, vQuaternion);  // 导出关键帧轨迹数据
    //---------------------------评估--------------------------------------------------------------------------------
    /*cloud_edge_slam::Evo evo_srv;
    evo_srv.request.dataset_name = datasetName;
    evo_srv.request.use_ref = false;
    evo_srv.request.poses.clear();
    for (int i = 0; i < vTimestammp.size(); i++) {
        double timestamp = vTimestammp[i];
        Eigen::Vector3f position = vPosition[i];
        Eigen::Quaternionf quaternion = vQuaternion[i];
        geometry_msgs::PoseStamped pose;
        pose.header.stamp = ros::Time(timestamp);
        pose.pose.position.x = position(0);
        pose.pose.position.y = position(1);
        pose.pose.position.z = position(2);
        pose.pose.orientation.x = quaternion.x();
        pose.pose.orientation.y = quaternion.y();
        pose.pose.orientation.z = quaternion.z();
        pose.pose.orientation.w = quaternion.w();
        evo_srv.request.poses.push_back(pose);
    }
    if (evoClient.call(evo_srv)) {
        cerr << "evo client success" << endl;

        ofstream outFile;
        outFile.open((full_path_ / "result.csv").string(), ios::out); // 打开模式可省略
        outFile << "ate" << ',' << evo_srv.response.ate << endl;
        outFile << "rate" << ',' << evo_srv.response.rate << endl;
        outFile << "duration" << ',' << duration << endl;
        outFile << "max_memory" << ',' << *max_element(igb.mvMemory.begin(), igb.mvMemory.end()) << endl;

        float sumBagSize = 0;
        for (auto &iter : igb.mvBagSize) {
            if (iter.first < vpKFs[vpKFs.size() - 1]->mTimeStamp && iter.first > vpKFs[0]->mTimeStamp) {
                sumBagSize += iter.second;
            }
        }
        float sumNoSamplingBagSize = 0;
        for (auto &iter : igb.mvNoSamplingBagSize) {
            if (iter.first < vpKFs[vpKFs.size() - 1]->mTimeStamp && iter.first > vpKFs[0]->mTimeStamp) {
                sumNoSamplingBagSize += iter.second;
            }
        }
        outFile << "sum_bag_size" << ',' << sumBagSize << endl;
        outFile << "sum_nosampling_bag_size" << ',' << sumNoSamplingBagSize << endl;
        outFile << "ref_traj_duration" << ',' << evo_srv.response.ref_traj_duration << endl;
        outFile << "est_traj_duration" << ',' << evo_srv.response.est_traj_duration << endl;
        outFile << "speed" << ',' << sumBagSize / evo_srv.response.ref_traj_duration << endl;
        outFile << "nosampling_speed" << ',' << sumNoSamplingBagSize / evo_srv.response.ref_traj_duration << endl;
        outFile.close();

        cv::imwrite((full_path_ / "traj.png").string(), cv_bridge::toCvCopy(evo_srv.response.traj_img)->image);
    }*/
    //---------------------------------------------------------------------------------------------------------------------
    // **********************************************

    // **********************************************
    // @note 系统清理和退出
    // rename dir for tag
    boost::filesystem::rename(full_path_, full_finish_path_);  // 重命名结果目录

    // @note 直接暂停
    exit(1000);  // 退出程序
    
    // Debug: 等待CloudMap
    ros::spin();  // ROS事件循环
    // auto finish = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration1 = finish - start;
    // std::cout <<"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
    // std::cout <<"Cloud-Edge-SLAM系统运行总时长:"<<duration1.count()<< "s\n";
    ros::Time time_end = ros::Time::now();  // 获取结束时间
    ros::Duration duration1 = time_end - time_begin;  // 计算总运行时长
    ROS_INFO("Cloud-Edge-SLAM系统运行总时长: %lf secs", duration1.toSec());  // 输出运行时长
    SLAM.Shutdown();  // 关闭SLAM系统
    ros::shutdown();  // 关闭ROS节点
    return 0;  // 程序正常退出
    // **********************************************
}
// **********************************************
// @note 工具函数
// toSophusPose函数：将ROS几何位姿转换为Sophus位姿
inline Sophus::SE3f toSophusPose(const geometry_msgs::Pose rosPose) {
    Eigen::Quaternion<float> quaterniond(rosPose.orientation.w, rosPose.orientation.x, rosPose.orientation.y, rosPose.orientation.z);  // 从ROS位姿提取四元数
    Eigen::Vector3f pos(rosPose.position.x, rosPose.position.y, rosPose.position.z);  // 从ROS位姿提取位置向量
    Sophus::SE3f pose(quaterniond, pos);  // 创建Sophus位姿对象
    return pose;  // 返回Sophus位姿
}

// getCameraInfo函数：根据图像尺寸和内参矩阵生成相机信息
inline sensor_msgs::CameraInfo getCameraInfo(int imageWidth, int imageHeight, Eigen::Matrix3f K) { // extract cameraInfo.
    sensor_msgs::CameraInfo cam;  // 创建相机信息对象

    // vector<double> D{0.000094, -0.011701, 0.000383, -0.000507, 0.000000};
    boost::array<double, 9> K_array = {  // 将内参矩阵转换为数组格式
        K(0, 0), K(0, 1), K(0, 2),
        K(1, 0), K(1, 1), K(1, 2),
        K(2, 0), K(2, 1), K(2, 2)};

    // boost::array<double, 12> P = {
    //     402.124725, 0.000000, 335.482488, 0.000000,
    //     0.000000, 403.765045, 250.954855, 0.000000,
    //     0.000000, 0.000000, 1.000000, 0.000000};
    // boost::array<double, 9> r = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    cam.width = imageWidth;  // 设置图像宽度
    cam.height = imageHeight;  // 设置图像高度
    cam.distortion_model = "plumb_bob";  // 设置畸变模型
    cam.K = K_array;  // 设置内参矩阵
    // cam.D = D;
    // cam.P = P;
    // cam.R = r;
    cam.binning_x = 0;  // 设置水平像素合并
    cam.binning_y = 0;  // 设置垂直像素合并
    cam.header.frame_id = "camera";  // 设置坐标系ID为camera
    cam.header.stamp = ros::Time::now();  // 设置时间戳
    cam.header.stamp.nsec = 0;  // 设置纳秒为0
    return cam;  // 返回相机信息
}
// **********************************************

// **********************************************
// @note Grabber类设置函数
// SetCloudImagesActionClient函数：设置云端图像Action客户端
void Grabber::SetCloudImagesActionClient(CloudClient *pCloudImagesActionClient) {
    mpCloudImagesActionClient = pCloudImagesActionClient;  // 设置云端图像Action客户端指针
}

// SetParameters函数：设置Grabber参数
void Grabber::SetParameters(bool bWaitCloudResult, float nMainLoopSleep, string savePath, bool bSaveCloudBag, bool bOldUdf, bool bNewUdf) {
    mbWaitCloudResult = bWaitCloudResult;  // 设置是否等待云端结果标志
    mnMainLoopSleep = nMainLoopSleep;  // 设置主循环睡眠时间
    mSaveDir = savePath;  // 设置保存目录
    mbSaveCloudBag = bSaveCloudBag;  // 设置是否保存云端数据包标志
    mbOldUdf = bOldUdf;  // 设置是否使用旧版UDF标志
    mbNewUdf = bNewUdf;  // 设置是否使用新版UDF标志
}

// SetNodeHandle函数：设置ROS节点句柄
void Grabber::SetNodeHandle(ros::NodeHandle *pNodeHandle) {
    mpNodeHandler = pNodeHandle;  // 设置ROS节点句柄指针
}

// SetOrbMapPublisher函数：设置ORB地图发布器
void Grabber::SetOrbMapPublisher(ros::Publisher *pPublisher) {
    mpOrbMapPub = pPublisher;  // 设置ORB地图发布器指针
}
// **********************************************

// **********************************************
// @note 图像追踪处理
// TrackImage函数：处理单帧图像追踪
void Grabber::TrackImage(const cv::Mat &img, const double &timestamp, const float &imageScale) 
{
    if (imageScale != 1.f) {  // 检查是否需要缩放图像
        int width = img.cols * imageScale;  // 计算缩放后宽度
        int height = img.rows * imageScale;  // 计算缩放后高度
        cv::resize(img, img, cv::Size(width, height));  // 执行图像缩放
    }

    static int cnt = 0;  // 静态计数器
    ROS_INFO_STREAM_DELAYED_THROTTLE(1.0, "Deal One Image: " << cnt);  // 节流输出处理图像计数
    cnt++;  // 计数器递增

    // @note input image
    mpSLAM->TrackMonocular(img, timestamp);  // 调用SLAM系统进行单目追踪

    // return;

    // 主动式检测Cloud Image
    std::vector<ORB_SLAM3::CloudImage> vCurrentProcessCloudImages;  // 当前处理的云端图像向量
    std::vector<ORB_SLAM3::CloudImage> vCurrentProcessCloudNoSamplingImages;  // 当前处理的未采样云端图像向量
    int edgeFrontMapId, edgeBackMapId;  // 前端和后端地图ID
    mpSLAM->GetCloudProcessImages(vCurrentProcessCloudImages, vCurrentProcessCloudNoSamplingImages, edgeFrontMapId, edgeBackMapId);  // 获取云端处理图像
    if (!vCurrentProcessCloudImages.empty())  // 检查是否有云端图像需要处理
    {
        // @note print
        cout << "Main: Pub Cloud Images !" << endl;  // 输出发布云端图像信息
        // debug: write cloud images
        static int index = 0;  // 静态索引
        index++;  // 索引递增
        if (mbSaveCloudBag) {  // 检查是否需要保存云端数据包
            string curCloudBagPath = (bfs::path(mSaveDir) / "cloud_").string() + to_string(index) + ".bag";  // 生成云端数据包路径
            WriteCloudImagesBag(vCurrentProcessCloudImages, curCloudBagPath);  // 写入云端图像数据包
            string curCloudNoSamplingBagPath = (bfs::path(mSaveDir) / "cloud_nosampling_").string() + to_string(index) + ".bag";  // 生成未采样云端数据包路径
            WriteCloudImagesBag(vCurrentProcessCloudNoSamplingImages, curCloudNoSamplingBagPath);  // 写入未采样云端图像数据包

            double curTimestamp = vCurrentProcessCloudImages[vCurrentProcessCloudImages.size() - 1].timestamp; //获取当前时间戳;
            //我们有一个向量vCurrentProcessCloudImages，它包含一些元素，每个元素有一个timestamp成员。这行代码的目的是获取向量中最后一个元素的timestamp，并将其赋值给变量curTimestamp。
            auto bagSize = float(bfs::file_size(curCloudBagPath) / (1024 * 1024));  // 计算数据包大小(MB)
            mvBagSize[curTimestamp] = bagSize;  // 记录数据包大小
            auto noSamplingBagSize = float(bfs::file_size(curCloudNoSamplingBagPath) / (1024 * 1024));  // 计算未采样数据包大小(MB)
            mvNoSamplingBagSize[curTimestamp] = noSamplingBagSize;  // 记录未采样数据包大小
        }

        // @note Publish Cloud Images Action
        if (this->mpCloudImagesActionClient)  // 检查云端Action客户端是否可用
        {
            // @note print
            ROS_INFO_STREAM("Action Client");  // 输出Action客户端信息
            sensor_msgs::CameraInfo cameraInfo = getCameraInfo(mpSLAM->GetSetting()->newImSize().width, mpSLAM->GetSetting()->newImSize().height, mpSLAM->GetCamera()->toK_());  // 获取相机信息

            cloud_edge_slam::Sequence imageSeqMsg;  // 创建图像序列消息
            imageSeqMsg.Header.stamp = ros::Time::now();  // 设置消息时间戳
            imageSeqMsg.camera = cameraInfo;  // 设置相机信息
            imageSeqMsg.edge_front_map_mnid = edgeFrontMapId;  // 设置前端地图ID
            imageSeqMsg.edge_back_map_mnid = edgeBackMapId;  // 设置后端地图ID
            for (auto &image : vCurrentProcessCloudImages)  //  遍历所有云端图像
            {
                std_msgs::Header header;  // 创建消息头
                header.stamp = ros::Time(image.timestamp);  // 设置图像时间戳
                header.frame_id = image.type;  // 设置图像类型
                string encode;  // 编码格式
                if (image.img.channels() == 3)  // 检查是否为3通道图像
                    encode = "bgr8";  // 设置BGR编码
                else if (image.img.channels() == 1)  // 检查是否为单通道图像
                    encode = "mono8";  // 设置单色编码
                sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, encode, image.img).toImageMsg();  // 将OpenCV图像转换为ROS图像消息

                imageSeqMsg.images.push_back(*msg);  // 添加图像到序列
                imageSeqMsg.timestamps.push_back(image.timestamp);  // 添加时间戳到序列
            }

            // debug bandwidth: write seq
            // WriteSeqBag(imageSeqMsg, "/home/ruanjh/Workspace/NewSpace/Cloud-Edge-SLAM/results/test_seq.bag");
            cout << "Now Pub Cloud Images !" << endl;  // 输出发布云端图像信息
            goal.sequence = imageSeqMsg;  // 设置目标序列
            mpCloudImagesActionClient->sendGoal(goal,  // 发送目标到Action服务器
                                                boost::bind(&Grabber::ActionFinishCb, this, _1, _2),  // 设置完成回调
                                                CloudClient::SimpleActiveCallback(),  // 设置激活回调
                                                CloudClient::SimpleFeedbackCallback());  // 设置反馈回调

            // @note 等待CloudMerging结束再继续
            if (mbWaitCloudResult) 
            {
                sleep(10);  // 睡眠10秒
                mpCloudImagesActionClient->waitForResult();  // 等待Action结果
                usleep(1000 * 1e3);  // 睡眠1秒
                while (mpSLAM->GetCloudMerger()->isRunning()) {  // 检查云端合并器是否在运行
                    usleep(100 * 1e3);  // 睡眠100毫秒
                }
            }

            mpSLAM->SetTrackLostTimestamp(timestamp);  // 设置追踪丢失时间戳
        }
        mpSLAM->ResetCloudProcessImages();  // 重置云端处理图像
        cout << "Finish Pub Cloud Images !" << endl;  // 输出发布云端图像信息
    }
}
// **********************************************

// **********************************************
// @note 文本数据运行模式
// RunTxt函数：处理文本格式的数据集
void Grabber::RunTxt(const string &txt_path) {
    string data_dir = txt_path.substr(0, txt_path.rfind('/'));  // 提取数据目录路径
    // Retrieve paths to images
    vector<string> vstrImageFilenames;  // 图像文件名向量
    vector<double> vTimestamps;  // 时间戳向量
    LoadImages(txt_path, vstrImageFilenames, vTimestamps);  // 加载图像信息

    int nImages = vstrImageFilenames.size();  // 获取图像总数
    float imageScale = mpSLAM->GetImageScale();  // 获取图像缩放比例

    // Vector for tracking time statistics
    vector<float> vTimesTrack;  // 追踪时间统计向量
    vTimesTrack.resize(nImages);  // 调整向量大小

    cout << endl
        << "-------" << endl;
    cout << "Start processing sequence ..." << endl;  // 输出开始处理序列信息
    cout << "Images in the sequence: " << nImages << endl
        << endl;

    double t_resize = 0.f;  // 缩放时间
    double t_track = 0.f;  // 追踪时间

    // Main loop
    cv::Mat im;  // 图像矩阵
    for (int ni = 0; ni < nImages; ni++)  // 遍历所有图像
    {
        // Read image from file
        im = cv::imread(string(data_dir) + "/" + vstrImageFilenames[ni], cv::IMREAD_UNCHANGED);  // 读取图像文件
        double tframe = vTimestamps[ni];  // 获取当前帧时间戳
        
        if (im.empty()) {  // 检查图像是否为空
            cerr << endl
                << "Failed to load image at: "
                 << string(data_dir) << "/" << vstrImageFilenames[ni] << endl;  // 输出加载失败信息
            return;  // 返回
        }
        // Pass the image to the SLAM system
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();  // 记录开始时间
        TrackImage(im, tframe, imageScale);  // 追踪图像
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();  // 记录结束时间

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();  // 计算追踪耗时

        vTimesTrack[ni] = ttrack;  // 记录追踪时间

        // Wait to load the next frame
        double T = 0;  // 时间间隔
        if (ni < nImages - 1)  // 检查是否为最后一帧,最后一帧不会进入到if函数
            T = vTimestamps[ni + 1] - tframe;  // 计算到下一帧的时间间隔
        else if (ni > 0)  // 检查是否为第一帧
            T = tframe - vTimestamps[ni - 1];  // 计算与上一帧的时间间隔

        if (ttrack < T) {  // 检查追踪时间是否小于时间间隔
            usleep(min((T - ttrack), 0.5) * 1e6);  // 睡眠剩余时间
        }

        if (mnMainLoopSleep != 0) {  // 检查是否设置了主循环睡眠
            usleep(mnMainLoopSleep * 1e3);  // 主循环睡眠
        }

        // Save Traj
        if (ni == nImages - 1) {  // 检查是否为最后一帧
            while (mpSLAM->GetCloudMerger()->isRunning()) {  // 等待云端合并完成
                usleep(10000);  // 睡眠10毫秒
            }
        }

        ros::spinOnce();  // 处理ROS回调
    }
}
// **********************************************

// **********************************************
// @note ROS包数据运行模式
// RunBag函数：处理ROS包格式的数据集
void Grabber::RunBag(const string &bag_path) {
    ROS_ERROR_STREAM("Begin read bag");  // 输出开始读取包信息
    rosbag::Bag input_bag;
    input_bag.open(bag_path, rosbag::bagmode::Read);  // 以只读模式打开ROS包
    ROS_ERROR_STREAM("End read bag");  // 输出结束读取包信息

    std::vector<std::string> topics;  // 话题列表
    const string image_topic_name = "/davis_left/image_raw"
    //const string image_topic_name = "/camera/color/image_raw/compressed";
    //const string image_topic_name = "/G0/camera/color/image_raw/compressed";
    // const string image_topic_name = "/camera/rgb/image_color";  // TUM、ICL等数据集图像话题
    //const string image_topic_name = "/camera/color/image_color";
    //const string image_topic_name = "/cam0/image_raw";  // Euroc数据集图像话题
    //const string image_topic_name = "/merge/image_raw/compressed";
    //const string image_topic_name = "/camera/color/image_raw";  // nan.bag图像话题
    topics.push_back(image_topic_name);  // 添加图像话题到列表

    int nImages = 0;  // 图像计数
    float imageScale = mpSLAM->GetImageScale();  // 获取图像缩放比例
    vector<double> vTimestamps;  // 时间戳向量
    rosbag::View view(input_bag, rosbag::TopicQuery(topics));  // 创建话题视图
    for (auto m : view) {  // 遍历包中所有消息
        sensor_msgs::Image::ConstPtr pImage = m.instantiate<sensor_msgs::Image>();  // 实例化图像消息
        if (pImage == nullptr) {  // 检查图像消息是否为空
            cerr << endl
                 << "read finish or fail" << endl;  // 输出读取完成或失败信息
            break;  // 跳出循环
        }
        nImages++;  // 图像计数递增
        ROS_ERROR_STREAM("load image index: " << nImages);  // 输出加载图像索引
        vTimestamps.push_back(pImage->header.stamp.toSec());  // 添加时间戳，toSec()将 ros::Time 转换为 double 类型的秒数
    }
    vector<float> vTimesTrack;  // 追踪时间向量
    vTimesTrack.resize(nImages);  // 调整向量大小
    ROS_ERROR_STREAM("bag image size: " << vTimestamps.size());  // 输出包中图像数量

    int ni = 0;  // 当前图像索引
    //rosbag::View view(input_bag, rosbag::TopicQuery(topics));
    for (auto m : view) 
    {  // 再次遍历包中消息进行处理
        double tframe = vTimestamps[ni];  // 获取当前帧时间戳
        //usleep(879000);
        sensor_msgs::Image::ConstPtr pImage = m.instantiate<sensor_msgs::Image>();  // 实例化图像消息
        if (pImage == nullptr) {  // 检查图像消息是否为空
            cerr << endl
                << "read finish or fail" << endl;  // 输出读取完成或失败信息
            break;  // 跳出循环
        }

        cv_bridge::CvImageConstPtr cv_ptr;  // OpenCV图像指针
        try 
        {
            cv_ptr = cv_bridge::toCvCopy(pImage);  // 将ROS图像消息转换为OpenCV图像
        } 
        catch (cv_bridge::Exception &e) 
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());  // 输出转换异常信息
            return;  // 返回
        }

        // Pass the image to the SLAM system
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();  // 记录开始时间
        TrackImage(cv_ptr->image, tframe, imageScale);  // 追踪图像
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();  // 记录结束时间

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();  // 计算追踪耗时

        vTimesTrack[ni] = ttrack;  // 记录追踪时间

        // Wait to load the next frame
        double T = 0;  // 时间间隔
        if (ni < nImages - 1)  // 检查是否为最后一帧
            T = vTimestamps[ni + 1] - tframe;  // 计算到下一帧的时间间隔
        else if (ni > 0)  // 检查是否为第一帧
            T = tframe - vTimestamps[ni - 1];  // 计算与上一帧的时间间隔

        if (ttrack < T) {  // 检查追踪时间是否小于时间间隔
            usleep(min((T - ttrack), 0.5) * 1e6);  // 睡眠剩余时间
        }

        if (mnMainLoopSleep != 0) {  // 检查是否设置了主循环睡眠
            usleep(mnMainLoopSleep * 1e3);  // 主循环睡眠
        }

        // Save Traj
        if (ni == nImages - 1) {  // 检查是否为最后一帧
            while (mpSLAM->GetCloudMerger()->isRunning()) {  // 等待云端合并完成
                usleep(10000);  // 睡眠10毫秒
            }
        }

        // usleep(30 * 1e3);

        ni++;  // 图像索引递增

        ros::spinOnce();  // 处理ROS回调
    }
}
// **********************************************
// @note 图像抓取回调函数
// GrabImage函数：处理ROS图像消息的回调函数
void Grabber::GrabImage(const sensor_msgs::ImageConstPtr &msg) {
    id = msg->header.seq;  // 获取图像序列号
    // ROS_WARN_STREAM("image seq: " << id);

    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;  // OpenCV图像指针
    try {
        cv_ptr = cv_bridge::toCvShare(msg);  // 将ROS图像消息转换为OpenCV图像
    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());  // 输出转换异常信息
        return;  // 返回
    }
    float imageScale = mpSLAM->GetImageScale();  // 获取图像缩放比例

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();  // 记录开始时间
    TrackImage(cv_ptr->image, msg->header.stamp.toSec(), imageScale);  // 追踪图像
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();  // 记录结束时间
}
// **********************************************

// **********************************************
// @note UDF工具函数
//cap-udf
// exists_test函数：检查文件是否存在
inline bool exists_test (const std::string& name ) {
    struct stat buffer;   // 文件状态结构体
    return (stat (name.c_str(), &buffer) == 0);  // 检查文件是否存在
}
//cap-udf
// **********************************************

// **********************************************
// @note Action完成回调函数
// ActionFinishCb函数：处理云端SLAM Action完成回调
void Grabber::ActionFinishCb(const actionlib::SimpleClientGoalState &state, const cloud_edge_slam::CloudSlamResultConstPtr &result) {
    // cout << "begin sleep" << endl;
    // usleep(10000 * 1e3);
    // while (1) {
    //     usleep(1000 * 1e3);
    // }
    // cout << "end sleep" << endl;

    // @note print

    //在此接收关键变量，并等待加载bag数据集，调用ROSMapToORBMap函数 //new_udf_cloud-edge
    if(this->mbNewUdf) {  // 检查是否使用新版UDF
        cloud_edge_slam::CloudMapConstPtr pMap(new cloud_edge_slam::CloudMap(result->map));  // 创建云端地图指针
        std::string pc_name = "/home/birl/Udf-Edge/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_point/test.xyz";  // 点云文件路径
        std::ofstream fout_pc_name(pc_name);  // 打开点云文件，fout_pc_name:文件流对象名（fout = file output）
        //创建一个名为 fout_pc_name 的 ofstream 对象，并尝试打开名为 pc_name 的文件。如果文件不存在，则会创建该文件；
        //如果文件已存在，默认情况下会截断文件（即清空原内容）。之后，我们可以使用 fout_pc_name 来向文件写入数据，就像使用 std::cout 向控制台输出一样
        for (int n = 0; n < pMap->map_points.size(); n++){  // 遍历所有地图点
            fout_pc_name << pMap->map_points[n].point.x << " " << pMap->map_points[n].point.y << " " << pMap->map_points[n].point.z << std::endl;  // 写入点云坐标
        }
        fout_pc_name.close();  // 关闭点云文件

        std::string id_name = "/home/birl/Udf-Edge/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_others/id.txt";  // ID文件路径
        std::ofstream fout_id_name(id_name);  // 打开ID文件

        fout_id_name << pMap->edge_front_map_mnid << std::endl;  // 写入前端地图ID
        fout_id_name << pMap->edge_back_map_mnid << std::endl;  // 写入后端地图ID
        fout_id_name.close();  // 关闭ID文件

        bool trans = true;  // 传输标志

        do{  // 等待传输完成
            bool test1 = exists_test("/home/birl/Udf-Edge/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_bag/test.bag");  // 检查bag文件是否生成

            if (test1 == true){  // 如果文件已生成
                trans = false;  // 设置传输完成
            }
        }
        while(trans);    
        //new_udf_cloud-edge

        //加载bag数据集
        ROS_ERROR_STREAM("Begin read cloud_map_bag");  // 输出开始读取云端地图包信息
        rosbag::Bag test_bag_all;  // ROS包对象
        sleep(3);  // 睡眠3秒
        test_bag_all.open("/home/birl/Udf-Edge/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_bag/test.bag", rosbag::bagmode::Read);  // 以只读模式打开ROS包
        ROS_ERROR_STREAM("End read cloud_map_bag");  // 输出结束读取云端地图包信息
        std::vector<std::string> topics;  // 话题列表
        const string map_topic_name = "/test";  // 地图话题名称
        topics.push_back(map_topic_name);  // 添加地图话题
        rosbag::View view(test_bag_all, rosbag::TopicQuery(topics));  // 创建话题视图
        for (auto m : view) {  // 遍历包中消息
            cloud_edge_slam::CloudSlamResultConstPtr test_bag = m.instantiate<cloud_edge_slam::CloudSlamResult>();  // 实例化云端SLAM结果
            cloud_edge_slam::CloudMapConstPtr mapPtr(new cloud_edge_slam::CloudMap(test_bag->map));  // 创建云端地图指针,创建一个名为 mapPtr 的智能指针,指向新构造的 CloudMap 对象
            /*
            步骤1：在堆上创建 CloudMap 对象
            cloud_edge_slam::CloudMap* raw_ptr = new cloud_edge_slam::CloudMap(test_bag->map);
            步骤2：用智能指针管理该对象
            cloud_edge_slam::CloudMapConstPtr mapPtr(raw_ptr);
            */
            mpSLAM->InsertCloudMap(ROSMapToORBMap(mapPtr));  // 插入转换后的ORB地图
        }
        // rosbag::MessageInstance const m = *view.begin();
        // cloud_edge_slam::CloudSlamResultConstPtr test_bag = m.instantiate<cloud_edge_slam::CloudSlamResult>();
        // cloud_edge_slam::CloudMapConstPtr mapPtr(new cloud_edge_slam::CloudMap(test_bag->map));
        // mpSLAM->InsertCloudMap(ROSMapToORBMap(mapPtr));
        ROS_INFO_STREAM("Cloud Process Finish!");  // 输出云端处理完成信息
        test_bag_all.close();  // 关闭ROS包
        system("rm /home/birl/Udf-Edge/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_bag/test.bag");  // 删除临时bag文件
    }
    //cap-udf-new
    else {  // 使用旧版UDF处理
        cloud_edge_slam::CloudMapConstPtr mapPtr(new cloud_edge_slam::CloudMap(result->map));  // 创建云端地图指针
        mpSLAM->InsertCloudMap(ROSMapToORBMap(mapPtr));  // 插入转换后的ORB地图
        ROS_INFO_STREAM("Cloud Process Finish!");  // 输出云端处理完成信息
    }
    
}
// **********************************************

// **********************************************
// @note 内存回调函数
// MemoryCb函数：处理内存使用情况回调
void Grabber::MemoryCb(const std_msgs::Float32ConstPtr &msg) {cloud_edge_slam::CloudMap

    ROS_ERROR_STREAM("Read Cloud Map End!");  // 输出结束读取云端地图信息
}
// **********************************************

// **********************************************
// @note 坐标转换工具函数
// toRosPose函数：将Sophus位姿转换为ROS几何位姿
inline geometry_msgs::Pose toRosPose(const Sophus::SE3f pose) {
    Eigen::Matrix4f T = pose.matrix();  // 获取位姿矩阵

    geometry_msgs::Pose rosPose;  // 创建ROS位姿
    rosPose.position.x = T(0, 3);  // 设置X坐标
    rosPose.position.y = T(1, 3);  // 设置Y坐标
    rosPose.position.z = T(2, 3);  // 设置Z坐标
    Eigen::Quaternionf q;  // 四元数
    q = T.block<3, 3>(0, 0);  // 从旋转矩阵提取四元数
    rosPose.orientation.x = q.x();  // 设置四元数X分量
    rosPose.orientation.y = q.y();  // 设置四元数Y分量
    rosPose.orientation.z = q.z();  // 设置四元数Z分量
    rosPose.orientation.w = q.w();  // 设置四元数W分量

    return rosPose;  // 返回ROS位姿
}
// **********************************************

// **********************************************
// @note ORB地图发布回调函数
// PubORBMapCb函数：发布ORB地图回调
void Grabber::PubORBMapCb(const std_msgs::Int16ConstPtr &msg) {
    int mapMnId = msg->data;  // 获取地图ID
    ORB_SLAM3::Map *pMap;  // ORB地图指针
    if (mapMnId == -1) {  // 检查是否为当前地图
        pMap = mpSLAM->GetAtlas()->GetCurrentMap();  // 获取当前地图
    } else {
        pMap = mpSLAM->GetAtlas()->GetSpecifyMap(mapMnId);  // 获取指定ID地图
    }
    mpOrbMapPub->publish(ORBMapToROSMap(pMap));  // 发布转换后的ROS地图
}
// **********************************************

// **********************************************
// @note 地图转换函数
// ORBMapToROSMap函数：将ORB地图转换为ROS地图格式
cloud_edge_slam::CloudMap Grabber::ORBMapToROSMap(ORB_SLAM3::Map *pMap) {
    cloud_edge_slam::CloudMap pubTestEdgeMap;  // 创建发布的云端地图

    std::vector<ORB_SLAM3::KeyFrame *> allKeyFrames = pMap->GetAllKeyFrames();  // 获取所有关键帧，
    std::vector<ORB_SLAM3::MapPoint *> allMapPoints = pMap->GetAllMapPoints();  // 获取所有地图点

    pubTestEdgeMap.header.seq = 0;  // 设置消息序列号
    pubTestEdgeMap.edge_front_map_mnid = pMap->GetId();  // 设置前端地图ID
    pubTestEdgeMap.edge_back_map_mnid = pMap->GetId() + 1;  // 设置后端地图ID

    // 提前取得指针与index的对应关系
    std::map<ORB_SLAM3::MapPoint *, int> mapPointMap;  // 地图点索引映射，<ORB_SLAM3::MapPoint *, int>：模板参数，指定键和值的类型
    std::map<ORB_SLAM3::KeyFrame *, int> keyFrameMap;  // 关键帧索引映射
    for (int mapPoint_i = 0; mapPoint_i < allMapPoints.size(); ++mapPoint_i) 
    { 
        mapPointMap[allMapPoints[mapPoint_i]] = mapPoint_i; }  // 建立地图点索引映射

    for (int keyFrame_i = 0; keyFrame_i < allKeyFrames.size(); ++keyFrame_i) 
    { 
        keyFrameMap[allKeyFrames[keyFrame_i]] = keyFrame_i;  // 建立关键帧索引映射
    }  

    for (int mapPoint_i = 0; mapPoint_i < allMapPoints.size(); ++mapPoint_i) {  // 遍历所有地图点
        ORB_SLAM3::MapPoint *mapPoint = allMapPoints[mapPoint_i];  // 获取地图点
        std::map<ORB_SLAM3::KeyFrame *, std::tuple<int, int>> observations = mapPoint->GetObservations();  // 获取观测关系
        /*std::tuple<int, int> - 包含两个整数的元组
                    observations 映射的内容：
        +-------------------------+-----------------------+
        |       键(Key)           |        值(Value)       |
        |     (KeyFrame指针)         |    (tuple<int, int>)   |
        +-------------------------+-----------------------+
        |     0x7ffeeb39b010      |       (15, -1)        |
        |     0x7ffeeb39b020      |       (8, 12)         |
        |     0x7ffeeb39b030      |       (23, -1)        |
        |     0x7ffeeb39b040      |       (5, 6)          |
        +-------------------------+-----------------------+
        */
        cloud_edge_slam::MapPoint rosMapPoint;  // 创建ROS地图点
        rosMapPoint.mnId = mapPoint->mnId;  // 设置地图点ID
        rosMapPoint.ref_keyframe_id = keyFrameMap[mapPoint->GetReferenceKeyFrame()];  // 设置参考关键帧ID
        Eigen::Vector3f point = mapPoint->GetWorldPos();  // 获取世界坐标
        rosMapPoint.point.x = point(0);  // 设置X坐标
        rosMapPoint.point.y = point(1);  // 设置Y坐标
        rosMapPoint.point.z = point(2);  // 设置Z坐标

        for (auto observation : observations) {  // 遍历所有观测
            cloud_edge_slam::Observation rosObservation;  // 创建ROS观测
            if (keyFrameMap.count(observation.first) >= 1)  // 检查关键帧是否存在
            {
                rosObservation.keyframe_id = keyFrameMap[observation.first];  // 设置关键帧ID
                rosObservation.refer_keypoint_index = (short)std::get<0>(observation.second);  // 设置关键点索引
                if ((short)std::get<0>(observation.second) > observation.first->mvKeys.size()) {  // 检查索引是否越界
                    ROS_ERROR_STREAM("observation keypoint index: " << (short)std::get<0>(observation.second) << "  " << observation.first->mvKeys.size());  // 输出错误信息
                }
                assert((short)std::get<0>(observation.second) < observation.first->mvKeys.size());  // 断言索引有效性
                rosMapPoint.observations.push_back(rosObservation);  // 添加观测
            }
        }
        rosMapPoint.num_obs = rosMapPoint.observations.size();  // 设置观测数量
        pubTestEdgeMap.map_points.push_back(rosMapPoint);  // 添加地图点到发布消息
    }

    // Add All KeyFrames
    for (auto KF : allKeyFrames) {  // 遍历所有关键帧
        Sophus::SE3f pose = KF->GetPose();  // 获取位姿
        std::vector<cv::KeyPoint> keyPoints = KF->mvKeys;  // 获取关键点
        std::vector<ORB_SLAM3::MapPoint *> matchMapPoint = KF->GetMapPointMatches();  // 获取匹配的地图点
        std::vector<cv::Mat> descriptors = ORB_SLAM3::Converter::toDescriptorVector(KF->mDescriptors);  // 获取描述子
        assert(matchMapPoint.size() == keyPoints.size() && keyPoints.size() == descriptors.size());  // 断言尺寸一致性

        cloud_edge_slam::KeyFrame rosKeyFrame;  // 创建ROS关键帧
        rosKeyFrame.mTimeStamp = KF->mTimeStamp;  // 设置时间戳
        rosKeyFrame.mnId = KF->mnId;  // 设置关键帧ID
        rosKeyFrame.pose_cw = toRosPose(pose);  // 设置位姿

        for (const auto &descriptor : descriptors) {  // 遍历所有描述子
            cloud_edge_slam::Descriptor rosDescriptor;  // 创建ROS描述子
            //            ROS_ERROR_STREAM("descriptor shape: " << descriptor.size);
            std::vector<double> descriptor_vector;  // 描述子向量
            descriptor.col(0).copyTo(descriptor_vector);  // 复制描述子数据
            std::copy(descriptor_vector.begin(), descriptor_vector.end(), rosDescriptor.descriptor.begin());  // 拷贝到ROS描述子
            rosKeyFrame.descriptors.push_back(rosDescriptor);  // 添加描述子
        }

        for (const auto &keyPoint : keyPoints) {  // 遍历所有关键点
            cloud_edge_slam::KeyPoint rosKeyPoint;  // 创建ROS关键点
            rosKeyPoint.x = keyPoint.pt.x;  // 设置X坐标
            rosKeyPoint.y = keyPoint.pt.y;  // 设置Y坐标
            rosKeyFrame.key_points.push_back(rosKeyPoint);  // 添加关键点
        }

        for (const auto &mapPoint : matchMapPoint) {  // 遍历所有匹配的地图点
            int matchMapPointIndex = -1;  // 匹配地图点索引
            if (mapPoint) {  // 检查地图点是否存在
                matchMapPointIndex = mapPointMap[mapPoint];  // 获取地图点索引
            }
            rosKeyFrame.mvp_map_points_index.push_back(matchMapPointIndex);  // 添加地图点索引
        }
        pubTestEdgeMap.key_frames.push_back(rosKeyFrame);  // 添加关键帧到发布消息
    }

    //    ROS_ERROR_STREAM("TEST pub msg: " << pubTestEdgeMap.key_frames.size());
    //    ROS_ERROR_STREAM("TEST pub msg: " << pubTestEdgeMap.map_points.size());
    return pubTestEdgeMap;  // 返回发布的云端地图
}

// ROSMapToORBMap函数：将ROS地图转换为ORB地图格式
ORB_SLAM3::Map *Grabber::ROSMapToORBMap(cloud_edge_slam::CloudMapConstPtr pMap) {
    ROS_ERROR_STREAM("ROS Map To ORB Map Start!");  // 输出开始转换信息

    // Cloud Image
    std::map<std::string, cv::Mat> dLastCloudProcessImages = mpSLAM->GetCloudImageSampler()->mdLastCloudProcessImages;  // 获取最后处理的云端图像

    // hype parameters
    const bool bIncludeDescriptor = false;  // 是否包含描述子标志
    // const bool bIncludeDescriptor = true;

    // SLAM System Info
    auto pVocabulary = mpSLAM->GetVocabulary();  // 获取词汇表
    auto pExtractor = mpSLAM->GetExtractor();  // 获取特征提取器
    auto pCamera = mpSLAM->GetCamera();  // 获取相机模型
    auto distCoef = mpSLAM->GetDistCoef();  // 获取畸变系数
    auto bf = mpSLAM->Getbf();  // 获取基线焦距
    auto thDepth = mpSLAM->GetThDepth();  // 获取深度阈值
    auto imuCalib = mpSLAM->GetImcCalib();  // 获取IMU标定参数

    static long unsigned int nCloudMapId = 1000;  // 静态云端地图ID
    ORB_SLAM3::Map *cloudMap = new ORB_SLAM3::Map(nCloudMapId++, true);  // 创建新的ORB地图
    // ORB_SLAM3::Map *cloudMap = new ORB_SLAM3::Map(); // TODO 放入Database中，否则会消亡
    // TODO 将msg的信息打包进Map中
    // TODO 需要添加KeyFrameDatabase，其实初步也不用，只用构建完成后放入队列
    cloudMap->edgeFrontMapMnId = pMap->edge_front_map_mnid;  // 设置前端地图ID
    cloudMap->edgeBackMapMnId = pMap->edge_back_map_mnid;  // 设置后端地图ID

    std::vector<ORB_SLAM3::KeyFrame *> vKeyFrames;  // 关键帧向量
    static int gen_start_id = 3565536;  // 静态起始ID
    gen_start_id -= 300000;  // 调整起始ID
    if (gen_start_id < 100000) {  // 检查ID范围
        gen_start_id = 3565536;  // 重置起始ID
    }
    long unsigned int gen_mnId = gen_start_id;  // 生成ID，使用大数避免与在线ID重叠
    for (auto &rosKeyFrame : pMap->key_frames) {  // 遍历所有ROS关键帧
        // mvKeys
        std::vector<cv::KeyPoint> vKeyPoints;  // 关键点向量
        for (auto &rosKeyPoint : rosKeyFrame.key_points) {  // 遍历所有关键点
            cv::KeyPoint keyPoint;  // 创建关键点
            keyPoint.pt.x = rosKeyPoint.x;  // 设置X坐标
            keyPoint.pt.y = rosKeyPoint.y;  // 设置Y坐标
            vKeyPoints.push_back(keyPoint);  // 添加关键点
        }

        // Descriptors
        cv::Mat descriptors;  // 描述子矩阵
        if (bIncludeDescriptor) {  // 检查是否包含描述子
            std::vector<cv::Mat> vDescriptors;  // 描述子向量
            for (auto &rosDescriptor : rosKeyFrame.descriptors) {  // 遍历所有描述子
                std::vector<float> tmpDescriptor(rosDescriptor.descriptor.size());  // 临时描述子向量
                copy(rosDescriptor.descriptor.begin(), rosDescriptor.descriptor.end(), tmpDescriptor.begin());  // 拷贝描述子数据
                cv::Mat descriptor(tmpDescriptor);  // 创建描述子矩阵
                vDescriptors.push_back(descriptor);  // 添加描述子
            }
            descriptors = vDescriptors[0];  // 初始化描述子矩阵
            for (int descriptor_i = 1; descriptor_i < vDescriptors.size(); ++descriptor_i) {  // 合并所有描述子
                cv::hconcat(descriptors, vDescriptors[descriptor_i], descriptors);  // 水平合并描述子
            }
            descriptors = descriptors.t();  // 转置描述子矩阵
        } else {  // 不包含描述子时生成空描述子
            descriptors = cv::Mat::zeros(vKeyPoints.size(), 32, CV_32FC1);  // 创建空描述子矩阵
        }

        // pose
        Sophus::SE3f pose = toSophusPose(rosKeyFrame.pose_cw);  // 转换位姿格式

        // @note construct frame
        ORB_SLAM3::Frame frame(vKeyPoints, rosKeyFrame.mTimeStamp, pExtractor, pVocabulary, pCamera, distCoef, bf, thDepth);  // 创建帧
        frame.mnId = gen_mnId--;  // 设置帧ID
        frame.mDescriptors = descriptors;  // 设置描述子
        frame.mvpMapPoints = std::vector<ORB_SLAM3::MapPoint *>(rosKeyFrame.mvp_map_points_index.size(), static_cast<ORB_SLAM3::MapPoint *>(NULL));  // 初始化地图点向量
        ORB_SLAM3::KeyFrame *keyFrame = new ORB_SLAM3::KeyFrame(frame, cloudMap, nullptr);  // 创建关键帧
        keyFrame->SetPose(pose);  // 设置位姿
        keyFrame->SetCloudFlag();  // 设置云端标志

        // @note add img
        if (dLastCloudProcessImages.count(std::to_string(frame.mTimeStamp))) {  // 检查是否有对应图像
            keyFrame->SetImgGray(dLastCloudProcessImages[std::to_string(frame.mTimeStamp)]);  // 设置灰度图像
        }

        vKeyFrames.push_back(keyFrame);  // 添加关键帧到向量
    }
    std::vector<ORB_SLAM3::MapPoint *> vMapPoints;  // 地图点向量
    gen_mnId = gen_start_id;  // 重置生成ID

    //cap-udf
    if (this->mbOldUdf) {  // 检查是否使用旧版UDF
        std::string pc_name = "/home/birl/Udf-Edge/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_point/test.xyz";  // 点云文件路径
        std::ofstream fout_pc_name(pc_name);  // 打开点云文件

        for (int n = 0; n < pMap->map_points.size(); n++){  // 遍历所有地图点
            fout_pc_name << pMap->map_points[n].point.x << " " << pMap->map_points[n].point.y << " " << pMap->map_points[n].point.z << std::endl;  // 写入点云坐标
        }
        fout_pc_name.close();  // 关闭点云文件

        bool trans = true;  // 传输标志

        do{  // 等待传输完成
            bool test1 = exists_test("/home/birl/Udf-Edge/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_point/test2.xyz");  // 检查点云文件是否生成

            if (test1 == true){  // 如果文件已生成
                trans = false;  // 设置传输完成
            }
        }
        while(trans);

        //system("rm /zclin/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_point/test2.xyz");
    }
    //cap-udf

    for (auto &rosMapPoint : pMap->map_points) {  // 遍历所有ROS地图点
        /*if (rosMapPoint.ref_keyframe_id == -1) {
            continue;
        }*/
        Eigen::Vector3f pos(rosMapPoint.point.x, rosMapPoint.point.y, rosMapPoint.point.z);  // 创建位置向量
        ORB_SLAM3::KeyFrame *refKeyFrame = vKeyFrames[rosMapPoint.ref_keyframe_id];  // 获取参考关键帧
        ORB_SLAM3::MapPoint *mapPoint = new ORB_SLAM3::MapPoint(gen_mnId--, pos, refKeyFrame, cloudMap);  // 创建地图点
        mapPoint->isEdge = false;  // 设置非边缘标志
        vMapPoints.push_back(mapPoint);  // 添加地图点到向量
    }

    // 完成初步KeyFrame和MapPoint对象的初始化，接着建立两者间的联系
    // 参考LocalMapping中CreateNewMapPoints、ProcessNewKeyFrame对两者进行处理
    for (int mapPoint_i = 0; mapPoint_i < vMapPoints.size(); ++mapPoint_i) {  // 遍历所有地图点
        ORB_SLAM3::MapPoint *mapPoint = vMapPoints[mapPoint_i];  // 获取地图点
        for (auto &observation : pMap->map_points[mapPoint_i].observations) {  // 遍历所有观测
            mapPoint->AddObservation(vKeyFrames[observation.keyframe_id], observation.refer_keypoint_index);  // 添加观测关系
            if ((short)observation.refer_keypoint_index > vKeyFrames[observation.keyframe_id]->mvKeys.size()) {  // 检查索引越界
                ROS_ERROR_STREAM("observation keypoint index: " << (short)observation.refer_keypoint_index << "  " << vKeyFrames[observation.keyframe_id]->mvKeys.size());  // 输出错误信息
            }
            assert((short)observation.refer_keypoint_index < vKeyFrames[observation.keyframe_id]->mvKeys.size());  // 断言索引有效性
            //            ROS_ERROR_STREAM("observation keypoint index: " << vKeyFrames[observation.keyframe_id]->mvKeys.size() << "  " << observation.refer_keypoint_index);
        }
        mapPoint->ComputeDistinctiveDescriptors();  // 计算独特描述子
        mapPoint->UpdateNormalAndDepth();  // 更新法向量和深度
    }
    for (int keyFrame_i = 0; keyFrame_i < vKeyFrames.size(); ++keyFrame_i) {  // 遍历所有关键帧
        ORB_SLAM3::KeyFrame *keyFrame = vKeyFrames[keyFrame_i];  // 获取关键帧
        keyFrame->ComputeBoW();  // 计算词袋模型
        int i = 0;  // 索引计数器
        for (auto &matchMapPointIndex : pMap->key_frames[keyFrame_i].mvp_map_points_index) {  // 遍历匹配的地图点索引
            if (matchMapPointIndex != -1)  // 检查索引有效性
                keyFrame->AddMapPoint(vMapPoints[matchMapPointIndex], i);  // 添加地图点
            i++;  // 索引递增
        }
        keyFrame->UpdateCloudConnections();  // 更新云端连接
    }

    // 处理完成后，添加到Map中
    for (auto &keyFrame : vKeyFrames) {  // 遍历所有关键帧
        cloudMap->AddKeyFrame(keyFrame);  // 添加关键帧到地图
    }
    for (auto &mapPoint : vMapPoints) {  // 遍历所有地图点
        cloudMap->AddMapPoint(mapPoint);  // 添加地图点到地图
    }

    ROS_ERROR_STREAM("ROS Map To ORB Map End!");  // 输出结束转换信息

    return cloudMap;  // 返回ORB地图
}
// **********************************************

// **********************************************
// @note 数据加载和保存函数
// LoadImages函数：从文本文件加载图像信息
void Grabber::LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps) {
    ifstream f;  // 输入文件流
    f.open(strFile.c_str());  // 打开文件

    // skip first three lines
    string s0;  // 临时字符串
    getline(f, s0);  // 跳过第一行
    getline(f, s0);  // 跳过第二行
    getline(f, s0);  // 跳过第三行

    while (!f.eof()) {  // 遍历文件直到结束
        string s;  // 行字符串
        getline(f, s);  // 读取一行
        if (!s.empty()) {  // 检查行是否为空
            stringstream ss;  // 字符串流
            ss << s;  // 输入字符串,eg：string s = "1234567890.123 image_001.png"
            double t;  // 时间戳
            string sRGB;  // 图像文件名
            ss >> t;  // 读取时间戳
            vTimestamps.push_back(t);  // 添加时间戳
            ss >> sRGB;  // 读取图像文件名
            vstrImageFilenames.push_back(sRGB);  // 添加图像文件名
        }
    }
}

// SaveORBMapCb函数：保存ORB地图回调
void Grabber::SaveORBMapCb(const std_msgs::Int16ConstPtr &msg) {
    int mapMnId = msg->data;  // 获取地图ID
    ORB_SLAM3::Map *pMap;  // ORB地图指针
    if (mapMnId == -1) {  // 检查是否为当前地图
        pMap = mpSLAM->GetAtlas()->GetCurrentMap();  // 获取当前地图
    } else {
        pMap = mpSLAM->GetAtlas()->GetSpecifyMap(mapMnId);  // 获取指定ID地图
    }
    WriteCloudMapBag(ORBMapToROSMap(pMap), "/home/birl/Udf-Edge/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/src/cloud_edge_slam/TestData/All_ORB_Test_Offline_Map/" + to_string(mapMnId) + ".bag");  // 写入云端地图包
}

// WriteCloudMapBag函数：将云端地图写入ROS包
void Grabber::WriteCloudMapBag(cloud_edge_slam::CloudMap map, const std::string &save_path) {
    rosbag::Bag bag;  // ROS包对象
    bag.open(save_path, rosbag::bagmode::Write);  // 以写入模式打开包
    bag.write("/test_cloud_map", ros::Time::now(), map);  // 写入云端地图
    bag.close();  // 关闭包
}

// WriteCloudImagesBag函数：将云端图像写入ROS包
void Grabber::WriteCloudImagesBag(std::vector<ORB_SLAM3::CloudImage> &vImages, const std::string &save_path) {
    rosbag::Bag bag;  // ROS包对象
    bag.open(save_path, rosbag::bagmode::Write);  // 以写入模式打开包
    for (auto &image : vImages) {  // 遍历所有图像
        std_msgs::Header header;  // 消息头
        header.stamp = ros::Time(image.timestamp);  // 设置时间戳
        header.frame_id = image.type;  // 设置帧ID
        string encode;  // 编码格式
        if (image.img.channels() == 3)  // 检查是否为3通道图像
            encode = "bgr8";  // 设置BGR编码
        else if (image.img.channels() == 1)  // 检查是否为单通道图像
            encode = "mono8";  // 设置单色编码
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, encode, image.img).toImageMsg();  // 转换为ROS图像消息
        bag.write("/test_cloud_images", ros::Time(image.timestamp), msg);  // 写入图像
    }
    bag.close();  // 关闭包
}

// WriteSeqBag函数：将序列写入ROS包
void Grabber::WriteSeqBag(const cloud_edge_slam::Sequence &seq, const std::string &save_path) {
    rosbag::Bag bag;  // ROS包对象
    bag.open(save_path, rosbag::bagmode::Write);  // 以写入模式打开包
    auto curTime = ros::Time::now();  // 获取当前时间
    bag.write("/test_seq", curTime, seq);  // 写入序列
    bag.close();  // 关闭包
}
// **********************************************