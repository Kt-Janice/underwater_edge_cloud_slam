import torch
import numpy as np
import os
import time
from scipy.spatial import cKDTree
from torch.utils.data import Dataset, DataLoader

# 设置线程优化
torch.set_num_threads(8)
os.environ["OMP_NUM_THREADS"] = "8"

def search_nearest_point_cpu(point_batch, point_gt):
    """使用 cKDTree 替代 GPU 最近邻搜索"""
    # 构建 cKDTree 索引
    tree = cKDTree(point_gt.cpu().numpy())  # 确保数据在 CPU
    # 批量查询最近邻
    _, indices = tree.query(point_batch.cpu().numpy(), k=1)
    return indices  # 返回最近邻索引（CPU 数组）

def process_data(data_dir, dataname):
    """完全基于 CPU 的数据处理函数"""
    start_time = time.time()
    
    # 加载数据
    pointcloud = np.loadtxt(os.path.join(data_dir, dataname) + '.xyz')
    os.remove(os.path.join(data_dir, dataname) + '.xyz')
    
    # 计算缩放参数
    bbox_min = np.min(pointcloud, axis=0)
    bbox_max = np.max(pointcloud, axis=0)
    shape_scale = np.max(bbox_max - bbox_min)
    shape_center = (bbox_max + bbox_min) / 2
    
    # 中心化和归一化
    pointcloud = (pointcloud - shape_center) / shape_scale

    # 点云采样
    POINT_NUM_GT = (pointcloud.shape[0] // 60) * 60
    point_idx = np.random.choice(pointcloud.shape[0], POINT_NUM_GT, replace=False)
    pointcloud = pointcloud[point_idx, :]
    
    # 使用 cKDTree 计算 sigmas（CPU 加速）
    ptree = cKDTree(pointcloud)
    sigmas = []
    # 分块处理避免内存溢出
    for p in np.array_split(pointcloud, min(100, POINT_NUM_GT//100), axis=0):
        d = ptree.query(p, 51)
        sigmas.append(d[0][:, -1])
    sigmas = np.concatenate(sigmas)
    
    # 生成样本
    QUERY_EACH = max(1, 1000000 // POINT_NUM_GT)
    scale = 0.25 * np.sqrt(POINT_NUM_GT / 20000)
    scale = min(scale, 0.25)
    
    sample_list, sample_near_list = [], []
    for _ in range(QUERY_EACH):
        noise = np.random.normal(0.0, 1.0, size=pointcloud.shape)
        tt = pointcloud + scale * np.expand_dims(sigmas, -1) * noise
        
        # CPU 最近邻搜索（避免 GPU 依赖）
        nearest_idx = search_nearest_point_cpu(
            torch.as_tensor(tt), 
            torch.as_tensor(pointcloud)
        )
        nearest_points = pointcloud[nearest_idx]
        
        sample_list.append(tt)
        sample_near_list.append(nearest_points)
    
    # 保存处理结果
    os.makedirs('data_pth/', exist_ok=True)
    np.savez(os.path.join('data_pth/', dataname) + '.npz', 
             sample=np.concatenate(sample_list),
             point=pointcloud,
             sample_near=np.concatenate(sample_near_list))
    
    print(f"数据处理完成, 耗时: {time.time()-start_time:.2f}s")
    return shape_scale, shape_center

class Dataset:  # 建议使用更具描述性的类名
    """优化后的数据集类"""
    def __init__(self, conf, dataname):
        super(Dataset,self).__init__()
        print('Loading Data Begin...')
        self.conf = conf
        
        # 处理数据
        self.scale, self.center = process_data(self.conf, dataname)
        self.data_name = dataname + '.npz'
        load_data = np.load(os.path.join('data_pth/', self.data_name))
        
        # 数据存储在 CPU（启用 pin_memory 加速 DataLoader）
        self.point = torch.as_tensor(load_data['sample_near'], device="cpu").pin_memory().float()
        self.sample = torch.as_tensor(load_data['sample'], device="cpu").pin_memory().float()
        self.point_gt = torch.as_tensor(load_data['point'], device="cpu").pin_memory().float()
        
        self.sample_points_num = len(self.sample)
        
        # 计算边界框
        self.object_bbox_min = torch.min(self.point, dim=0)[0] - 0.05
        self.object_bbox_max = torch.max(self.point, dim=0)[0] + 0.05
        print('边界框:', self.object_bbox_min, self.object_bbox_max)
        
        # 初始化 DataLoader 参数
        self.batch_size = 4096
        self.num_workers = min(4, os.cpu_count())
        
        print('Loading Data End...')

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        idx = min(max(idx, 0), len(self.sample) - 1)
        return self.sample[idx], self.point[idx]

    def create_dataloader(self):
        """启用异步数据加载"""
        return DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,  # 加速 CPU→GPU 传输
            prefetch_factor=2,  # 预取批次减少等待
            persistent_workers=True
        )
 
    
    def get_scale(self):
        return self.scale

    def get_center(self):
        return self.center
        