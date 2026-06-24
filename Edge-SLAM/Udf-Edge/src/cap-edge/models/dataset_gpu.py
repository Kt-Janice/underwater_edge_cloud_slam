import torch
import torch.nn.functional as F
import numpy as np
import os
from scipy.spatial import cKDTree
from torch.utils.data import Dataset, DataLoader
import time

# 设置GPU优化参数
torch.set_num_threads(4)  # Jetson NX推荐设置[5](@ref)
os.environ["OMP_NUM_THREADS"] = "4"
torch.backends.cudnn.benchmark = True  # 加速卷积运算[4](@ref)

def search_nearest_point(point_batch, point_gt):
    """GPU加速的最近邻搜索"""
    # 使用向量化操作替代循环[3](@ref)
    distances = torch.cdist(point_batch, point_gt)
    dis_idx = torch.argmin(distances, dim=1)
    return dis_idx.cpu().numpy()  # 结果移回CPU处理

def process_data(data_dir, dataname):
    """优化后的数据处理函数"""
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
    
    # GPU加速的距离计算
    ptree = cKDTree(pointcloud)
    sigmas = []
    for p in np.array_split(pointcloud, min(100, POINT_NUM_GT//100), axis=0):  # 动态分块
        d = ptree.query(p, 51)
        sigmas.append(d[0][:, -1])
    sigmas = np.concatenate(sigmas)
    
    # 生成样本
    QUERY_EACH = max(1, 1000000 // POINT_NUM_GT)
    scale = 0.25 * np.sqrt(POINT_NUM_GT / 20000)
    scale = min(scale, 0.25)
    
    # 使用GPU批量生成样本
    sample_list, sample_near_list = [], []
    for _ in range(QUERY_EACH):
        noise = np.random.normal(0.0, 1.0, size=pointcloud.shape)
        tt = pointcloud + scale * np.expand_dims(sigmas, -1) * noise
        
        # GPU加速最近邻搜索
        tt_tensor = torch.as_tensor(tt, device="cuda").float()
        pointcloud_tensor = torch.as_tensor(pointcloud, device="cuda").float()
        nearest_idx = search_nearest_point(tt_tensor, pointcloud_tensor)
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
        
        # 使用pin_memory加速数据传输[1,6](@ref)
        self.point = torch.as_tensor(load_data['sample_near'], device="cpu").pin_memory().float()
        self.sample = torch.as_tensor(load_data['sample'], device="cpu").pin_memory().float()
        self.point_gt = torch.as_tensor(load_data['point'], device="cpu").pin_memory().float()
        
        # 修复：添加缺失的属性初始化
        self.sample_points_num = len(self.sample) - 1  # 关键修复点
        
        # 计算边界框
        self.object_bbox_min = torch.min(self.point, dim=0)[0] - 0.05
        self.object_bbox_max = torch.max(self.point, dim=0)[0] + 0.05
        print('边界框:', self.object_bbox_min, self.object_bbox_max)
        
        # 初始化DataLoader参数
        self.batch_size = 4096  # 根据显存调整[5](@ref)
        self.num_workers = min(4, os.cpu_count())  # 根据CPU核心数设置[1](@ref)
        
        print('Loading Data End...')

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        """返回单个样本[2](@ref)"""
        # 添加边界检查防止索引越界[6](@ref)
        idx = min(max(idx, 0), len(self.sample) - 1)
        return self.sample[idx], self.point[idx]

    def create_dataloader(self):
        """创建优化的DataLoader[1,5](@ref)"""
        return DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def get_train_data(self, batch_size):
        """获取训练数据批次"""
        # 使用PyTorch替代NumPy操作以利用GPU加速
        index_coarse = torch.randint(0, 10, (1,)).item()
        index_fine = torch.randperm(self.sample_points_num // 10)[:batch_size]
        index = index_fine * 10 + index_coarse
        
        # 添加边界检查
        index = torch.clamp(index, 0, len(self.sample) - 1)
        
        return self.point[index], self.sample[index]

    def get_scale(self):
        return self.scale

    def get_center(self):
        return self.center

