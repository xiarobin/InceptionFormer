import os
import open3d as o3d
import torch
import numpy as np
import torch.utils.data as data
import h5py
import math
import transforms3d
import random
from tensorpack import dataflow

import chardet

class PCN_pcd(data.Dataset):
    def __init__(self, path, prefix="train"):
        if prefix=="train":
            self.file_path = os.path.join(path,'train') 
        elif prefix=="val":
            self.file_path = os.path.join(path,'val')  
        elif prefix=="test":
            self.file_path = os.path.join(path,'test') 
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")

        self.prefix = prefix
        self.label_map ={'02691156': '0', '02933112': '1', '02958343': '2',
                         '03001627': '3', '03636649': '4', '04256520': '5',
                         '04379243': '6', '04530566': '7', 'all': '8'}
        
        self.label_map_inverse ={'0': '02691156', '1': '02933112', '2': '02958343',
                         '3': '03001627', '4': '03636649', '5': '04256520',
                         '6': '04379243', '7': '04530566', '8': 'all'}

        self.input_data, self.labels = self.get_data(os.path.join(self.file_path, 'partial'))

        random.shuffle(self.input_data)

        self.len = len(self.input_data)

        self.scale = 0
        self.mirror = 1
        self.rot = 0
        self.sample = 1

    def __len__(self):
        return self.len

    def read_pcd(self, path):
        pcd = o3d.io.read_point_cloud(path)
        points = np.asarray(pcd.points)
        return points

    def get_data(self, path):
        cls = os.listdir(path)
        data = []
        labels = []
        for c in cls:
            objs = os.listdir(os.path.join(path, c))
            for obj in objs:
                f_names = os.listdir(os.path.join(path, c, obj))
                obj_list = []
                for f_name in f_names:
                    data_path = os.path.join(path, c, obj, f_name)
                    obj_list.append(data_path)
                    # points = self.read_pcd(os.path.join(path, c, obj, f_name))
                data.append(obj_list)
                labels.append(self.label_map[c])


        return data, labels

    def randomsample(self, ptcloud ,n_points):
        choice = np.random.permutation(ptcloud.shape[0])
        ptcloud = ptcloud[choice[:n_points]]

        if ptcloud.shape[0] < n_points:
            zeros = np.zeros((n_points - ptcloud.shape[0], 3))
            ptcloud = np.concatenate([ptcloud, zeros])
        return ptcloud

    def upsample(self, ptcloud, n_points):
        curr = ptcloud.shape[0]
        need = n_points - curr

        if need < 0:
            return ptcloud[np.random.permutation(n_points)]

        while curr <= need:
            ptcloud = np.tile(ptcloud, (2, 1))
            # ptcloud = np.concatenate([ptcloud,np.zeros_like(ptcloud)],dim=0)
            need -= curr
            curr *= 2

        choice = np.random.permutation(need)
        ptcloud = np.concatenate((ptcloud, ptcloud[choice]))

        return ptcloud


    def get_transform(self, points):
        result = []
        rnd_value = np.random.uniform(0, 1)

        if self.mirror and self.prefix == 'train':
            trfm_mat = transforms3d.zooms.zfdir2mat(1)
            trfm_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
            trfm_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)
            if rnd_value <= 0.25:
                trfm_mat = np.dot(trfm_mat_x, trfm_mat)
                trfm_mat = np.dot(trfm_mat_z, trfm_mat)
            elif rnd_value > 0.25 and rnd_value <= 0.5:    # lgtm [py/redundant-comparison]
                trfm_mat = np.dot(trfm_mat_x, trfm_mat)
            elif rnd_value > 0.5 and rnd_value <= 0.75:
                trfm_mat = np.dot(trfm_mat_z, trfm_mat)
        for ptcloud in points:
            ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)

            if self.scale:
                ptcloud = ptcloud * self.scale
            result.append(ptcloud)

        return result[0],result[1]

    def __getitem__(self, index):

        partial_path = self.input_data[index]
        n_sample = len(partial_path)
        idx = random.randint(0, n_sample-1)
        partial_path = partial_path[idx]

        partial = self.read_pcd(partial_path)

        # if self.prefix == 'train' and self.sample:
        partial = self.upsample(partial, 2048)

        gt_path = partial_path.replace('/'+partial_path.split('/')[-1],'.pcd')
        gt_path = gt_path.replace('partial','complete')


        if self.prefix == 'train':
            complete = self.read_pcd(gt_path)
            partial, complete = self.get_transform([partial, complete])
        else:
            complete = self.read_pcd(gt_path)

        complete = torch.from_numpy(complete)
        partial = torch.from_numpy(partial)
        label = partial_path.split('/')[-3]
        label = self.label_map[label]
        obj = partial_path.split('/')[-2]
        
        if self.prefix == 'test':
            return label, partial, complete, obj
        else:
            return label, partial, complete


class C3D_h5(data.Dataset):
    def __init__(self, path, prefix="train"):
        if prefix=="train":
            self.file_path = os.path.join(path,'train') 
        elif prefix=="val":
            self.file_path = os.path.join(path,'val')  
        elif prefix=="test":
            self.file_path = os.path.join(path,'test') 
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")

        self.prefix = prefix
        self.label_map ={'02691156': '0', '02933112': '1', '02958343': '2',
                         '03001627': '3', '03636649': '4', '04256520': '5',
                         '04379243': '6', '04530566': '7', 'all': '8'}

        if prefix is not "test":
            self.input_data, self.labels = self.get_data(os.path.join(self.file_path, 'partial'))
            self.gt_data, _ = self.get_data(os.path.join(self.file_path, 'gt'))
            print(len(self.gt_data), len(self.labels))
        else:
            self.input_data, self.labels = self.get_data(os.path.join(self.file_path, 'partial'))

        print(len(self.input_data))

        self.len = len(self.input_data)

        self.scale = 1
        self.mirror = 1
        self.rot = 0
        self.sample = 1

    def __len__(self):
        return self.len

    def get_data(self, path):
        cls = os.listdir(path)
        data = []
        labels = []
        for c in cls:
            objs = os.listdir(os.path.join(path, c))
            for obj in objs:
                data.append(os.path.join(path,c,obj))
                if self.prefix == "test":
                    labels.append(obj)
                else:
                    labels.append(self.label_map[c])

        return data, labels


    def get_transform(self, points):
        result = []
        rnd_value = np.random.uniform(0, 1)
        angle = random.uniform(0,2*math.pi)
        scale = np.random.uniform(1/1.6, 1)

        trfm_mat = transforms3d.zooms.zfdir2mat(1)
        if self.mirror and self.prefix == 'train':

            trfm_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
            trfm_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)
            if rnd_value <= 0.25:
                trfm_mat = np.dot(trfm_mat_x, trfm_mat)
                trfm_mat = np.dot(trfm_mat_z, trfm_mat)
            elif rnd_value > 0.25 and rnd_value <= 0.5:    # lgtm [py/redundant-comparison]
                trfm_mat = np.dot(trfm_mat_x, trfm_mat)
            elif rnd_value > 0.5 and rnd_value <= 0.75:
                trfm_mat = np.dot(trfm_mat_z, trfm_mat)
        if self.rot:
                trfm_mat = np.dot(transforms3d.axangles.axangle2mat([0,1,0],angle), trfm_mat)
        for ptcloud in points:
            ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)

            if self.scale:
                ptcloud = ptcloud * scale
            result.append(ptcloud)

        return result[0],result[1]

    def __getitem__(self, index):
        partial_path = self.input_data[index]
        with h5py.File(partial_path, 'r') as f:
            partial = np.array(f['data'])

        if self.prefix == 'train' and self.sample:
            choice = np.random.permutation((partial.shape[0]))
            partial = partial[choice[:2048]]
            if partial.shape[0] < 2048:
                zeros = np.zeros((2048-partial.shape[0],3))
                partial = np.concatenate([partial,zeros])

        if self.prefix not in ["test"]:
            complete_path = partial_path.replace('partial','gt')
            with h5py.File(complete_path, 'r') as f:
                complete = np.array(f['data'])

            partial, complete = self.get_transform([partial, complete])

            complete = torch.from_numpy(complete)
            label = (self.labels[index])
            partial = torch.from_numpy(partial)

            return label, partial, complete
        else:
            partial = torch.from_numpy(partial)
            label = (self.labels[index])
            return label, partial, partial


class TreeCompletion3D(data.Dataset):
    """
    TreeCompletion3D 是一个 PyTorch 数据集类，用于处理3D点云数据，执行点云数据增强和数据加载。
    """

    def __init__(self, path, prefix="train"):
        """
        初始化数据集类，指定数据路径以及数据集的类型（train/val/test）。

        :param path: 数据存放路径
        :param prefix: 数据集类型，取值为 "train"、"val" 或 "test"
        """
        # 根据输入的prefix（数据集类型）设置文件路径
        if prefix == "Train":
            self.file_path = os.path.join(path, 'Train')  # 训练集路径
        elif prefix == "Validation":
            self.file_path = os.path.join(path, 'Validation')  # 验证集路径
        elif prefix == "Test":
            self.file_path = os.path.join(path, 'Validation')  # 测试集路径
        else:
            raise ValueError("ValueError prefix should be [Train/Validation/Test]")  # 若prefix不是train/val/test中的任意一个，抛出异常

        # 设置数据集类型和标签映射
        self.prefix = prefix
        self.label_map = {
            'AcePse': '0', 'CarBet': '1', 'FagSyl': '2',
            'JugReg': '3', 'LacDec': '4', 'PicAbi': '5',
            'PinSyl': '6', 'PruAvi': '7', 'PseMen': '8',
            'QuePet': '9', 'QueRub': '10', 'all': '11'  # 每种类别对应一个标签（数字）
        }

        # 如果数据集不是测试集，则加载partial点云和ground truth（gt）数据
        if prefix != "Test":
            self.input_data, self.labels = self.get_data(os.path.join(self.file_path, 'partial'))  # 加载partial点云
            self.gt_data, _ = self.get_data(os.path.join(self.file_path, 'gt'))  # 加载ground truth点云（训练或验证集）
            print(prefix, "gt DataSet Size:", len(self.gt_data), "partial DataSet Size:",len(self.labels))  # 打印ground truth数据和标签的数量
        else:
            # 如果是测试集，仅加载partial点云数据
            self.input_data, self.labels = self.get_data(os.path.join(self.file_path, 'partial'))

        # 打印input_data的长度
        print(len(self.input_data))

        # 设置数据集的大小
        self.len = len(self.input_data)

        # 数据增强操作的参数（是否进行缩放、镜像、旋转）
        self.scale = 1.05  # 是否进行缩放
        self.mirror = 0  # 是否进行镜像操作
        self.rot = 0  # 是否进行旋转
        self.sample = 1  # 是否进行点云采样

    def __len__(self):
        """
        返回数据集的大小（即点云数据的数量）
        """
        return self.len

    def get_data(self, path):
        """
        从指定路径加载数据，并为每个类别分配标签，只获取 .h5 文件

        :param path: 数据存放的路径
        :return: 数据和对应标签的列表
        """
        cls = os.listdir(path)  # 获取目录中的所有类别文件夹
        data = []  # 存储所有点云数据文件的路径
        labels = []  # 存储每个点云数据对应的标签

        # 遍历每个类别
        for c in cls:
            objs = os.listdir(os.path.join(path, c))  # 获取当前类别下的所有文件
            for obj in objs:
                # 只处理 .h5 文件
                if obj.endswith('.h5'):
                    data.append(os.path.join(path, c, obj))  # 将 .h5 文件路径添加到数据列表中
                    if self.prefix == "test":
                        labels.append(obj)  # 如果是测试集，使用文件名作为标签
                    else:
                        labels.append(self.label_map[c])  # 如果是训练集或验证集，使用类别的标签映射

        return data, labels

    def get_transform(self, points):
        """
        对点云数据执行随机数据增强操作（包括旋转、镜像、缩放等）

        :param points: 输入的点云数据
        :return: 变换后的点云数据
        """
        result = []  # 存储变换后的点云数据
        rnd_value = np.random.uniform(0, 1)  # 随机值，用于控制变换操作
        angle = random.uniform(0, 2 * math.pi)  # 随机旋转角度
        scale = np.random.uniform(1 / 1.6, 1)  # 随机缩放因子

        trfm_mat = transforms3d.zooms.zfdir2mat(1)  # 初始化变换矩阵

        # 如果进行镜像操作
        if self.mirror and self.prefix == 'Train':
            trfm_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)  # X轴镜像
            trfm_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)  # Z轴镜像
            if rnd_value <= 0.25:
                trfm_mat = np.dot(trfm_mat_x, trfm_mat)  # 执行X轴镜像变换
                trfm_mat = np.dot(trfm_mat_z, trfm_mat)  # 执行Z轴镜像变换
            elif rnd_value > 0.25 and rnd_value <= 0.5:
                trfm_mat = np.dot(trfm_mat_x, trfm_mat)  # 执行X轴镜像变换
            elif rnd_value > 0.5 and rnd_value <= 0.75:
                trfm_mat = np.dot(trfm_mat_z, trfm_mat)  # 执行Z轴镜像变换

        # 如果需要进行旋转变换
        if self.rot:
            trfm_mat = np.dot(transforms3d.axangles.axangle2mat([0, 1, 0], angle), trfm_mat)  # 绕Y轴旋转

        # 对每个点云进行变换
        for ptcloud in points:
            ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)  # 执行矩阵变换（旋转、镜像等）

            if self.scale:
                ptcloud = ptcloud * scale  # 缩放点云

            result.append(ptcloud)  # 将变换后的点云添加到结果列表

        return result[0], result[1]  # 返回变换后的第一个和第二个点云

    def read_point_cloud(self, path):
        """
        从文件中读取点云数据。

        :param path: 点云文件路径。
        :return: 点云数据，格式为 N×3 的 numpy 数组。
        """
        point_cloud = []
        print(path)
        # 首先读取文件的一部分来检测编码
        with open(path, 'rb') as f:
            raw_data = f.read(20)  # 读取部分数据进行编码检测
            result = chardet.detect(raw_data)
            encoding = result['encoding']

        with open(path, 'r', encoding=encoding) as file:
            for line in file:
                values = line.strip().split(" ")  # 根据分隔符分割每行数据
                if len(values) == 3:  # 确保每行有三个数
                    point_cloud.append([float(v) for v in values])
        return np.array(point_cloud, np.float32)  # 转换为 float32 类型的 numpy 数组

    def __getitem__(self, index):
        """
        获取指定索引位置的点云数据，并返回相应的标签、部分点云和完整点云数据。

        :param index: 数据索引
        :return: 标签、部分点云数据和完整点云数据
        """
        partial_path = self.input_data[index]  # 获取部分点云数据的路径
        with h5py.File(partial_path, 'r') as f:
            partial = np.array(f['data'])

        # 如果是训练集并且需要采样点云，执行随机采样
        # if self.prefix not in ["Test"] and self.sample:
        if self.prefix not in [""] and self.sample:
            choice = np.random.permutation((partial.shape[0]))  # 随机打乱点云数据
            partial = partial[choice[:4096]]  # 采样前2048个点
            # 如果采样的点数小于2048，填充零点云
            if partial.shape[0] < 4096:
                zeros = np.zeros((4096 - partial.shape[0], 3))
                partial = np.concatenate([partial, zeros])  # 填充零点云

        # 如果不是测试集，加载完整点云数据
        if self.prefix not in ["Test"]:
            complete_path = partial_path.replace('partial', 'gt')  # 替换路径中的'partial'为'gt'，得到完整点云路径
            with h5py.File(complete_path, 'r') as f:
                complete = np.array(f['data'])

            # 对部分和完整点云进行数据增强变换
            # partial, complete = self.get_transform([partial, complete])

            # 将点云数据转换为Torch张量
            complete = torch.from_numpy(complete)
            label = self.labels[index]  # 获取该点云的标签
            partial = torch.from_numpy(partial)

            return label, partial, complete  # 返回标签、部分点云和完整点云
        else:
            # 如果是测试集，返回部分点云和标签
            partial = torch.from_numpy(partial)
            label = self.labels[index]
            return label, partial, partial, os.path.splitext(os.path.basename(partial_path))[0] + 'output.h5'

from tqdm import tqdm

if __name__ == '__main__':
    # filepath = '/media/user/Elements SE/Deep_Learning/Completion/PointAttN-main/data/dataset2019/shapenet'
    # dataset = C3D_h5(prefix='val', path=filepath)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=32,
    #                                          shuffle=True, num_workers=2)
    # for i, data in enumerate(tqdm(dataloader, desc="Training Progress"), 0):
    #     continue
    filepath = '/media/user/CUG_DataStorage/TreeCompletion3D/Sample_Data/GT_Remain/txt'
    dataset = TreeCompletion3D(prefix='Test', path=filepath)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                             shuffle=True, num_workers=2)
    for i, data in enumerate(tqdm(dataloader, desc="Training Progress"), 0):
        continue


