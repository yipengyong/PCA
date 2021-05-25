import numpy as np
import os
from PIL import Image


def Load_data(path):
    """
    输入os生成的路径path，输出numpy.array的train
    parameter data: os生成的路径
    return: numpy.array类型的数据集
    """
    tem = []
    train = []
    for i in path:
        tem.append(i)
    for i in tem:
        train.append(np.array(Image.open("dataSet/AFLW/" + i).convert("L"), dtype=float).reshape(150 * 150))
    return np.array(train).T


def zero_center(train):
    """
    函数进行去中心化
    parameter train: np.array格式的数据集
    return: np.array格式的数据集
    """
    mean = np.mean(train, 1).reshape(len(np.mean(train, 1)), 1)
    tem = np.ones((1, train.shape[1]))
    # print(mean.shape)
    # print(tem.shape)
    mean = np.matmul(mean, tem)
    # print(mean)
    train -= mean
    return train


def cov(train):
    """
    计算协方差矩阵
    parameter train: np.array
    return: 协方差矩阵
    """
    n = train.shape[0]
    cov = np.matmul(train, train.T) / n
    return cov


def eig(cov, train, n_dim):
    """
    进行矩阵的特征值分解并排序
    parameter train: np.array矩阵
    """
    eigvals, eigvects = np.linalg.eig(cov)
    # print(eigvals.shape)
    # print(eigvects.shape)
    # for i in range(len(eigvals)):
    #     dir[str(eigvals[i])] =
    indexs_ = np.argsort(-eigvals)[:n_dim]
    picked_eig_values = eigvals[indexs_]
    picked_eig_vector = eigvects[:, indexs_]
    data_ndim = np.dot(train.T, picked_eig_vector)
    return data_ndim


if __name__ == "__main__":
    path = os.listdir("dataSet/AFLW")
    train = Load_data(path)
    train = zero_center(train)
    cov = cov(train)
    data_ndim = eig(cov, train, 30)
