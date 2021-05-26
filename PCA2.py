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
    return np.array(train)


def PCA(train, n_dim):
    N = train.shape[0]
    train = train - np.mean(train, axis=0, keepdims=True)
    cov = np.matmul(train, train.T)
    print(cov.shape)
    eigvalues, eigvector = np.linalg.eig(cov)
    indexs_ = np.argsort(-eigvalues)[:n_dim]
    Npicked_eig_values = eigvalues[indexs_]
    Npicked_eig_vector = eigvector[:, indexs_]
    picked_eig_vector = np.dot(train.T, Npicked_eig_vector)
    picked_eig_vector = picked_eig_vector / (N * Npicked_eig_values.reshape(-1, n_dim)) ** 0.5
    data_ndim = np.matmul(train, picked_eig_vector)
    return data_ndim


if __name__ == '__main__':
    path = os.listdir("dataSet/AFLW")
    train = Load_data(path)
    print(train.shape)
    data_ndim = PCA(train, 30)
    np.savetxt('result.txt', data_ndim, fmt="%0.8f")
