import PCA
import os
import numpy as np

data_path = os.listdir("dataSet/AFLW")
label_path = 'dataSet/label.txt'


def load_data(data_path, label_path):
    """
    导入数据与标签
    prrameter *_path: 数据的路径
    return train: np.array数据集一行一个数据
    return label:np.array数据集[1,len]
    """
    train = PCA.Load_data(data_path)
    train = train.T
    labeltem = open(label_path, encoding="UTF-8")
    tem = []
    label = []
    for i in labeltem:
        tem.append(i.split())

    for i in tem:
        label.append(float(i[11]))
    return train, np.array(label).reshape(1, np.array(label).shape[0])
    # print(labeltem)


if __name__ == "__main__":
    train, label = load_data(data_path, label_path)
    print(train.shape)
    print(label.shape)