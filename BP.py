import numpy as np
import matplotlib.pyplot as plt
import Load_data
import os

data_path = os.listdir("dataSet/AFLW")
label_path = 'dataSet/label.txt'


def load_data(path):
    data = np.loadtxt(path, dtype=np.float32)
    return data


sourcetrain, sourceY = Load_data.load_data(data_path, label_path)
X = load_data("train.txt")

train_index = list(np.loadtxt('train_index.txt', dtype=int))

Y = sourceY[:, train_index]

X = X.T
Y = Y.reshape(1, Y.shape[1])


def ini(n_x, n_h1, n_h2, n_y):
    """

    :param n_x: 输入层维度
    :param n_h1: 隐藏一层维度
    :param n_h2: 隐藏二层维度
    :param n_y: 输出层维度
    :return: 存储参数的字典
    """
    np.random.seed(0)

    # 初始化参数

    w1 = np.random.randn(n_h1, n_x)
    b1 = np.zeros((n_h1, 1))
    w2 = np.random.randn(n_h2, n_h1)
    b2 = np.zeros((n_h2, 1))
    w3 = np.random.randn(n_y, n_h2)
    b3 = np.zeros((n_y, 1))
    parameters = {
        'w1': w1,
        'b1': b1,
        'w2': w2,
        'b2': b2,
        'w3': w3,
        'b3': b3
    }
    return parameters


def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    return a


def forward(X, parameters):
    """
    前向传播
    :param X: 神经网络输入
    :param parameters: 神经网络参数
    :return:
    A3 : 神经网络输出
    cache： 中间变量
    """
    # 参数
    w1 = parameters['w1']
    w2 = parameters['w2']
    w3 = parameters['w3']
    b1 = parameters['b1']
    b2 = parameters['b2']
    b3 = parameters['b3']

    # 输入层 -> 隐藏一层
    z1 = np.dot(w1, X) + b1
    A1 = sigmoid(z1)

    # 隐藏一层 -> 隐藏二层
    z2 = np.dot(w2, A1) + b2
    A2 = sigmoid(z2)

    # 隐藏二层 -> 输出层
    z3 = np.dot(w3, A2) + b3
    # print(z3)
    A3 = sigmoid(z3) + 1
    # for i in range(len(A3[0])):
    #     if A3[0][i] >= 1.5:
    #         A3[0][i] = 2
    #     else:
    #         A3[0][i] = 1
    cache = {
        'z1': z1,
        'z2': z2,
        'z3': z3,
        'A1': A1,
        'A2': A2,
        'A3': A3,
    }

    return A3, cache


def loss(A3, Y):
    """
    交叉熵损失
    :param A3: 神经网络输出
    :param Y: 样本真实标签
    :return:
    cost :交叉熵损失函数
    """
    # 样本个数
    m = Y.shape[1]

    # print(A3)
    # cross_entropy = -(Y * np.log(A3) + (1 - Y) * np.log(1 - A3))
    # print(Y.shape)
    # print(A3.shape)

    loss = 0.5 * np.sum((Y - A3) * (Y - A3))
    cost = 1.0 / m * np.sum(loss)
    return cost


def back(X, Y, parameters, cache):
    """

    :param X: 神经网络输入
    :param Y: 样本真实标签
    :param parameters: 网络参数
    :param cache: 中间变量
    :return: 梯度
    """
    # 样本个数
    m = X.shape[1]

    # 神经网络参数
    w1 = parameters['w1']
    w2 = parameters['w2']
    w3 = parameters['w3']
    b1 = parameters['b1']
    b2 = parameters['b2']
    b3 = parameters['b3']

    # 中间变量
    z1 = cache['z1']
    z2 = cache['z2']
    z3 = cache['z3']
    A1 = cache['A1']
    A2 = cache['A2']
    A3 = cache['A3']

    # 计算梯度
    dz3 = (A3 - Y)
    dw3 = 1.0 / m * np.dot(dz3, A2.T)
    db3 = 1.0 / m * np.sum(dz3, axis=1, keepdims=True)
    dz2 = np.dot(w3.T, dz3) * (A2 * (1 - A2))
    dw2 = 1.0 / m * np.dot(dz2, A1.T)
    db2 = 1.0 / m * np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.dot(w2.T, dz2) * (A1 * (1 - A1))
    dw1 = 1.0 / m * np.dot(dz1, X.T)
    db1 = 1.0 / m * np.sum(dz1, axis=1, keepdims=True)

    grads = {
        'dw1': dw1,
        'dw2': dw2,
        'dw3': dw3,
        'db1': db1,
        'db2': db2,
        'db3': db3,

    }
    # print(grads)
    return grads


def update(parameters, grads, learning_rate=0.1):
    """

    :param patameters: 网络参数
    :param grads: 梯度
    :param learning_rate:学习率
    :return: 更新后的参数
    """
    # 参数
    w1 = parameters['w1']
    w2 = parameters['w2']
    w3 = parameters['w3']
    b1 = parameters['b1']
    b2 = parameters['b2']
    b3 = parameters['b3']

    # 梯度
    dw1 = grads['dw1']
    dw2 = grads['dw2']
    dw3 = grads['dw3']
    db1 = grads['db1']
    db2 = grads['db2']
    db3 = grads['db3']

    # 梯度下降
    w1 = w1 - learning_rate * dw1
    w2 = w2 - learning_rate * dw2
    w3 = w3 - learning_rate * dw3
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2
    b3 = b3 - learning_rate * db3
    # print(w1)
    parameters = {
        'w1': w1,
        'b1': b1,
        'w2': w2,
        'b2': b2,
        'w3': w3,
        'b3': b3
    }
    return parameters


# 构建网络

def nn_model(X, Y, n_h1=3, n_h2=3, num_iterations=200, learning_rate=0.1):
    """

    :param X: 输入样本
    :param Y: 标签
    :param n_h1: 隐一层神经元个数
    :param n_h2: 二层神经元个数
    :param num_iterations: 训练次数
    :param learning_rate: 学习率
    :return: 训练完成后的网络参数
    """
    # 定义输入输出
    n_x = X.shape[0]
    n_y = 1

    # 初始化参数
    parameters = ini(n_x, n_h1, n_h2, n_y)

    # 迭代训练
    for i in range(num_iterations):
        # 正向传播
        A3, cache = forward(X, parameters)
        # print(A3)
        # 计算损失
        cost = loss(A3, Y)
        # 反向传播
        grads = back(X, Y, parameters, cache)
        # 参数更新
        parameters = update(parameters, grads, learning_rate)

        if (i + 1) % 20 == 0:
            print("循环次数： %d, cost = %f" % (i + 1, cost))

    return parameters


# 训练
parameters = nn_model(X, Y, n_h1=15, n_h2=15, num_iterations=100000, learning_rate=0.2)


# 预测
def predict(X, parameters):
    """
    预测
    :param X: 网络输入
    :param parameters: 网络参数
    :return: 预测标签
    """
    Y_pred, cache = forward(X, parameters)
    return Y_pred


X_test = load_data("test.txt")
X_test = X_test.T
test_index = list(np.loadtxt('test_index.txt', dtype=int))
Y_test = sourceY[:, test_index]
Y_test = Y_test.reshape(1, Y_test.shape[1])
# X_test, Y_test, X1_test, X2_test = load_data('数据集/train.txt')
# X_test = X_test.T
# X1_test = X1_test.T
# X2_test = X2_test.T
Y_pred = predict(X_test, parameters)

Y_pred = Y_pred.reshape(1, Y_pred.shape[1])
for i in range(len(Y_pred[0])):
    if Y_pred[0][i] >= 1.5:
        Y_pred[0][i] = 2
    else:
        Y_pred[0][i] = 1
print(Y_pred)
wrongnum = 0
for i in range(len(Y_pred[0])):
    if Y_pred[0][i] != Y_test[0][i]:
        wrongnum += 1
print(wrongnum)
# Y_test = Y_test.reshape(1, Y_test.shape[0])
