import numpy as np
import random


def load_data(path):
    data = np.loadtxt(path, dtype=np.float32)
    return data


X = load_data("result.txt")

length = X.shape[0]
tem = []
for i in range(length):
    tem.append(i)
index = random.sample(tem, int(length * 0.3))
tem = [tem[i] for i in range(0, length, 1) if i not in index]



test = X[index]
train = X[tem]

np.savetxt("train.txt", train, fmt="%0.8f")
np.savetxt("test.txt", test, fmt="%0.8f")
np.savetxt("train_index.txt", np.array(tem), fmt="%0.8f")
np.savetxt("test_index.txt", np.array(index), fmt="%0.8f")
# print(index)
