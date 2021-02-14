'''
训练手写数据集
'''
from keras.datasets import mnist
import matplotlib.pyplot as plt
import pickle
from sigmoid import sigmoid
from softmax import softmax
import numpy as np

# 导入手写数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

#  画图其中一张图
# print(y_train[0])
x_train = x_train / 255  # Normalization
# plt.imshow(x_train[0], cmap='gray')
# plt.show()


def init_network():
    # 加载已经训练好的w 和b
    with open(r'D:\0work\深度学习入门\【源代码】深度学习入门：基于Python的理论与实现\ch03\sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    # x 应该是每一幅图
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)
    return y


if __name__ == '__main__':
    network = init_network()
    # x_train = x_train.reshape(60000,784)[:10]
    x_train = x_train.reshape(60000, 784)[0:100]
    batch_size = 10

    for i in range(0,  len(x_train), batch_size):
        x_batch = x_train[i: i+batch_size]  # 加入batch的概念
        # p = predict(network, x_train[i])
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        print('预测值是：', p)
        print('实际值是：', y_train[i: i+batch_size])



