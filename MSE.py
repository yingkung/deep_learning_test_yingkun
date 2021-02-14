'''
均方误差 - 损失函数
'''
import numpy as np


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


if __name__ == '__main__':
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    res1 = mean_squared_error(np.array(y1), np.array(t))
    res2 = mean_squared_error(np.array(y2), np.array(t))
    print('y1的损失函数：', res1)
    print('y2的损失函数：', res2)

