'''
交叉熵误差 - 损失函数
'''
import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))

if __name__ == '__main__':
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    res1 = cross_entropy_error(np.array(y1), np.array(t))
    res2 = cross_entropy_error(np.array(y2), np.array(t))
    print('y1的损失函数：', res1)
    print('y2的损失函数：', res2)