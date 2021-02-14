'''
创建sigmoid 函数
'''

import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))


if __name__ == '__main__':
    # x = np.array([-1.0, 1.0, 2.0])
    x = np.arange(-10, 10, 0.1)
    y = sigmoid(x)
    # 移动坐标轴
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    plt.plot(x, y)
    plt.show()
    print('OK')
