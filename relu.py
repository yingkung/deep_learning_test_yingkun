'''
relu函数, 大于0的部分为自身，小于0的取0
'''
import numpy as np
import matplotlib.pylab as plt
from sigmoid import sigmoid

def relu(x):
    return np.maximum(x, 0)

if __name__ == '__main__':
    x = np.arange(-10, 10, 0.1)
    y1 = relu(x)
    y2 = sigmoid(x)
    # 移动坐标轴
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))

    plt.plot(x, y1)  # relu的曲线
    plt.plot(x, y2)  # sigmoid 的曲线
    plt.legend(['relu', 'sigmoid'])  # 添加图例
    plt.ylim([0, 3])  # 调整y坐标轴的范围
    plt.xlim([-3,3])
    plt.show()


