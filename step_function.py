'''
创建阶跃函数
'''

# def step_function(x):
#     if x >0:
#         return 1
#     else:
#         return 0

# 将上述代码修改为支持numpy数组的形式
import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    y = x > 0
    return y.astype(np.int)


if __name__ == '__main__':
    x = np.arange(-5, 5, 0.01)
    y = step_function(x)
    # 移动坐标轴
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))


    plt.plot(x, y, marker='*')
    plt.show()
    print('OK')