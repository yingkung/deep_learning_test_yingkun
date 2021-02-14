'''
softmax 函数
'''
import numpy as np


# def softmax(x):
#     exp_x = np.exp(x)
#     sum_exp_x = np.sum(exp_x)
#     y = exp_x / sum_exp_x
#     return y

# 上面的函数存在一个问题：就是溢出
# 因为e的幂会出现非常大的值，造成溢出
def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y


if __name__ == "__main__":
    x = np.array([0.3, 2.9, 0.4])
    y = softmax(x)
    print(y)
