import numpy as np
import matplotlib.pyplot as plt
from src import Activation

if __name__ == '__main__':
    v = np.linspace(-5, 5, 100)

    # 画出激活函数的图像
    plt.figure()

    plt.plot(v, Activation.I_Taylor(v), label='Original output characteristics')
    plt.plot(v, Activation.I_Taylor_w_OperationalRange(v, operational_range=(-3,3)), label='Clipped at Operational Range')

    plt.show(block=True)  # 显示图片（非console运行代码时，只有加这一句代码才能plot图）