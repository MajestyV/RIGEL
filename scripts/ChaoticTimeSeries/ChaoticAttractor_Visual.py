import matplotlib.pyplot as plt
from src import dynamics, Activation, ESN, VISION, Dataset_makeup

if __name__ == '__main__':
    # 定义要学习的动态系统
    num_step = 10001  # 总步数
    step_length = 0.01
    time, data = dynamics.Lorenz_63(origin=(3, 2, 16), parameter=(10, 28, 8.0 / 3.0),
                                    num_step=num_step, step_length=step_length)

    num_init = 2000  # 前面的点可能包含初始点的信息，会是我们的拟合偏移，因此我们从一定点数之后开始取值
    num_train = 3000  # 训练集长度
    num_test = 3000  # 预测集长度
    initialization_set, training_set, testing_set = Dataset_makeup(time, data,
                                                                   num_init=num_init, num_train=num_train,
                                                                   num_test=num_test)

    # 解压数据
    t_init, x_init, y_init = initialization_set
    t_train, x_train, y_train = training_set
    t_test, x_test, y_test = testing_set

    VISION.QuickView_3D_Trajectory(data)
    plt.show(block=True)  # 显示图片（非console运行代码时，只有加这一句代码才能plot图）