import numpy as np
import matplotlib.pyplot as plt
from src import dynamics, DelayRC

if __name__=="__main__":
    # 生成NARMA-10数据集用于进行系统baseline测试
    num_init = 2000
    num_train = 4000
    num_test = 2000
    num_step = num_init + num_train + num_test + 1  # 动态系统总长度（+1防止报错）

    time, dynamical_system = dynamics.NARMA_10(num_step=num_step)  # 生成动态系统数据

    # 对动态系统初始数据进行分割
    initialization_set, training_set, testing_set = dynamics.Dataset_makeup(time, dynamical_system,
                                                                       num_init=num_init,
                                                                       num_train=num_train,
                                                                       num_test=num_test)

    t_train, x_train, y_train = training_set
    t_test, x_test, y_test = testing_set
    x_train,y_train,x_test,y_test = (x_train.T,y_train.T,x_test.T,y_test.T)  # 将数据转置，方便后续计算操作
    # print(x_train.shape,y_train.shape)

    # 调用Delay RC模型
    N = 600  # number of virtual nodes
    # get reservoir activity for trainings data and the initialized model
    r_train, model = DelayRC.Run_delayRC(x_train,N)
    # remove warmup values
    # R_Train = R_Train[warmup_cycles:]

    # calcualte weights, using pseudoinverse
    weights = np.dot(np.linalg.pinv(r_train), y_train)
    # print(weights)

    r_test, _ = DelayRC.Run_delayRC(x_test, N, model)

    # calculate prediction values
    yhat = np.dot(r_test, weights)

    # for calculating the NRMSE, dont use the first 75 values, since the model
    # first needs to get "swinging"
    y_consider = y_test[500:]
    yhat_consider = yhat[500:]

    # calculate normalized root mean squared error
    NRMSE = np.sqrt(np.divide( \
        np.mean(np.square(y_consider - yhat_consider)), \
        np.var(y_consider)))

    # plot
    plt.figure("Prediction Plot")
    plt.title("Prediction of NARMA10 Series with NRMSE {0:.4f}".format(NRMSE))
    plt.plot(y_consider, color="blue", linewidth=2, label="NARMA10")
    plt.plot(yhat_consider, color="red", linewidth=0.5, linestyle='-', label="Model Prediction")
    plt.xlabel("Timesteps")
    plt.legend()
    # plt.savefig("./images/prediction_plot.png")
    plt.show(block=True)

    print("[*] NRMSE = {0:.4f}!".format(NRMSE))