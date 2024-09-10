import matplotlib.pyplot as plt
from src import dynamics

if __name__ == '__main__':
    # data_file = 'D:/PycharmProjects/ReservoirComputer/Dataset/NARMA.txt'

    # time, dynamical_system = dsg.LoadDataset(data_file)

    num_init = 2000
    num_train = 6000
    num_test = 2000
    num_step = num_init+num_train+num_test+1  # 动态系统总长度（+1防止报错）

    time,dynamical_system = dynamics.NARMA_10(num_step=num_step)  # 生成动态系统数据

    # 对动态系统初始数据进行分割
    initialization_set, training_set, testing_set = dynamics.Dataset_makeup(time,dynamical_system,
                                                                            num_init=num_init,
                                                                            num_train=num_train,
                                                                            num_test=num_test)

    # 检视数据完备程度
    t, input, output = training_set
    plt.plot(t, input[0,:])
    plt.show(block=True)

    # 数据保存模块
    data_directory = 'D:/PycharmProjects/ReservoirComputer/Dataset'  # 数据文件存放的文件夹
    file_train = 'NARMA_training'
    file_test ='NARMA_testing'
    # 保存训练集
    NARMA_train = dynamics.GenDataset(data=training_set,file_name=file_train,saving_directory=data_directory)
    NARMA_train.SaveDataset()
    # 保存测试集
    NARMA_test = dynamics.GenDataset(data=testing_set, file_name=file_test, saving_directory=data_directory)
    NARMA_test.SaveDataset()