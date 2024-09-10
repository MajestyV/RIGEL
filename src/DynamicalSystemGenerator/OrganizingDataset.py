# 此函数可以对时间序列进行分割，从而得到Reservoir computing所需的数据集
# 对动态系统数据进行切片以得到我们的训练集跟预测集（应注意，python切片是左闭右开的，如[3:6]只包含下标为3，4，5的）
# 同时，应注意做切片时要用未重整化的数据
# num_intial - 初始化（initialization）阶段，要舍弃的数据（前面的点可能包含初始点的信息，会是我们的拟合偏移，因此我们从一定点数之后开始取值）
# num_train - 训练集长度
# num_predict - 测试集长度
# 最重要的是，由于python的indexing从0开始，所以必须要保证时序信号长度data_length > num_discard+num_train+num_predict
def Dataset_makeup(time,data_series,num_init=2000,num_train=2000,num_test=2000):
    # 初始化阶段的数据
    init_start, init_end = [0, num_init]  # 初始化阶段的起点跟终点
    t_init = time[init_start:init_end]
    x_init = data_series[:, init_start:init_end]
    y_init = data_series[:, (init_start + 1):(init_end + 1)]
    initialization_set = (t_init, x_init, y_init)
    # 训练集
    train_start, train_end = [num_init, num_init + num_train]  # 训练集的起点跟终点
    t_train = time[train_start:train_end]
    x_train = data_series[:,train_start:train_end]
    y_train = data_series[:,(train_start+1):(train_end+1)]
    training_set = (t_train,x_train,y_train)
    # 测试集
    test_start, test_end = [num_init + num_train, num_init + num_train + num_test]  # 测试集的起点跟终点
    t_test = time[test_start:test_end]
    x_test = data_series[:,test_start:test_end]
    y_test = data_series[:,(test_start+1):(test_end+1)]
    testing_set = (t_test,x_test,y_test)

    return initialization_set, training_set, testing_set