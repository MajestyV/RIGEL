import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
# 直接调用ASHEN包内含的模块，提高效率
from ..Evaluation.Regression import Deviation  # 误差分析函数
from .Initialize import GlobalSetting  # 画图初始化函数
from .colors import iColormap, xkcd_rgb, crayons  # 色彩模块

# This function is designed to provide visual analyze on the performance of reservoir computing.
def Analyze_1D_systems(time, training_set, testing_set, wspace=1, hspace=0.8, **kwargs):
    GlobalSetting()  # 导入全局画图设定

    # 数据解压
    data_length = len(time)  # 数据长度
    t = time[:] - time[0]  # 重整时序，将起点设为0
    output_train, output_train_network = training_set  # 训练集
    output_predict, output_predict_network = testing_set  # 测试集
    train_length, test_length = (output_train.shape[1], output_predict.shape[1])  # 训练集和测试集的长度

    # 画时序图所需数据
    ground_truth = np.hstack((output_train, output_predict))
    network_output = np.hstack((output_train_network, output_predict_network))
    x_truth, x_network = (ground_truth[0,:], network_output[0,:])

    deviation = np.array([Deviation(network_output[:,i], ground_truth[:,i]) for i in range(data_length)])  # 计算每一时刻的绝对误差

    # 画1D动态系统相位图所需数据
    tau = kwargs['tau'] if 'tau' in kwargs else 1  # 相图的延迟参数
    phase_train_truth = np.zeros((train_length-tau,2))
    phase_train_network = np.zeros((train_length-tau,2))
    phase_predict_truth = np.zeros((test_length-tau, 2))
    phase_predict_network = np.zeros((test_length-tau, 2))
    for i in range(train_length-tau):
        phase_train_truth[i] = np.array([output_train[0,i+tau],output_train[0,i]])
        phase_train_network[i] = np.array([output_train_network[0,i+tau], output_train_network[0,i]])
    for i in range(test_length-tau):
        phase_predict_truth[i] = np.array([output_predict[0,i+tau],output_predict[0,i]])
        phase_predict_network[i] = np.array([output_predict_network[0,i+tau], output_predict_network[0,i]])

    # 画图模块
    fig = plt.figure(figsize=(9, 10))  # 控制图像大小，（长，高）
    grid = plt.GridSpec(5, 4, wspace=wspace, hspace=hspace)  # 创建柔性网格用于空间分配，输入为(行数, 列数)
    # wspace和hspace可以调整子图间距

    # 色彩参数
    color_truth = kwargs['color_truth'] if 'color_truth' in kwargs else crayons['Navy Blue']
    color_network = kwargs['color_network'] if 'color_network' in kwargs else crayons['Red Orange']
    color_deviation = kwargs['color_deviation'] if 'color_deviation' in kwargs else 'k'

    # 时序图以及误差分析图绘制
    # 分配子图位置
    t_x_truth = fig.add_subplot(grid[0,:])  # t-x时序图 - 目标系统
    t_x_network = fig.add_subplot(grid[1,:],sharex=t_x_truth)  # t-x时序图 - 网络预测结果
    deviation_map = fig.add_subplot(grid[2,:], sharex=t_x_truth)  # 误差分析图
    # 画图
    t_x_truth.plot(t, x_truth, color=color_truth)
    t_x_network.plot(t, x_network, color=color_network)
    deviation_map.plot(t, deviation, color=color_deviation)

    # X-Z相位图
    phase_portrait_mode = kwargs['phase_protrait_mode'] if 'phase_protrait_mode' in kwargs else 'seperated'  # 默认为将目标系统跟网络预测结果分开展示
    phase_portrait_style = kwargs['phase_protrait_style'] if 'phase_protrait_style' in kwargs else 'scatter'  # 默认为散点图
    if phase_portrait_mode == 'seperated':
        # 将目标系统跟网络预测结果分开展示
        truth_diagram = fig.add_subplot(grid[3:, :2])
        network_diagram = fig.add_subplot(grid[3:, 2:])
        if phase_portrait_style == 'scatter':
            truth_diagram.scatter(phase_predict_truth[:, 0], phase_predict_truth[:, 1], c=color_truth, s=2)
            network_diagram.scatter(phase_predict_network[:, 0], phase_predict_network[:, 1], c=color_network, s=2)
        elif phase_portrait_style == 'curve':
            truth_diagram.plot(phase_predict_truth[:, 0], phase_predict_truth[:, 1], color=color_truth)
            network_diagram.plot(phase_predict_network[:, 0], phase_predict_network[:, 1], color=color_network)
        else:
            print('Please specify a valid phase protrait style ! ! !')
            pass
    elif phase_portrait_mode == 'merged':
        # 将训练阶段跟预测阶段分开展示
        training_diagram = fig.add_subplot(grid[3:, :2])
        testing_diagram = fig.add_subplot(grid[3:, 2:])
        if phase_portrait_style == 'scatter':
            training_diagram.scatter(phase_train_truth[:,0],phase_train_truth[:,1],c=color_truth,s=2)
            training_diagram.scatter(phase_train_network[:,0],phase_train_network[:,1],c=color_network,s=2)
            testing_diagram.scatter(phase_predict_truth[:, 0], phase_predict_truth[:, 1], c=color_truth,s=2)
            testing_diagram.scatter(phase_predict_network[:, 0], phase_predict_network[:, 1], c=color_network,s=2)
        elif phase_portrait_style == 'curve':
            training_diagram.plot(phase_train_truth[:,0],phase_train_truth[:,1],color=color_truth)
            training_diagram.plot(phase_train_network[:,0],phase_train_network[:,1],color=color_network)
            testing_diagram.plot(phase_predict_truth[:, 0], phase_predict_truth[:, 1], color=color_truth)
            testing_diagram.plot(phase_predict_network[:, 0], phase_predict_network[:, 1], color=color_network)
        else:
            print('Please specify a valid phase protrait style ! ! !')
            pass
    else:
        print('Please specify a valid phase protrait mode ! ! !')
        pass

    # 细节设置
    data_min, data_max = kwargs['data_range'] if 'data_range' in kwargs else (0, 1.5)
    deviation_min, deviation_max = kwargs['deviation_range'] if 'deviation_range' in kwargs else (-0.2, 1.2)

    t_x_truth.set_xlim(min(t), max(t))
    t_x_truth.set_ylim(data_min, data_max)
    t_x_network.set_ylim(data_min, data_max)
    deviation_map.set_ylim(deviation_min, deviation_max)

    plt.show(block=True)

    return

# This function is designed to provide visual analyze on the performance of reservoir computing.
def Analyze_3D_systems(time, training_set, testing_set, wspace=1, hspace=0.8, mode='3D_projection', **kwargs):
    # GlobalSetting()  # 导入全局画图设定

    # 数据解压
    data_length = len(time)  # 数据长度
    # print(data_length)
    t = time[:] - time[0]  # 重整时序，将起点设为0
    output_train, output_train_network = training_set  # 训练集
    output_test, output_test_network = testing_set  # 测试集
    sep_line_pos = t[output_test.shape[1]]  # 训练集与测试集分割线位置


    # 画时序图所需数据
    ground_truth = np.hstack((output_train, output_test))
    network_output = np.hstack((output_train_network, output_test_network))
    x_truth, y_truth, z_truth = ground_truth
    x_network, y_network, z_network = network_output

    deviation = np.array([Deviation(network_output[:,i], ground_truth[:,i]) for i in range(data_length)])  # 计算每一时刻的绝对误差

    # 画3D相位图所需数据
    # ground truth in training stage (t_t)
    xt_t, yt_t, zt_t = output_train
    # network output in training stage (n_t)
    xn_t, yn_t, zn_t = output_train_network
    # ground truth in predicting stage (t_p)
    xt_p, yt_p, zt_p = output_test
    # network output in predicting stage (n_p)
    xn_p, yn_p, zn_p = output_test_network

    # 画图模块
    fig = plt.figure(figsize=(12, 6))  # 控制图像大小
    grid = plt.GridSpec(4, 6, wspace=wspace, hspace=hspace)  # 创建柔性网格用于空间分配，输入为(行数, 列数)
    # wspace和hspace可以调整子图间距

    # 画图参数
    color_truth = kwargs['color_truth'] if 'color_truth' in kwargs else crayons['Navy Blue']
    color_network = kwargs['color_network'] if 'color_network' in kwargs else crayons['Red Orange']
    color_deviation = kwargs['color_deviation'] if 'color_deviation' in kwargs else 'k'

    # 分配子图
    # 系统图
    if mode == '3D_projection':
        # 分配3D投影图位置
        training_diagram = fig.add_subplot(grid[:2, :2], projection='3d')
        testing_diagram = fig.add_subplot(grid[-2:, :2], projection='3d')
        # 训练阶段
        training_diagram.plot(xt_t, yt_t, zt_t, color=color_truth)
        training_diagram.plot(xn_t, yn_t, zn_t, color=color_network)
        training_diagram.grid(False)  # 隐藏背景网格线
        # 测试阶段
        testing_diagram.plot(xt_p, yt_p, zt_p, color=color_truth)
        testing_diagram.plot(xn_p, yn_p, zn_p, color=color_network)
        testing_diagram.grid(False)  # 隐藏背景网格线
    elif mode == 'X-Z_phase_portrait':
        # 分配X-Z相位图
        training_diagram = fig.add_subplot(grid[:2, :2])
        testing_diagram = fig.add_subplot(grid[-2:, :2])
        # 训练阶段
        training_diagram.plot(xt_t, zt_t, color=color_truth)
        training_diagram.plot(xn_t, zn_t, color=color_network)
        # 测试阶段
        testing_diagram.plot(xt_p, zt_p, color=color_truth)
        testing_diagram.plot(xn_p, zn_p, color=color_network)
    else:
        print('Please select an appropriate visualization mode !!!')
        pass

    # 时序分析图
    t_x_map = fig.add_subplot(grid[0, -4:])  # t-x关系
    t_y_map = fig.add_subplot(grid[1, -4:], sharex=t_x_map)  # t-y
    t_z_map = fig.add_subplot(grid[2, -4:], sharex=t_x_map)  # t-z
    deviation_map = fig.add_subplot(grid[3, -4:], sharex=t_x_map)
    # sharex的设置会使t_x，t_y和t_z拥有一模一样的横坐标轴

    t_x_map.plot(t, x_truth, color=color_truth)
    t_x_map.plot(t, x_network, color=color_network)

    t_y_map.plot(t, y_truth, color=color_truth)
    t_y_map.plot(t, y_network, color=color_network)

    t_z_map.plot(t, z_truth, color=color_truth)
    t_z_map.plot(t, z_network, color=color_network)

    deviation_map.plot(t, deviation, color=color_deviation)  # 误差时序图

    # 细节设置
    t_x_map.set_xlim(min(t),max(t))

    x_min, x_max = kwargs['x_range'] if 'x_range' in kwargs else (-22,22)
    y_min, y_max = kwargs['y_range'] if 'y_range' in kwargs else (-28, 28)
    z_min, z_max = kwargs['z_range'] if 'z_range' in kwargs else (0, 60)
    deviation_min, deviation_max = kwargs['deviation_range'] if 'deviation_range' in kwargs else (0, 2)

    t_x_map.set_ylim(x_min,x_max)
    t_y_map.set_ylim(y_min,y_max)
    t_z_map.set_ylim(z_min,z_max)
    deviation_map.set_ylim(deviation_min,deviation_max)

    t_x_map.vlines(sep_line_pos, x_min, x_max, colors='k', linestyles='dashed', linewidth=1.0)
    t_y_map.vlines(sep_line_pos, y_min, y_max, colors='k', linestyles='dashed', linewidth=1.0)
    t_z_map.vlines(sep_line_pos, z_min, z_max, colors='k', linestyles='dashed', linewidth=1.0)
    deviation_map.vlines(sep_line_pos, deviation_min, deviation_max, colors='k', linestyles='dashed', linewidth=1.0)

    # plt.show(block=True)  # 因为savefig需要在show之前，所以这里不加show，在外部调用时再加show

    return

# 此函数专用于可视化ESN网络长期预测的结果，用以检测网络的capability of general synchronization
def Analyze_3D_systems_synchronization(time, training_set, testing_set, wspace=1, hspace=0.8, **kwargs):
    GlobalSetting()  # 导入全局画图设定

    # 数据解压
    data_length = len(time)  # 数据长度
    t = time[:] - time[0]  # 重整时序，将起点设为0
    output_train, output_train_network = training_set  # 训练集
    output_test, output_test_network = testing_set  # 测试集

    # 画时序图所需数据
    ground_truth = np.hstack((output_train, output_test))
    network_output = np.hstack((output_train_network, output_test_network))
    x_truth, y_truth, z_truth = ground_truth
    x_network, y_network, z_network = network_output

    deviation = np.array([Deviation(network_output[:,i], ground_truth[:,i]) for i in range(data_length)])  # 计算每一时刻的绝对误差

    # 画3D相位图所需数据

    xt_t, yt_t, zt_t = output_train  # ground truth in training stage (t_t)
    xn_t, yn_t, zn_t = output_train_network  # network output in training stage (n_t)
    xt_p, yt_p, zt_p = output_test  # ground truth in predicting stage (t_p)
    xn_p, yn_p, zn_p = output_test_network  # network output in predicting stage (n_p)

    # 画图模块
    fig = plt.figure(figsize=(12, 6))  # 控制图像大小
    grid = plt.GridSpec(4, 6, wspace=wspace, hspace=hspace)  # 创建柔性网格用于空间分配，输入为(行数, 列数)
    # wspace和hspace可以调整子图间距

    # 画图参数
    color_truth = kwargs['color_truth'] if 'color_truth' in kwargs else crayons['Navy Blue']
    color_network = kwargs['color_network'] if 'color_network' in kwargs else crayons['Red Orange']
    color_deviation = kwargs['color_deviation'] if 'color_deviation' in kwargs else 'k'

    # 分配子图
    # 分配3D投影图位置
    target_system = fig.add_subplot(grid[:2, :2], projection='3d')  # 目标系统
    reconstructed_system = fig.add_subplot(grid[-2:, :2], projection='3d')  # ESN重建结果
    # 分配时序分析图位置
    t_x_map = fig.add_subplot(grid[0, -4:])  # t-x关系
    t_y_map = fig.add_subplot(grid[1, -4:], sharex=t_x_map)  # t-y
    t_z_map = fig.add_subplot(grid[2, -4:], sharex=t_x_map)  # t-z
    deviation_map = fig.add_subplot(grid[3, -4:], sharex=t_x_map)  # sharex的设置会使t_x，t_y和t_z拥有一模一样的横坐标轴

    target_system.plot(xt_p, yt_p, zt_p, color=color_truth)  # 目标系统的3D轨迹投影图
    target_system.grid(False)  # 隐藏背景网格线

    reconstructed_system.plot(xn_p, yn_p, zn_p, color=color_network)  # ESN重建结果的3D轨迹投影图
    reconstructed_system.grid(False)  # 隐藏背景网格线

    t_x_map.plot(t, x_truth, color=color_truth)
    t_x_map.plot(t, x_network, color=color_network)

    t_y_map.plot(t, y_truth, color=color_truth)
    t_y_map.plot(t, y_network, color=color_network)

    t_z_map.plot(t, z_truth, color=color_truth)
    t_z_map.plot(t, z_network, color=color_network)

    deviation_map.plot(t, deviation, color=color_deviation)  # 误差时序图

    # 细节设置
    t_x_map.set_xlim(min(t),max(t))

    x_min, x_max = kwargs['x_range'] if 'x_range' in kwargs else (-22,22)
    y_min, y_max = kwargs['y_range'] if 'y_range' in kwargs else (-28, 28)
    z_min, z_max = kwargs['z_range'] if 'z_range' in kwargs else (0, 60)
    deviation_min, deviation_max = kwargs['deviation_range'] if 'deviation_range' in kwargs else (0, 2)

    t_x_map.set_ylim(x_min,x_max)
    t_y_map.set_ylim(y_min,y_max)
    t_z_map.set_ylim(z_min,z_max)
    deviation_map.set_ylim(deviation_min,deviation_max)

    plt.show(block=True)

    return

# This function is designed to provide visual analyze on the performance of reservoir computing.
def Analyze_Spatiotemporal(time, training_data, testing_data, wspace=1, hspace=0.8, **kwargs):
    GlobalSetting()  # 导入全局画图设定

    # 数据解压
    data_length = len(time)  # 数据长度
    t = time[:] - time[0]  # 重整时序，将起点设为0
    output_train, output_train_network = training_data  # 训练集
    output_test, output_test_network = testing_data  # 测试集
    train_length, test_length = (output_train.shape[1],output_test.shape[1])

    # 可以选择加上初始化情况或不加
    including_init = kwargs['including_init'] if 'including_init' in kwargs else False
    if including_init:
        if 'initial_condition' in kwargs:
            init = kwargs['initial_condition']
        else:
            print('Please provide initial condition !!!')
            exit()
        ground_truth = np.hstack((init,output_train, output_test))
        network_output = np.hstack((init,output_train_network, output_test_network))
        sep_line_pos = t[train_length+init.shape[1]]  # 训练集与测试集分割线位置
    else:
        ground_truth = np.hstack((output_train, output_test))
        network_output = np.hstack((output_train_network, output_test_network))
        sep_line_pos = t[train_length]  # 训练集与测试集分割线位置

    # deviation = np.abs(ground_truth-network_output)/np.abs(ground_truth)  # 计算相对偏移误差图
    error = np.abs(ground_truth-network_output)  # 计算绝对误差图
    # error = ground_truth - network_output  # 计算误差图

    # 画图模块
    # 创建二维网格
    L = error.shape[0]
    length = kwargs['length'] if 'length' in kwargs else np.linspace(start=0, stop=L, num=L)
    L_min, L_max = (min(length),max(length))
    xx, tt = np.meshgrid(t, length)

    # 画图参数设置
    figsize = kwargs['figsize'] if 'figsize' in kwargs else (8,4)
    levels_task = kwargs['levels_task'] if 'levels_task' in kwargs else np.arange(-4, 4, 0.2)
    levels_deviation = kwargs['levels_deviation'] if 'levels_deviation' in kwargs else np.arange(0, 8, 0.2)
    cmap_task = kwargs['cmap_task'] if 'cmap_task' in kwargs else cm.viridis
    cmap_deviation = kwargs['cmap_deviation'] if 'cmap_deviation' in kwargs else cm.afmhot_r
    # 关于色条反转：https://blog.csdn.net/weixin_37888343/article/details/108751695

    fig = plt.figure(figsize=figsize)  # 控制图像大小
    plt.subplots_adjust(wspace=wspace, hspace=hspace)  # 空间分配，wspace和hspace可以调整子图间距

    map_1 = fig.add_subplot(3, 1, 1)
    map_2 = fig.add_subplot(3, 1, 2)
    map_3 = fig.add_subplot(3, 1, 3)

    map_1.contourf(xx, tt, ground_truth, levels=levels_task, cmap=cmap_task)
    map_2.contourf(xx, tt, network_output, levels=levels_task, cmap=cmap_task)
    map_3.contourf(xx, tt, error, levels=levels_deviation, cmap=cmap_deviation)

    #map_1.imshow(ground_truth)
    #map_2.imshow(network_output)
    #map_3.imshow(error)

    map_1.vlines(sep_line_pos,L_min,L_max,colors='k',linestyles='dashed',linewidth=0.5)
    map_2.vlines(sep_line_pos, L_min, L_max, colors='k',linestyles='dashed', linewidth=0.5)
    map_3.vlines(sep_line_pos, L_min, L_max, colors='k',linestyles='dashed', linewidth=0.5)

    # plt.tight_layout()  # 防止画图时，图像分布失衡，部分文字显示被遮挡的情况
    # plt.show(block=True)

    return error  # 输出误差结果，方便分析