import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from .colors import crayons  # 导入色彩值字典
from src import dynamics, Evaluation

# 全局画图设置：一些用于文章级结果图的matplotlib参数，可以作为matplotlib的全局变量载入
def GlobalSetting(**kwargs):
    # 设置刻度线方向
    plt.rcParams['xtick.direction'] = 'in'  # 将x轴的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度线方向设置向内

    # 确认是否显示刻度线
    bottom_tick = kwargs['bottom_tick'] if 'bottom_tick' in kwargs else True  # 底坐标轴刻度
    top_tick = kwargs['top_tick'] if 'top_tick' in kwargs else False  # 顶坐标轴刻度
    left_tick = kwargs['left_tick'] if 'left_tick' in kwargs else True  # 左坐标轴刻度
    right_tick = kwargs['right_tick'] if 'right_tick' in kwargs else False  # 右坐标轴刻度
    plt.tick_params(bottom=bottom_tick, top=top_tick, left=left_tick, right=right_tick)
    figsize = kwargs['figsize'] if 'figsize' in kwargs else (6.4,4.8)
    plt.figure(figsize=figsize)

    # 创建图例对象
    ax = plt.subplot(111)  # 注意有些参数（比如刻度）一般都在ax中设置,不在plot中设置

    for m in ['top', 'bottom', 'left', 'right']:
        ax.spines[m].set_linewidth(0.5)  # 设置图像边框粗细

    # 刻度参数
    x_major_tick = kwargs['x_major_tick'] if 'x_major_tick' in kwargs else 10  # 设置x轴主刻度标签
    y_major_tick = kwargs['y_major_tick'] if 'y_major_tick' in kwargs else 10  # 设置y轴主刻度标签
    x_minor_tick = kwargs['x_minor_tick'] if 'x_minor_tick' in kwargs else x_major_tick / 5.0  # 设置x轴次刻度标签
    y_minor_tick = kwargs['y_minor_tick'] if 'y_minor_tick' in kwargs else y_major_tick / 5.0  # 设置y轴次刻度标签

    # 控制是否关闭坐标轴刻度
    hide_tick = kwargs['hide_tick'] if 'hide_tick' in kwargs else ''  # 控制关闭哪根坐标轴的刻度
    if hide_tick == 'x':
        ax.set_xticks([])  # 设置x轴刻度为空
    elif hide_tick == 'y':
        ax.set_yticks([])  # 设置y轴刻度为空
    elif hide_tick == 'both':
        ax.set_xticks([])  # 设置x轴刻度为空
        ax.set_yticks([])  # 设置y轴刻度为空
    else:
        # 设置主刻度
        x_major_locator = MultipleLocator(x_major_tick)  # 将x主刻度标签设置为x_major_tick的倍数
        y_major_locator = MultipleLocator(y_major_tick)  # 将y主刻度标签设置为y_major_tick的倍数
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        # 设置次刻度
        x_minor_locator = MultipleLocator(x_minor_tick)  # 将x主刻度标签设置为x_major_tick/5.0的倍数
        y_minor_locator = MultipleLocator(y_minor_tick)  # 将y主刻度标签设置为y_major_tick/5.0的倍数
        ax.xaxis.set_minor_locator(x_minor_locator)
        ax.yaxis.set_minor_locator(y_minor_locator)
    # 设置刻度文本选项
    # 控制是否隐藏刻度文本的模块
    hide_ticklabel = kwargs['hide_ticklabel'] if 'hide_ticklabel' in kwargs else ''  # 控制隐藏哪根坐标轴的刻度
    if hide_ticklabel == 'x':
        ax.xaxis.set_ticklabels([])  # 设置x轴刻度标签为空
    elif hide_ticklabel == 'y':
        ax.yaxis.set_ticklabels([])  # 设置y轴刻度标签为空
    elif hide_ticklabel == 'both':
        ax.xaxis.set_ticklabels([])  # 设置x轴刻度标签为空
        ax.yaxis.set_ticklabels([])  # 设置y轴刻度标签为空
    else:
        pass

    # 设置主次刻度线长度
    plt.tick_params(which='major', length=3, width=0.5)  # 设置主刻度长度
    plt.tick_params(which='minor', length=1.5, width=0.5)  # 设置次刻度长度

    # 设置全局字体选项
    font_type = kwargs['font_type'] if 'font_type' in kwargs else 'Arial'  # 默认字体为Arial
    font_config = {'font.family': font_type, 'font.weight': 'normal', 'font.size': 12}  # font.family设定所有字体为font_type
    plt.rcParams.update(font_config)  # 但是对于希腊字母(e.g. α, β, γ等)跟各种数学符号之类的不适用, Latex语法如Γ会被判断为None
    return

###############################################################################################################
# 动态系统及回声状态网络绘图模块
# 这个函数可以对Dynamical_Systems模块中生成的三维动态系统进行初步可视化，可用于初步检验动态系统的是否健康（有效）
def VisualizeDynamicalSystem_3D(dynamical_system,**kwargs):
    data_rearranged = dynamics.Rearrange(dynamical_system)  # 数据重整化
    t,x,y,z = data_rearranged  # 对数据进行解压，依次是时间步，x坐标，y坐标，z坐标

    color = kwargs['color'] if 'color' in kwargs else crayons('Navy Blue')  # 默认颜色

    ax = plt.subplot(projection='3d')  # 创建三维图图像
    ax.plot(x, y, z, color=color)      # 画图

    ax.grid(False)  # 隐藏背景网格线
    plt.show()      # 显示图片（用console运行代码时，只有加这一句代码才能plot图）

    return

# 此函数可用于可视化神经网络模拟动态系统的输出结果
# ground_truth跟network_output的格式皆为[[x1,y1,z1],[x2,y2,z2], ..., [xn,yn,zn]]，其中[xn,yn,zn]是轨迹坐标点
def QuickView(ground_truth,network_output):
    # 重整输入变量
    num_point = len(ground_truth)  # 轨迹总点数
    t = range(0, len(ground_truth))  # 时间步个数即轨迹的点数
    x_truth, y_truth, z_truth = [[ground_truth[i][0] for i in range(num_point)],
                                 [ground_truth[j][1] for j in range(num_point)],
                                 [ground_truth[k][2] for k in range(num_point)]]
    x_network, y_network, z_network = [[network_output[i][0] for i in range(num_point)],
                                       [network_output[j][1] for j in range(num_point)],
                                       [network_output[k][2] for k in range(num_point)]]

    # 画图模块
    fig = plt.figure(figsize=(10, 10))  # 控制图像大小
    grid = plt.GridSpec(3, 3, wspace=0.6, hspace=0.4)  # 创建柔性网格用于空间分配，输入为(行数, 列数)
    # wspace和hspace可以调整子图间距

    # 分配子图位置
    # X-Z相位图
    # 时序图
    t_x_map = fig.add_subplot(grid[0, :])  # t-x关系
    t_y_map = fig.add_subplot(grid[1, :], sharex=t_x_map)  # t-y
    t_z_map = fig.add_subplot(grid[2, :], sharex=t_x_map)  # t-z
    # sharex的设置会使t_x，t_y和t_z拥有一模一样的横坐标轴

    t_x_map.plot(t, x_truth)
    t_x_map.plot(t, x_network)

    t_y_map.plot(t, y_truth)
    t_y_map.plot(t, y_network)

    t_z_map.plot(t, z_truth)
    t_z_map.plot(t, z_network)

    t_x_map.tick_params('x', labelbottom=False)  # 对于subplot，要调整坐标轴刻度样式的话，需要采用tick_params函数
    t_y_map.tick_params('x', labelbottom=False)  # 如果用别的函数如set_xticklabels()，sharex的设置会把这个函数拷贝到所有share的轴上

    plt.show()

    return

# 这个函数可用于可视化Reservoir Computing模型对三维动态系统的拟合及预测结果，以用于快速分析数据
# 输入变量应满足：真实数据的总点数 = 网络训练集点数+网络预测集点数
def DynamicalSystemApproximation(ground_truth,network_training,network_predicting,**kwargs):
    truth_rearranged = dynamics.Rearrange(ground_truth)  # 默认输入的数据未经重整化
    training_rearranged = dynamics.Rearrange(network_training)
    predicting_rearranged = dynamics.Rearrange(network_predicting)

    # 解压数据
    t_truth, x_truth, y_truth, z_truth = truth_rearranged
    t_train, x_train, y_train, z_train = training_rearranged
    t_predict, x_predict, y_predict, z_predict = predicting_rearranged

    # 对真实数据进行分割，得到训练部分和预测部分，用于画x-z相图
    train_step, predict_step = [len(t_train), len(t_predict)]
    x_truth_train, x_truth_predict = [x_truth[0:train_step], x_truth[train_step:train_step + predict_step]]
    # y_truth_train, y_truth_predict = [y_truth[0:train_step], y_truth[train_step:train_step+predict_step]]
    z_truth_train, z_truth_predict = [z_truth[0:train_step], z_truth[train_step:train_step + predict_step]]
    # 由于网络的预测部分的时间经重整化之后是从零开始的，所以要加上训练步数
    t_predict = t_predict + train_step

    # 画图模块
    fig = plt.figure(figsize=(9, 9))  # 控制图像大小
    grid_mesh = (15, 15)  # matplotlib子图分配时使用的柔性网格尺寸
    height, length = grid_mesh  # 从网格尺寸中获取高（行数）和长（列数）的网格数
    plt.GridSpec(height, length, wspace=0.8, hspace=0.8)  # 创建柔性网格用于空间分配，wspace和hspace可以调整子图间距

    # 分配子图位置
    # X-Z相位图
    phase_training = plt.subplot2grid(grid_mesh, (0, 0), rowspan=6, colspan=7)  # 训练结果
    phase_predicting = plt.subplot2grid(grid_mesh, (0, 8), rowspan=6, colspan=7)  # 网络预测结果
    # phase_training = fig.add_subplot(grid[:4.5, :4])   # 训练结果
    # phase_predicting = fig.add_subplot(grid[:4.5, -4:])  # 网络预测结果
    # 时序图
    t_x = plt.subplot2grid(grid_mesh, (7, 0), rowspan=2, colspan=length)  # t-x时序图
    t_y = plt.subplot2grid(grid_mesh, (9, 0), rowspan=2, colspan=length)  # t-y时序图
    t_z = plt.subplot2grid(grid_mesh, (11, 0), rowspan=2, colspan=length)  # t-z时序图
    error_map = plt.subplot2grid(grid_mesh, (13, 0), rowspan=2, colspan=length)  # 误差时序图

    # 以下是一些常用的画图参数
    linewidth = kwargs['linewidth'] if 'linewidth' in kwargs else 2.0  # 线宽，默认为2.0
    tick_size = kwargs['tick_size'] if 'tick_size' in kwargs else 10  # 控制x轴跟y轴（数字）刻度大小的参数
    label_size = kwargs['label_size'] if 'label_size' in kwargs else 10  # 控制坐标轴名称大小的参数
    legend_size = kwargs['legend_size'] if 'legend_size' in kwargs else 10  # 控制图例大小的参数
    title_size = kwargs['title_size'] if 'title_size' in kwargs else 12  # 控制标题大小的参数
    # 线条颜色
    color_truth = kwargs['color_truth'] if 'color_truth' in kwargs else crayons['Red']
    color_network = kwargs['color_network'] if 'color_network' in kwargs else crayons['Navy Blue']
    color_error = kwargs['color_error'] if 'color_error' in kwargs else crayons['Black']
    # X-Z相图的范围
    protrait_xrange = kwargs['protrait_xrange'] if 'protrait_xrange' in kwargs else (-27, 27)  # x轴范围（X）
    protrait_zrange = kwargs['protrait_zrange'] if 'protrait_zrange' in kwargs else (3, 57)  # y轴范围（Z）
    # 时序图的y轴范围
    t_x_range = kwargs['t_x_range'] if 't_x_range' in kwargs else (-35, 40)
    t_y_range = kwargs['t_y_range'] if 't_y_range' in kwargs else (-35, 42)
    t_z_range = kwargs['t_z_range'] if 't_z_range' in kwargs else (-5, 70)
    error_range = kwargs['error_range'] if 'error_range' in kwargs else (0, 110)

    # 设置子图刻度，细节请参考：matplotlib.axes.Axes.locator_params()
    # 链接：https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.locator_params.html#matplotlib.axes.Axes.locator_params
    # 例子：https://matplotlib.org/stable/tutorials/intermediate/constrainedlayout_guide.html
    locator_factors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 刻度标识位置（locator）的因子，只要是列表中元素的倍数处都可以允许存在刻度
    # nbins_protrait = 4  # nbins - Maximum number of intervals; one less than max number of ticks
    nbins_series_x = 8
    nbins_series_y = 3
    t_x.locator_params('y', nbins=nbins_series_y, steps=locator_factors)  # 关于nbins参数，见官网及上面备注
    t_y.locator_params('y', nbins=nbins_series_y, steps=locator_factors)  # 感觉效果有争议，出来感觉就是nbins=ticks个数
    t_z.locator_params('y', nbins=nbins_series_y, steps=locator_factors)
    error_map.locator_params('x', nbins=nbins_series_x)
    error_map.locator_params('y', nbins=nbins_series_y)

    # 画x-z相图，phase_training为训练阶段（Training stage）相图，phase_predicting为预测阶段（Predicting stage）相图
    phase_portrait_fig = [phase_training, phase_predicting]
    legends = [['Ground truth', 'Network output'], ['Ground truth', 'Network output']]
    titles = ['X-Z phase portrait of training stage', 'X-Z phase portrait of predicting stage']
    X_truth_data, Z_truth_data = [[x_truth_train, x_truth_predict], [z_truth_train, z_truth_predict]]  # 目标系统数据
    X_network_data, Z_network_data = [[x_train, x_predict], [z_train, z_predict]]  # 网络输出数据
    for i in range(2):
        # 画目标系统
        phase_portrait_fig[i].plot(X_truth_data[i], Z_truth_data[i], linewidth=linewidth, color=color_truth)
        # 画网络输出
        phase_portrait_fig[i].plot(X_network_data[i], Z_network_data[i], linewidth=linewidth, color=color_network)
        phase_portrait_fig[i].set_xlabel('X', size=label_size)  # 设置x轴名称
        phase_portrait_fig[i].set_ylabel('Z', size=label_size)  # 设置y轴名称
        # phase_portrait_fig[i].legend(labels=legends[i],loc='best',fontsize=legend_size,frameon=False)  # 设置图例参数
        phase_portrait_fig[i].legend(labels=legends[i], loc='upper left', fontsize=legend_size, frameon=False)  # 设置图例参数
        phase_portrait_fig[i].set_title(titles[i], size=title_size)  # 设置标题
        phase_portrait_fig[i].set_xlim(protrait_xrange[0], protrait_xrange[1])  # 设置x轴范围
        phase_portrait_fig[i].set_ylim(protrait_zrange[0], protrait_zrange[1])  # 设置y轴范围
        phase_portrait_fig[i].tick_params('both', labelsize=tick_size)  # 设置刻度样式

    # 画t-x，t-y，t-z时序图
    time_series_fig = [t_x, t_y, t_z]
    truth_data = [x_truth, y_truth, z_truth]
    train_data = [x_train, y_train, z_train]
    predict_data = [x_predict, y_predict, z_predict]
    plotting_range = [t_x_range, t_y_range, t_z_range]
    label = ['X', 'Y', 'Z']
    for i in range(3):
        time_series_fig[i].plot(t_truth, truth_data[i], color=color_truth, label='Ground truth')  # Ground truth
        time_series_fig[i].plot(t_train, train_data[i], color=color_network,
                                label='Network output')  # Network train output
        time_series_fig[i].plot(t_predict, predict_data[i], color=color_network)  # Netwrok predict output
        time_series_fig[i].vlines(train_step, plotting_range[i][0], plotting_range[i][1], color='k', linestyles='-.')
        # 画图设置
        # time_series_fig[i].legend(loc='upper left',frameon=False,ncol=2)  # ncol参数表示图例中的元素列数，可以以此设置图例横排
        time_series_fig[i].legend(loc=(0.01, 0.73), frameon=False, ncol=2)  # ncol参数表示图例中的元素列数，可以以此设置图例横排
        time_series_fig[i].set_ylabel(label[i], size=label_size)
        time_series_fig[i].set_xlim(0, len(ground_truth))
        time_series_fig[i].set_ylim(plotting_range[i][0], plotting_range[i][1])
        # 对于subplot，要调整坐标轴刻度样式的话，需要采用tick_params函数
        # 如果用别的函数如set_xticklabels()，sharex的设置会把这个函数拷贝到所有share的轴上
        time_series_fig[i].tick_params('x', labelbottom=False)
        time_series_fig[i].tick_params('y', labelsize=tick_size, labelbottom=False)

    # 误差时序图
    network_output = np.vstack((network_training, network_predicting))  # 将网络的训练结果跟预测结果拼接到一起
    # print(len(network_output))
    # print(len(ground_truth))
    error = []
    for i in range(len(network_output)):
        e = Evaluation.Deviation(network_output[i], ground_truth[i])
        error.append(e)
    error_map.plot(t_truth, error, color=color_error)
    error_map.vlines(train_step, error_range[0], error_range[1], color='k', linestyles='-.')
    error_map.set_xlim(0, len(ground_truth))
    error_map.set_ylim(error_range[0], error_range[1])
    error_map.set_xlabel('Time step', size=label_size)
    error_map.set_ylabel('Error', size=label_size)
    error_map.tick_params('both', labelsize=tick_size)  # 设置刻度样式

    fig.align_labels()  # 多子图标签对齐，参见https://blog.csdn.net/itnerd/article/details/109628273

    return

if __name__=='__main__':
    x1 = np.linspace(0,5,100)
    x2 = np.linspace(0,10,100)
    y1 = np.cos(x1)
    y2 = np.sin(x2)