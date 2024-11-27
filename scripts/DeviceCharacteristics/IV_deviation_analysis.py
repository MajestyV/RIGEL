import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from src import AnalysisToolkit, VISION

working_loc = 'Lingjiang'

data_dir_dict = {'Lingjiang': 'E:/PhD_research/PhaseTransistor/Data/IV characteristics/Evolution/Raw data/M-8精选/Stabilized',
                 'MMW502': 'D:/Projects/PhaseTransistor/Data/Figures/IV characteristics/Evolution/M-8精选/Stablized_cycles',
                 'JCPGH1': 'D:/PhD_research/Gallery/IV characteristics/Evolution/Raw data/M-8精选/Stabilized'}

b2bdm = AnalysisToolkit.B2BDM()
GD = AnalysisToolkit.GetData()        # 调用GetaData模块
CF = AnalysisToolkit.CurveFitting()  # 调用CurveFitting模块

########################################################################################################################
# 常用函数
# 此函数可用于将数据分割为上升和下降两部分
def Spiltting(data) -> tuple:
    data_rising = []
    data_falling = []
    for i in range(len(data)):
        length = len(data[i])
        seperating_point = int(length/2)
        rising_part = list(data[i])[0:seperating_point]
        falling_part = list(data[i])[seperating_point:length]
        data_rising.append(np.array(rising_part))
        data_falling.append(np.array(falling_part))
    return np.array(data_rising), np.array(data_falling)

# 此函数可以将分开储存的数据（具有层次，比如深层列表）整理为一个序列（数组）
def Merging(data) -> np.ndarray:
    merged_data = []
    for i in range(len(data)):
        data_sub = data[i]
        for j in range(len(data_sub)):
            merged_data.append(data_sub[j])

    return np.array(merged_data)

if __name__ == '__main__':
    file_name = ['CycleSet1.xls', 'CycleSet2.xls', 'CycleSet3.xls', 'CycleSet4.xls', 'CycleSet5.xls']

    sheet_list = ['Data', 'Append1', 'Append2', 'Append3', 'Append4', 'Append5']

    data_dict = dict.fromkeys(file_name)  # 根据给定的数据文件名创建字典方便存放数据

    V, I, I_abs = [[], [], []]  # 批量定义变量为空列表
    for n in file_name:
        data_file = f"{data_dir_dict[working_loc]}/{n}"  # 数据文件的绝对地址
        data = GD.GetExcelData(data_file, sheet_list=sheet_list, header=0, col_list=[2, 1])  # 0-Va, 1-Ia
        data_dict[n] = data  # 将数据保存在字典中，方便读取调用

        for m in range(len(sheet_list)):
            V.append(data[m][0])
            I.append(data[m][1])
            I_abs.append(abs(data[m][1]))

    # 对数据进行切割，分为上升部分跟下降部分
    V_rising, V_falling = Spiltting(V)
    I_rising, I_falling = Spiltting(I)

    V_rising_merge, V_falling_merge = [-Merging(V_rising), -Merging(V_falling)]  # 将分开储存的数据融合为一个序列
    I_rising_merge, I_falling_merge = [-Merging(I_rising), -Merging(I_falling)]  # 加负号把第三象限的曲线转移到第一象限

    ExpData_rising = I_rising_merge[np.argwhere(V_rising_merge==4)].reshape(-1)
    ExpData_falling = I_falling_merge[np.argwhere(V_falling_merge==4)].reshape(-1)

    ####################################################################################################################
    # 导入拟合曲线模型

    rising_coefficient = np.array([0, 2.668991231, -22.94906632, 33.36935753, -10.48490711, 1.444052337]) * 1e-9
    falling_coefficient = np.array([0, 4.946302714, -6.536767123, 14.87714698, -1.146996822]) * 1e-9

    def IV_device(V, mode) -> float:
        if mode == 'rising':
            return CF.Y_PolyFit(V, rising_coefficient)
        elif mode == 'falling':
            return CF.Y_PolyFit(V, falling_coefficient)
        else:
            raise ValueError('Invalid mode')

    ####################################################################################################################
    # 统计模型偏差

    # print(ExpData_rising.shape)

    # breakpoint()

    scaling_factor = 1e9  # 缩放因子, 缩放为纳安

    num_data = ExpData_rising.shape[0]

    deviation = np.zeros(num_data*2)
    for i in range(num_data):
        deviation_rising = IV_device(4, 'rising') - ExpData_rising[i]
        deviation_falling = IV_device(4, 'falling') - ExpData_falling[i]
        deviation[i] = deviation_rising*scaling_factor
        deviation[i+num_data] = deviation_falling*scaling_factor

    ####################################################################################################################
    # 统计模型偏差的分布

    # 使用scipy的norm类来拟合数据
    mean, std = norm.fit(deviation)
    xmin, xmax = plt.xlim(-50,50)
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean, std)

    ####################################################################################################################
    # 画图模块
    # VISION.GlobalSetting()  # 引入全局画图变量

    # 设置刻度线方向
    plt.rcParams['xtick.direction'] = 'in'  # 将x轴的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度线方向设置向内

    plt.hist(deviation, bins=15, density=True, alpha=0.6, color='b')
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu = %.2f,  std = %.2f" % (mean, std)
    plt.title(title)

    saving_dir = 'E:/PhD_research/NonlinearNode_for_InformationSecurity/Manuscript_Theory/补充数据'  # Lingjiang
    for fmt in ['png', 'pdf', 'eps']:
        plt.savefig(f'{saving_dir}/deviation.{fmt}')

    plt.show(block=True)