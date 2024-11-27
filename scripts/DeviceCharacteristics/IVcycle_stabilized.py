import numpy as np
import matplotlib.pyplot as plt
from src import AnalysisToolkit, VISION

working_loc = 'Lingjiang'

data_dir_dict = {'Lingjiang': 'E:/PhD_research/PhaseTransistor/Data/IV characteristics/Evolution/Raw data/M-8精选/Stabilized',
                 'MMW502': 'D:/Projects/PhaseTransistor/Data/Figures/IV characteristics/Evolution/M-8精选/Stablized_cycles',
                 'JCPGH1': 'D:/PhD_research/Gallery/IV characteristics/Evolution/Raw data/M-8精选/Stabilized'}

b2bdm = AnalysisToolkit.B2BDM()
GD = AnalysisToolkit.GetData()        # 调用GetaData模块
CF = AnalysisToolkit.CurveFitting()  # 调用CurveFitting模块
VI = VISION.Visualization_basic.plot()  # 调用Visualization模块

##############################################################################################################
# 常用函数
# 此函数可用于将数据分割为上升和下降两部分
def Spiltting(data):
    data_rising = []
    data_falling = []
    for i in range(len(data)):
        length = len(data[i])
        seperating_point = int(length/2)
        rising_part = list(data[i])[0:seperating_point]
        falling_part = list(data[i])[seperating_point:length]
        data_rising.append(np.array(rising_part))
        data_falling.append(np.array(falling_part))
    return data_rising, data_falling

# 此函数可以将分开储存的数据（具有层次，比如深层列表）整理为一个序列（数组）
def Merging(data):
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

    scaling_factor = 1e6  # 缩放因子, 缩放为微安

    # 对数据进行切割，分为上升部分跟下降部分
    V_rising, V_falling = Spiltting(V)
    I_rising, I_falling = Spiltting(I)

    # parameter_test = [1,1,1,1,0]  # 电流设置太大了会出错，这个值最好从数据图中读出 (B2BDM)
    # parameter_test = [0.73e-2,140.5]  # Diode

    # a = b2bdm.Least_square(parameter_test,abs_Ia_rising[0]*1e9,-Va_rising[0])  # B2BDM
    # a = b2bdm.Least_square_diode(parameter_test,-Merging(Va_falling),Merging(abs_Ia_falling)*1e6)  # Diode
    # a = b2bdm.DeviceFitting_polynomial(Va_falling_reshape,abs_Ia_falling_reshape, degree=4)  # Taylor expand

    # print(a)

    # v_fitted = model.Voltage_asym(i,a)

    ##############################################################################################################
    # 利用多项式回归拟合曲线

    # 拟合模块
    V_rising_merge, V_falling_merge = [-Merging(V_rising), -Merging(V_falling)]  # 将分开储存的数据融合为一个序列
    I_rising_merge, I_falling_merge = [-Merging(I_rising), -Merging(I_falling)]  # 加负号把第三象限的曲线转移到第一象限

    # rising_coefficient = CF.PolynomialRegression(V_rising_merge,I_rising_merge,degree=9,reshaping='True')
    # falling_coefficient = CF.PolynomialRegression(V_falling_merge,I_falling_merge,degree=9,reshaping='True')

    # 重构模块
    rising_coefficient = np.array([0, 2.668991231, -22.94906632, 33.36935753, -10.48490711, 1.444052337]) * 1e-9
    falling_coefficient = np.array([0, 4.946302714, -6.536767123, 14.87714698, -1.146996822]) * 1e-9

    V_poly = np.linspace(0, 4, 100)
    I_poly_rising = CF.Y_PolyFit(V_poly, rising_coefficient)
    I_poly_falling = CF.Y_PolyFit(V_poly, falling_coefficient)

    ##############################################################################################################
    # 画图模块

    VI.GlobalSetting(x_major_tick=1, y_major_tick=0.2, figsize=(1.5748, 1.732))  # 引入全局画图变量

    # 常用色值
    red = VI.MorandiColor('Redred')
    blue = VI.MorandiColor('Lightblue')

    # 可视化多项式回归的结果
    VI.Visualize(V_poly, I_poly_rising * scaling_factor, color=red, label='Fitting', zorder=1)
    VI.Visualize(V_poly, I_poly_falling * scaling_factor, color=red, zorder=1)

    # 可视化测试数据
    for i in range(len(V)):
        if i == 0:
            # 加符号把第三象限的曲线转移到第一象限
            VI.Visualize(-V_falling[i], -I_falling[i] * scaling_factor, color=blue, label=r'$I$-$V$ cycles', zorder=0)
        else:
            VI.Visualize(-V_falling[i], -I_falling[i] * scaling_factor, color=blue, zorder=0)
        VI.Visualize(-V_rising[i], -I_rising[i] * scaling_factor, color=blue, zorder=0)

    # 对于每幅图的一些定制化设定
    VI.FigureSetting(xlim=(0, 4.2), ylim=(-0.03, 0.65), xlabel='Voltage (V)', ylabel=r"Current ($\mathrm{\mu A}$)")

    plt.legend(loc='best', fontsize=6, frameon=False)

    plt.show(block=True)