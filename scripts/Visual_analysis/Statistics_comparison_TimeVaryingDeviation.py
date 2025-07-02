# NOTE: 导入环境
import os, sys
current_path = os.path.abspath(os.path.dirname(__file__))  # 获取文件目录
project_path = current_path[:current_path.find('RIGEL') + len('RIGEL')]  # 获取项目根路径，内容为当前项目的名字，即RIGEL
sys.path.append(project_path)  # 将项目根路径添加到系统路径中，以便导入项目中的模块

result_dir = f'{project_path}/results'  # 用于存放结果的目录

# NOTE: 导入所需的库和模块
import pandas as pd  # 导入pandas库用于数据处理
import seaborn as sns  # 导入seaborn库用于绘图
import matplotlib.pyplot as plt  # 导入matplotlib库用于绘图

def DataFrame_reformat(result_DF_1: pd.DataFrame) -> pd.DataFrame:

    result_dict_reformat = {'R2': [], 'MSE': [], 'eval_length': []}  # 初始化结果字典

    eval_length = [500, 1000, 1500]  # 定义评估长度

    for length in eval_length:
        mse_col = f'MSE_{length}'
        r2_col = f'R2_{length}'

        if mse_col in result_DF.columns and r2_col in result_DF.columns:
            result_dict_reformat['MSE'].extend(result_DF[mse_col].tolist())
            result_dict_reformat['R2'].extend(result_DF[r2_col].tolist())
            result_dict_reformat['eval_length'].extend([length] * len(result_DF))

    return pd.DataFrame(result_dict_reformat)  # 返回重构后的DataFrame

if __name__ == '__main__':

    result_file = f'{result_dir}/Analog-ESN_comparison_with_TimeVaryingActivation/AnalogESN_w_TimeVaryingActivation_100.csv'  # 结果文件路径
    result_DF = pd.read_csv(result_file)  # 读取结果文件

    result_DF = DataFrame_reformat(result_DF)  # 重构DataFrame

    grid = sns.FacetGrid(result_DF, col='eval_length', sharex=False)  # 创建一个网格图 (x轴不共享)
    grid.map(sns.histplot, 'MSE', kde=True)  # 绘制每个子图的直方图

    for fmt in ['eps', 'png', 'pdf']:
        plt.savefig(f'{result_dir}/Analog-ESN_comparison_with_TimeVaryingActivation/AnalogESN_w_TimeVaryingActivation_100.{fmt}')

    plt.show(block=True)  # 显示图形