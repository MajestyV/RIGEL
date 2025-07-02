# NOTE: 导入环境
import os, sys
current_path = os.path.abspath(os.path.dirname(__file__))  # 获取文件目录
project_path = current_path[:current_path.find('RIGEL') + len('RIGEL')]  # 获取项目根路径，内容为当前项目的名字，即RIGEL
sys.path.append(project_path)  # 将项目根路径添加到系统路径中，以便导入项目中的模块

default_config_dir = f'{project_path}/configs'  # 默认配置文件目录

# NOTE: 导入所需的库和模块
import json

def Get_ESN_hyperparam(run_mode: str, ESN_hyperparam_file: str = f'{default_config_dir}/AnalogESN_hyperparam.json') -> dict:
    """
    Get the hyperparameters for the ESN model from a JSON file.
    
    Returns:
        dict: A dictionary containing the hyperparameters.
    """
    with open(ESN_hyperparam_file, 'r') as file:
        hyperparam = json.load(file)
    return hyperparam[run_mode]

if __name__ == '__main__':
    # Example usage
    hyperparam = Get_ESN_hyperparam(run_mode='AnalogLiESN_DeviceIV')
    print(hyperparam)
    # Output: {'input_scaling': 0.01, 'reservoir_dim': 400, 'spectral_radius': 0.8, 'reservoir_density': 0.1, 'leaking_rate': 0.95}