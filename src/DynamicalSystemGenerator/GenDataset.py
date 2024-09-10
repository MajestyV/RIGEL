import os
import numpy as np
from . import Mackey_Glass

class GenDataset:
    ''' This class of function is designed for generating dataset for dynamical system approximation. '''
    def __init__(self,file_name='untitled',format='txt',using_defualt_settings=True,**kwargs):
        self.num_step = kwargs['num_step'] if 'num_step' in kwargs else 5000  # 动态系统步数

        self.file_name = file_name  # 数据集文件名
        self.format = format  # 数据集文件格式

        # 设置默认数据保存路径
        default_saving_directory = os.path.abspath(os.path.dirname(__file__) + os.path.sep + "../Dataset")  # 默认保存文件夹
        self.saving_directory = kwargs['saving_directory'] if 'saving_directory' in kwargs else default_saving_directory

        self.data = kwargs['data'] if 'data' in kwargs else Mackey_Glass()

        # if using_defualt_settings:
            # self.data = Mackey_Glass()
        # else:
            # self.data =

    # 此函数可以将DynamicalSystems中生成的3D数组降维成2D数组
    def Deform(self,dataset):
        time, input, output = dataset  # 先解压数据
        dimension, data_length, = input.shape  # 获取数据维度

        data_deform = np.zeros((data_length,2*dimension+1))  # 创建空数组以存放数据

        for i in range(data_length):
            data_deform[i,0] = time[i]
            for j in range(dimension):
                data_deform[i,j+1] = input[j,i]  # 保存输入数据
                data_deform[i,j+1+dimension] = output[j,i]  # 保存输出数据

        return data_deform

    # 动态系统数据保存函数
    def SaveDataset(self):
        saving_path = self.saving_directory+'/'+self.file_name+'.'+self.format
        np.savetxt(saving_path,self.Deform(self.data))
        return

if __name__ == '__main__':
    geda = GenDataset()
    geda.SaveDataset()