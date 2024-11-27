import pandas as pd

class geda:
    """ This class of function is designed for extracting data from files using pandas package. """
    def __init__(self):
        self.name = geda

    # 此函数可以利用pandas提取csv文件中的测试数据
    def GetData(self, data_file, **kwargs):
        # 一些关于数据文件的参数
        header = kwargs['header'] if 'header' in kwargs else None  # 文件中的数据列，默认为没有列名，第一行作为数据读取
        x_col = kwargs['x_col'] if 'x_col' in kwargs else 0  # 默认第一列为自变量所在列
        y_col = kwargs['y_col'] if 'y_col' in kwargs else 1  # 默认第二列为因变量所在列

        # 利用pandas提取数据，得到的结果为DataFrame格式
        data_DataFrame = pd.read_csv(data_file, header=header)  # 若header=None的话，则设置为没有列名
        data_array = data_DataFrame.values  # 将DataFrame格式的数据转换为数组
        wavelength = data_array[:, x_col]  # 自变量所在列为波长
        intensity = data_array[:, y_col]  # 因变量所在列为强度

        return wavelength, intensity

    # 此函数可以利用pandas提取excel文件中的测试数据
    def GetExcelData(self, data_file, **kwargs):
        # 一些关于数据文件的参数
        # pandas中对header的解释：Row number(s) to use as the column names, and the start of the data.
        header = kwargs['header'] if 'header' in kwargs else None  # 文件中的数据列，默认为没有列名，第一行作为数据读取
        sheet_list = kwargs['sheet_list'] if 'sheet_list' in kwargs else ['Sheet1']  # 要读取的sheet的名字
        col_list = kwargs['col_list'] if 'col_list' in kwargs else [0,1]  # 要读取数据的列的序号，默认读取前两列

        # 利用pandas提取数据，得到的结果为DataFrame格式
        # 设置sheet_name=None，可以读取全部的sheet，返回字典，key为sheet名字，value为sheet表内容
        data_DataFrame = pd.read_excel(data_file, header=header, sheet_name=None)

        num_sheet = len(sheet_list)
        num_col = len(col_list)
        data_extracted = [[] for i in range(num_sheet)]
        for n in sheet_list:

            data_array = data_DataFrame[n].values  # 将DataFrame格式的数据转换为数组

            sheet_index = sheet_list.index(n)  # 目前所读取的列表的序号
            for m in range(num_col):
                col_index = col_list[m]
                data_extracted[sheet_index].append(data_array[:,col_index])  # 将数据分配到对应的列表中

        return data_extracted

