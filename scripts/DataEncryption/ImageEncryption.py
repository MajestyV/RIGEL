# 对于循环次数多的情况，在terminal运行会比IDE更快
# 有时pycharm的文件结构和cmd的文件结构不一样，在cmd中运行会显示：ModuleNotFoundError: No module named 'src'
# 这可以通过在脚本开头添加项目根目录到sys.path中解决，详情请参考：https://blog.csdn.net/qq_42730750/article/details/119799157
import os
import sys

script_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
project_path = os.path.abspath(os.path.join(script_path, '..'))  # 获取项目路径
sys.path.append(project_path)  # 添加路径到系统路径中

import cv2
import numpy as np
from src import API_for_DataEncryption
import matplotlib.pyplot as plt
from PIL import Image

working_loc = 'default'  # 用于指定工作目录

saving_dir_dict = {'default': f"{project_path}/demo",
                   'Lingjiang': 'D:/Projects/NonlinearNode/Data/DataEncryption'}

image_dict = {'CUHK': f"{project_path}/gallery/CUHK.jpg",
              'Jolie_Ville_Luxor_Hotel': f"{project_path}/gallery/Jolie_Ville_Luxor_Hotel.jpg"}

# 此函数用于OpenCV读取图片matplotlib展示图片
# OpenCv读的图片是BGR的，而matplotlib的格式是RGB，所以不能直接输出图片
def OpenCV_to_Matplotlib(image):  return image[:, :, ::-1]

if __name__=='__main__':
    # 加载本地图像，转成np矩阵
    # image_name = 'Jolie_Ville_Luxor_Hotel'
    image_name = 'CUHK'
    image = cv2.imread(image_dict[image_name])

    print(image.shape)

    # 显示原始图像
    # cv2.imshow("Original image", image)
    # cv2.waitKey()  # 原始图像持续显示时间

    # 将图像转化成字节
    image_bytes = image.tobytes()
    print("Original image bytes: " + str(len(image_bytes)))

    ####################################################################################################################
    # 数据加密模块
    encrypt_tool = API_for_DataEncryption.ImageEncryption(mode='ECB')  # 加载数据加密套件

    # 加密部分 ----------------------------------------------------------------------------------------------
    random_bit_datafile = f"{project_path}/demo/Key_LogisticHyperChaotic"
    key, init_vec = encrypt_tool.GenKey_from_datafile(random_bit_datafile,key_size=32)  # 产生密钥
    image_encrypted = encrypt_tool.Encrypt(image,key=key,init_vec=init_vec)  # 对图像进行加密

    # 显示加密后的图像
    # cv2.imshow("Encrypted image", image_encrypted)
    # cv2.waitKey()

    # 解密部分 ----------------------------------------------------------------------------------------------
    image_decrypted = encrypt_tool.Decrypt(image_encrypted,key=key)  # 对图像进行解密

    # 显示解密后的图像
    # cv2.imshow("Decrypted Image", image_decrypted)
    # cv2.waitKey()

    # 关闭所有窗口
    # cv2.destroyAllWindows()

    ####################################################################################################################
    # 画图模块

    image_mpl = OpenCV_to_Matplotlib(image)
    image_encrypted_mpl = OpenCV_to_Matplotlib(image_encrypted)
    image_decrypted_mpl = OpenCV_to_Matplotlib(image_decrypted)

    mode = 'color_dist'
    if mode == 'image':
        fig = plt.figure(figsize=(12, 5))
        fig.subplots_adjust(hspace=0.2, wspace=0.2)

        # 分配子图位置
        img_1 = fig.add_subplot(1, 3, 1)
        img_2 = fig.add_subplot(1, 3, 2)
        img_3 = fig.add_subplot(1, 3, 3)

        # 画原图以及加解密图
        img_1.imshow(Image.fromarray(np.uint8(image_mpl)))
        img_2.imshow(Image.fromarray(np.uint8(image_encrypted_mpl)))
        img_3.imshow(Image.fromarray(np.uint8(image_decrypted_mpl)))

    elif mode == 'color_dist':
        fig = plt.figure(figsize=(12, 12))
        fig.subplots_adjust(hspace=0.2, wspace=0.2)

        color_dist = [[fig.add_subplot(3, 3, 1),fig.add_subplot(3, 3, 2),fig.add_subplot(3, 3, 3)],  # R通道色彩分布
                      [fig.add_subplot(3, 3, 4),fig.add_subplot(3, 3, 5),fig.add_subplot(3, 3, 6)],  # G通道色彩分布
                      [fig.add_subplot(3, 3, 7),fig.add_subplot(3, 3, 8),fig.add_subplot(3, 3, 9)]]  # B通道色彩分布

        # 利用OpenCV计算图像色彩分布
        x = np.linspace(0, 256, 256)
        color_distribution = [  # R通道色彩分布
            [cv2.calcHist([image_mpl], [0], None, [256], [0, 256])[:, 0],  # Original
             cv2.calcHist([image_encrypted_mpl], [0], None, [256], [0, 256])[:, 0],  # Encrypted
             cv2.calcHist([image_decrypted_mpl], [0], None, [256], [0, 256])[:, 0]],  # Decrypted
            # G通道色彩分布
            [cv2.calcHist([image_mpl], [1], None, [256], [0, 256])[:, 0],
             cv2.calcHist([image_encrypted_mpl], [1], None, [256], [0, 256])[:, 0],
             cv2.calcHist([image_decrypted_mpl], [1], None, [256], [0, 256])[:, 0]],
            # B通道色彩分布
            [cv2.calcHist([image_mpl], [2], None, [256], [0, 256])[:, 0],
             cv2.calcHist([image_encrypted_mpl], [2], None, [256], [0, 256])[:, 0],
             cv2.calcHist([image_decrypted_mpl], [2], None, [256], [0, 256])[:, 0]]
        ]

        for i in range(3):  # R, G, B channel
            for j in range(3):  # Original, encrypted, decrypted
                color_dist[i][j].plot(x, color_distribution[i][j])
                color_dist[i][j].fill_between(x, color_distribution[i][j])

                color_dist[i][j].set_xlim(0, 255)
                if i == 0:
                    color_dist[i][j].set_ylim(0, 16000)
                elif i == 1:
                    color_dist[i][j].set_ylim(0, 16000)
                else:
                    color_dist[i][j].set_ylim(0, 16000)

    else:
        exit()

    # plt.savefig('C:/Users/DELL/Desktop/临时数据文件夹/ImageEncryption.eps',format='eps',dpi=600)

    plt.show(block=True)