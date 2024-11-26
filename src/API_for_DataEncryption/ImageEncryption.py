import sys
import numpy as np
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

class ImageEncryption:
    ''' 此类函数专为图像加密而设 '''
    def __init__(self,mode='CBC'):
        # 进行加密模式自检
        if mode == 'CBC' or mode == 'ECB':  # 目前仅支持CBC或者ECB加密模式
            self.mode = mode
        else:
            print('Please specify a valid mode ! ! !')
            exit()

    # 此函数专用于将二进制的比特串转换为特定位数的随机字节串
    # This function is designed for converting random bit train to random bytes for data encryption usage.
    def Bit_to_Byte(self,bit_train, bytes_length=32, binary_digit=8):
        byte_train = bytearray(bytes_length)  # 按照需要长度创建一个空字节数组

        for i in range(bytes_length):
            bit_array = bit_train[i * binary_digit:i * binary_digit + 8]  # 从输入比特流中截取比特串（应注意，python取值左闭右开）
            random_int = 0  # 初始化随机整型
            for j in range(binary_digit):
                random_int += bit_array[j] * 2 ** j  # 计算随机整型

            byte_train[i] = random_int

        return byte_train

    # 此函数可以通过存储了随机比特流的数据文件生成密钥
    def GenKey_from_datafile(self,random_bit_datafile, key_size=32, binary_digit=8):
        with open(random_bit_datafile, "r") as file:  # 打开文件
            data = file.read()  # 读取文件

        init_vec_size = AES.block_size if self.mode == 'CBC' else 0  # init_vec_size = 16 初始向量长度不变

        key_bit_train = []  # 创建空列表用于存放密钥的随机比特流
        init_vec_bit_trian = []  # 创建空列表用于存放初始向量的随机比特流（用于CBC模式的加密）
        for i in range(key_size * binary_digit + init_vec_size * binary_digit):
            if i < key_size * binary_digit:
                key_bit_train.append(int(data[i]))  # 先生成密钥
            else:
                init_vec_bit_trian.append(int(data[i]))  # 再生成初始向量

        key = self.Bit_to_Byte(key_bit_train, bytes_length=key_size, binary_digit=binary_digit)
        init_vec = self.Bit_to_Byte(init_vec_bit_trian, bytes_length=init_vec_size, binary_digit=binary_digit) \
            if self.mode == 'CBC' else bytes()

        return key, init_vec

    # 此函数专用于检查图像的宽度是否低于图像加密的宽度限制
    def CheckImageWidth(self,image):
        image_row, image_col, image_channel = image.shape  # 获取图像大小信息，分别为行数，列数，色彩通道数
        print("AES.block_size: " + str(AES.block_size))
        min_width = (AES.block_size + AES.block_size) // image_channel + 1
        print("Minimum width: " + str(min_width))
        if image_col < min_width:
            print(
                'The minimum width of the image must be {} pixels, so that IV and padding can be stored in a single additional row!'.format(
                    min_width))
            sys.exit()
        return

    # 图像加密函数
    def Encrypt(self,image, key, init_vec=bytes()):
        image_row, image_col, image_channel = image.shape  # 获取图像大小信息，分别为行数，列数，色彩通道数
        image_bytes = image.tobytes()  # 将图像转化成字节

        # 初始化AES加密器
        cipher = AES.new(key, AES.MODE_CBC, init_vec) if self.mode == 'CBC' else AES.new(key, AES.MODE_ECB)

        image_BytesPadded = pad(image_bytes, AES.block_size)
        cipher_text = cipher.encrypt(image_BytesPadded)

        # bytes(s) 返回字节
        padded_size = len(image_BytesPadded) - len(image_bytes)  # 填充的位数
        print('paddedSize: ' + str(padded_size))

        init_vec_size = AES.block_size if self.mode == 'CBC' else 0  # init_vec_size = 16 初始向量长度不变
        void = image_col * image_channel - init_vec_size - padded_size
        init_vec_ciphertext_void = init_vec + cipher_text + bytes(void)

        # frombuffer将data以流的形式读入转化成ndarray对象
        # 第一参数为stream,第二参数为返回值的数据类型，第三参数指定从stream的第几位开始读入
        # data是字符串的时候，Python3默认str是Unicode类型，所以要转成bytestring在原str前加上
        # 因为进行了数据填充，所有加密后的图像会比原图像多1行
        image_encrypted = np.frombuffer(init_vec_ciphertext_void, dtype=image.dtype).reshape(image_row + 1, image_col,
                                                                                             image_channel)

        return image_encrypted

    # 图像解密函数
    def Decrypt(self,image_encrypted, key, mode='CBC'):
        # 获取加密图像大小信息，分别为行数，列数，色彩通道数
        encrypted_img_row, encrypted_img_col, encrypted_img_channel = image_encrypted.shape
        encrypted_img_row = encrypted_img_row - 1  # 因为进行了数据填充，所有加密后的图像会比原图像多1行

        encrypted_bytes = image_encrypted.tobytes()  # np矩阵转字节

        init_vec_size = AES.block_size if mode == 'CBC' else 0  # init_vec_size = 16 初始向量长度不变
        init_vec = encrypted_bytes[:init_vec_size]  # 取前init_vec_size位为init_vec
        image_BytesSize = encrypted_img_row * encrypted_img_col * encrypted_img_channel
        padded_size = (image_BytesSize // AES.block_size + 1) * AES.block_size - image_BytesSize  # 确定填充的字节数
        encrypted = encrypted_bytes[init_vec_size:init_vec_size + image_BytesSize + padded_size]  # 确定图像的密文

        # 解密
        cipher = AES.new(key, AES.MODE_CBC, init_vec) if mode == 'CBC' else AES.new(key, AES.MODE_ECB)
        decrypted_image_BytesPadded = cipher.decrypt(encrypted)

        # print(len(decrypted_image_BytesPadded))
        # print(AES.block_size)
        # decrypted_image_bytes = unpad(decrypted_image_BytesPadded, AES.block_size)  # 去除填充的数据
        decrypted_image_bytes = unpad(decrypted_image_BytesPadded, 32)

        # 把字节转化成图像
        image_decrypted = np.frombuffer(decrypted_image_bytes, image_encrypted.dtype).reshape(encrypted_img_row,
                                                                                            encrypted_img_col,
                                                                                            encrypted_img_channel)

        return image_decrypted

if __name__=='__main__':
    pass
