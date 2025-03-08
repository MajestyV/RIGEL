import numpy as np
import matplotlib.pyplot as plt

def Gen_noise_seq(noise_type: str='normal', noise_dist_param: float or tuple=None, amp_avg: float=1., seq_length: int=10000) \
                  -> tuple[np.ndarray, np.ndarray]:
    '''
    这个函数可以生成一个随机噪声序列
    '''

    # 生成一个与数据大小相同的随机噪声基序列
    if noise_type == 'normal':           # 高斯噪声 (正态分布)
        drift, width = noise_dist_param  # 获取噪声的均值和标准差
        noise_basic = np.random.normal(loc=drift, scale=width, size=seq_length)

    elif noise_type == 'uniform':        # 均匀噪声 (区间内均匀分布，可以是偏态或者非偏态)
        inf, sup = noise_dist_param      # 获取下确界和上确界
        noise_basic = np.random.uniform(low=inf, high=sup, size=seq_length)

    elif noise_type == 'Poisson':        # 泊松噪声 (偏态分布, 模拟散粒噪声) [https://numpy.org/doc/2.1/reference/random/generated/numpy.random.poisson.html]
        lam = noise_dist_param           # 获取泊松噪声的均值参数
        noise_basic = np.random.poisson(lam=lam, size=seq_length)

    elif noise_type == 'Rayleigh':       # 瑞利噪声 (偏态分布) [https://numpy.org/doc/2.1/reference/random/generated/numpy.random.rayleigh.html]
        scale = noise_dist_param         # 获取瑞利噪声的尺度参数
        noise_basic = np.random.rayleigh(scale=scale, size=seq_length)
    else:
        raise ValueError('The noise type is not supported!')

    noise_adjusted = noise_basic * amp_avg  # 根据给定噪声平均幅值调节噪声信号

    return noise_basic, noise_adjusted

if __name__ == '__main__':
    noise_setting_dict = {'normal': ['normal',(0., 0.1)],
                          'uniform': ['uniform',(-0.5, 0.5)],
                          'Poisson': ['Poisson', 10],
                          'Rayleigh': ['Rayleigh', 0.1]}

    seq_length = 10000
    noise_type, noise_dist_param = noise_setting_dict['uniform']
    amp_avg = 10.

    num_bins = 100

    noise, noise_adj = Gen_noise_seq(noise_type=noise_type, noise_dist_param=noise_dist_param, amp_avg=amp_avg, seq_length=seq_length)

    # 可视化噪声分布
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].hist(noise, bins=num_bins, color='b', label='Basic noise')
    ax[1].hist(noise_adj, bins=num_bins, color='r', label='Adjusted noise')

    plt.show(block=True)