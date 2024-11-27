import numpy as np
import matplotlib.pyplot as plt
from src import Activation

if __name__ == '__main__':

    Vmin, Vmax, npoints_per_cycle = (3,5,100)
    V = np.linspace(Vmin, Vmax, npoints_per_cycle)

    n_cycle = 1000
    I = np.zeros((n_cycle, npoints_per_cycle))
    for i in range(n_cycle):
        I[i] = Activation.I_Taylor_w_deviation(V)

    I_mean = np.mean(I, axis=0)
    I_upper = np.max(I, axis=0)
    I_lower = np.min(I, axis=0)

    # 画图模块
    plt.plot(V, I_mean, label='Nominal')
    plt.fill_between(V, I_upper, I_lower, alpha=0.5, label='Error')

    plt.xlim(Vmin, Vmax)

    saving_dir = 'E:/PhD_research/NonlinearNode_for_InformationSecurity/Manuscript_Theory/补充数据'  # Lingjiang
    for fmt in ['png', 'pdf', 'eps']:
        plt.savefig(f'{saving_dir}/IV_w_error bands_ZoomIn.{fmt}')

    plt.show(block=True)