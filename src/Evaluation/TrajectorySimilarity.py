import numpy as np
from scipy.spatial.distance import directed_hausdorff

def Hausdorff_distance(traj1: np.ndarray, traj2: np.ndarray, npoints_eval: int=None) -> float:
    '''
    Calculate the Hausdorff distance between two trajectories.
    [1] Code on scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.directed_hausdorff.html
    [2] An efficient algorithm for calculating the exact Hausdorff distance, TPAMI (2015). [https://ieeexplore.ieee.org/document/7053955]
    Following
    :param traj1: The first trajectory, shape=(D, N), N is the number of points, D is the dimension of the trajectory.
    :param traj2: The second trajectory, shape=(D, M), M is the number of points, D is the dimension of the trajectory.
    :return:
    '''

    if npoints_eval is not None:
        # 要转置才符合这套代码的规范，这样可以保持外部调用的统一性
        traj1, traj2 = (traj1[:, :npoints_eval].T, traj2[:, :npoints_eval].T)
    else:
        traj1, traj2 = (traj1.T, traj2.T)  # 同理，要转置

    D_traj1_to_traj2 = directed_hausdorff(traj1, traj2)[0]  # 计算 traj1 到 traj2 的 directed Hausdorff distance
    D_traj2_to_traj1 = directed_hausdorff(traj2, traj1)[0]  # 计算 traj2 到 traj1 的 directed Hausdorff distance

    return max(D_traj1_to_traj2,D_traj2_to_traj1)  # Hausdorff distance 是两者的最大值

if __name__ == '__main__':
    print('Test')