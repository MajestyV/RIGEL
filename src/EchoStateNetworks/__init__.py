from .EchoStateNetwork_standard import ESN as ESN_standard                             # 标准回声状态网络模型

from .LeakyIntegrator_EchoStateNetwork_standard import Li_ESN as Li_ESN_standard       # 标准泄漏整合-回声状态网络模型

from .ESN_for_Trajectory import ESN4Trajectory as ESN4Trajectory                       # 轨迹追踪专用的回声状态网络模型
from .ESN_for_Trajectory_VerZero import ESN4Trajectory as ESN4Trajectory_v0            # 轨迹回声网络 - 零号机
from .ESN_for_Trajectory_Ver1 import ESN4Trajectory as ESN4Trajectory_v1               # 轨迹回声网络 - 初号机

# 并行Echo State Network
from .ParallelEchoStateNetwork_VerZero import Parallel_ESN as Parrllel_ESN_VerZero     # 并行ESN - 零号机

# Echo state network 衍生模型
from .EchoStateRecurrenceRelationApproximator_DevVer import ERA as ERA_DevVer          # ERA - 开发版本
from .EchoStateRecurrenceRelationApproximator_VerZero import ERA as ERA_VerZero        # ERA - 零号机
from .EchoStateRecurrenceRelationApproximator_Ver1 import ERA as ERA_v1                # ERA_version1.0

from .GenerativeAdversarialEchoStateNetwork_VerZero import GAESN as GAESN_VerZero      # GAN-ESN - 零号机

