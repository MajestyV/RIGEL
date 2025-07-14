#!/bin/bash

python_script=$1  # 获取第一个参数作为Python脚本路径

# 指定conda环境名称
read -p "Please enter the desired conda environment: (Press enter to use default setting) " conda_env
if [ -z "$conda_env" ]; then
    conda_env=RIGEL
fi

# 激活指定的conda环境
eval "$(conda shell.bash hook)"  # 通过加载conda的shell hook来激活环境
conda activate $conda_env

# 检查conda环境是否激活成功
if [[ "$CONDA_DEFAULT_ENV" != "$conda_env" ]]; then
    echo "Error: Conda environment '$conda_env' not activated."
    exit 1
else
    echo "Conda environment ($conda_env) have been successfully activated."
fi

log_file=logs/output.log

echo "Starting the Python script: $python_script" > $log_file  # 将开始信息写入日志文件
echo -n "Start time: " >> $log_file; date >> $log_file  # 记录开始时间

# 执行Python脚本并将输出重定向到日志文件
(time nohup python "$python_script" >> $log_file 2>&1) >> $log_file 2>&1