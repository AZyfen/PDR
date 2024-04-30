import numpy as np
import pandas as pd
import transforms3d

# 假设CSV文件的路径
gyroscope_data_path = 'path_to_gyroscope_data.csv'
accelerometer_data_path = 'path_to_accelerometer_data.csv'

# 加载陀螺仪和加速度计数据
gyroscope_data = pd.read_csv(gyroscope_data_path)
acceleration_data = pd.read_csv(accelerometer_data_path)

# 选取最近5秒内的数据进行分析
time_threshold = 5  # 时间阈值为5秒
current_time = gyroscope_data['timestamp'].iloc[-1]  # 假设时间戳列名为'timestamp'
recent_gyro_data = gyroscope_data[gyroscope_data['timestamp'] > (current_time - time_threshold)]

# 检查Z轴变化来判断是转弯还是直行
z_axis_threshold = 0.1  # Z轴变化的阈值，根据实际情况调整
z_axis_var = np.var(recent_gyro_data['z'])  # 假设Z轴数据列名为'z'

if z_axis_var < z_axis_threshold:
    # 直行状态，计算这5秒内的平均速度
    recent_acc_data = acceleration_data[acceleration_data['timestamp'] > (current_time - time_threshold)]
    # 计算加速度变化，此处仅为示例，实际计算应根据具体逻辑进行调整
    average_velocity = np.mean(recent_acc_data[['x', 'y', 'z']].values, axis=0)  # 假设加速度计数据也有xyz轴
    print("Direct movement detected.")
    print("Average velocity in the last 5 seconds:", average_velocity)
else:
    print("Turning detected.")
