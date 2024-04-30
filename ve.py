import numpy as np
import pandas as pd
import transforms3d
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt

# # 假设 navigation_utils.py 已经提供了这些函数的定义
# # from navigation_utils import raw2euler, euler2tfm

# # 数据预处理：滤波
# def butter_lowpass_filter(data, cutoff, fs, order=5):
#     nyq = 0.5 * fs
#     normal_cutoff = cutoff / nyq
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     y = filtfilt(b, a, data)
#     return y

# def load_and_filter_data(filepath, cutoff=2.5, fs=100):
#     data = pd.read_csv(filepath)
#     for column in ['x', 'y', 'z']:
#         data[column] = butter_lowpass_filter(data[column], cutoff, fs)
#     return data

# # 加载并预处理数据
# acceleration_data = load_and_filter_data('data/walking/inhand-20-steps-Cxh/Accelerometer.csv')
# gyroscope_data = load_and_filter_data('data/walking/inhand-20-steps-Cxh/Gyroscope.csv')

# # 将陀螺仪的数据转换为欧拉角
# def raw2euler(angle_vec):
#     TFM = np.array([[1, np.sin(angle_vec[0]) * np.tan(angle_vec[1]), np.cos(angle_vec[0]) * np.tan(angle_vec[1])],
#                     [0, np.cos(angle_vec[0]), -np.sin(angle_vec[0])],
#                     [0, np.sin(angle_vec[0]) / np.cos(angle_vec[1]), np.cos(angle_vec[0]) / np.cos(angle_vec[1])]])
#     return TFM
# #将欧拉角转换为四元数，再将四元数数据转换为旋转矩阵
# def euler2tfm(euler_angle_vec):
#     quaternion = transforms3d.euler.euler2quat(*euler_angle_vec)       #将欧拉角转化为四元数表示
#     return transforms3d.quaternions.quat2mat(quaternion)    #将四元数表示转换为对应的旋转矩阵

# # 初始化参数
# gyro_biases = {'x': 0, 'y': 0, 'z': 0}  # 从校准数据中获取陀螺仪偏差
# acc_biases = {'x': 0, 'y': 0, 'z': 0}  # 从校准数据中获取加速度计偏差
# acc_scale = {'x': 1, 'y': 1, 'z': 1}  # 从校准数据中获取加速度计比例因子
# dt_gyro = 0.01  # 陀螺仪采样周期
# dt_acc = 0.01  # 加速度计采样周期

# # 初始化数组
# pose = np.array([0, 0, 0])  # 初始化姿态数组
# angular_velocity = np.array([0, 0, 0])  # 初始化角速度数组
# tfm = np.identity(3)  # 初始化变换矩阵
# acceleration = np.array([0, 0, 0])  # 初始化加速度数组
# velocity = np.array([[0, 0, 0]])  # 使用二维数组以便于使用 np.vstack
# location = np.array([[0, 0, 0]])  # 使用二维数组以便于使用 np.vstack

# # 对加速度数据进行迭代处理
# for index, row_acc in acceleration_data.iterrows():
#     if index == 0:
#         continue  # 跳过第一行，因为它只是用于初始化

#     # 假设姿态和角速度的更新代码保持不变

#     # 更新加速度、速度和位置
#     acceleration = np.matmul(tfm.T, np.array([(row_acc.y - acc_biases['y']) * acc_scale['y'],
#                                               (row_acc.x - acc_biases['x']) * acc_scale['x'],
#                                               (row_acc.z - acc_biases['z']) * acc_scale['z']])).reshape(1, 3)
#     velocity = np.vstack((velocity, velocity[-1] + acceleration * dt_acc))
#     location = np.vstack((location, location[-1] + velocity[-1] * dt_acc))

# # 绘制速度和陀螺仪数据的图表
# def plot_data(data, title, xaxis_title, yaxis_title, legends):
#     fig = go.Figure()
#     for i, legend in enumerate(legends):
#         fig.add_trace(go.Scatter(x=list(range(len(data))), y=data[:, i], mode='lines', name=legend))
#     fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
#     fig.show()

# plot_data(velocity, 'Velocity Information', 'Time', 'Velocity', ['Velocity x', 'Velocity y', 'Velocity z'])
# plot_data(gyroscope_data[['x', 'y', 'z']].to_numpy(), 'Gyroscope Information', 'Time', 'Angular Velocity', ['Gyroscope x', 'Gyroscope y', 'Gyroscope z'])

# # 设置转弯检测阈值
# turning_threshold = 0.5  # 需要根据实际数据进行调整

# # 遍历陀螺仪数据，以检测转弯
# for start_time in np.arange(gyroscope_data['seconds_elapsed'].min(), gyroscope_data['seconds_elapsed'].max(), 5):
#     end_time = start_time + 5
#     # 筛选当前窗口的陀螺仪数据
#     current_window_gyro = gyroscope_data[(gyroscope_data['seconds_elapsed'] >= start_time) & (gyroscope_data['seconds_elapsed'] < end_time)]
#     # 计算Z轴的平均变化
#     z_axis_change = np.abs(current_window_gyro['z']).mean()

#     if any(abs(current_window_gyro['z']) > turning_threshold):
#         print(f"Turning detected in window starting at {start_time} seconds.")
#     else:
#         # 如果没有转弯，计算并输出这5秒内的平均速度
#         current_window_acc = acceleration_data[(acceleration_data['seconds_elapsed'] >= start_time) & (acceleration_data['seconds_elapsed'] < end_time)]
#         average_velocity = current_window_acc[['x', 'y', 'z']].mean().values
#         print(f"No turning detected in window starting at {start_time} seconds.")
#         print(f"Average velocity in window: {average_velocity}")



import numpy as np
import pandas as pd
import transforms3d
import plotly.graph_objects as go
from navigation_utils import raw2euler, euler2tfm
from scipy.signal import butter, filtfilt

#数据预处理：滤波 （低通滤波器：布特沃斯滤波器）        
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def load_and_filter_data(filepath, cutoff=2.5, fs=100):
    data = pd.read_csv(filepath)
    for column in ['x', 'y', 'z']:
        data[column] = butter_lowpass_filter(data[column], cutoff, fs)
    return data

# 加载并预处理数据
acceleration_data = load_and_filter_data('data/walking/inhand-20-steps-Cxh/Accelerometer.csv')
gyroscope_data = load_and_filter_data('data/walking/inhand-20-steps-Cxh/Gyroscope.csv')

# # 加载加速度数据
# acceleration_data = pd.read_csv('data\walking\inhand-20-steps-Cxh\Accelerometer.csv')
# #加载陀螺仪数据
# gyroscope_data = pd.read_csv('data\walking\inhand-20-steps-Cxh\Gyroscope.csv')

# 将陀螺仪的数据转换为欧拉角
def raw2euler(angle_vec):
    TFM = np.array([[1, np.sin(angle_vec[0]) * np.tan(angle_vec[1]), np.cos(angle_vec[0]) * np.tan(angle_vec[1])],
                    [0, np.cos(angle_vec[0]), -np.sin(angle_vec[0])],
                    [0, np.sin(angle_vec[0]) / np.cos(angle_vec[1]), np.cos(angle_vec[0]) / np.cos(angle_vec[1])]])
    return TFM
#将欧拉角转换为四元数，再将四元数数据转换为旋转矩阵
def euler2tfm(euler_angle_vec):
    quaternion = transforms3d.euler.euler2quat(*euler_angle_vec)       #将欧拉角转化为四元数表示
    return transforms3d.quaternions.quat2mat(quaternion)    #将四元数表示转换为对应的旋转矩阵


# 初始化参数
gyro_biases = {'x': 0, 'y': 0, 'z': 0}  # 从校准数据中获取陀螺仪偏差
acc_biases = {'x': 0, 'y': 0, 'z': 0}  # 从校准数据中获取加速度计偏差
acc_scale = {'x': 1, 'y': 1, 'z': 1}  # 从校准数据中获取加速度计比例因子
dt_gyro = 0.01  # 陀螺仪采样周期
dt_acc = 0.01  # 加速度计采样周期

# 初始化数组
pose = np.array([0, 0, 0])  # 初始化姿态数组
angular_velocity = np.array([0, 0, 0])  # 初始化角速度数组
tfm = np.identity(3)  # 初始化变换矩阵
acceleration = np.array([0, 0, 0])  # 初始化加速度数组
velocity = np.array([0, 0, 0])  # 初始化速度数组
location = np.array([0, 0, 0])  # 初始化位置数组

# 遍历加速度计数据，计算速度和位置信息
for row_acc in acceleration_data.itertuples():
    # 计算新的姿态
    new_pose = pose + np.matmul(raw2euler(pose), angular_velocity) * dt_gyro
    pose = new_pose

    # 将最新的姿态转换为变换矩阵
    tfm = euler2tfm(pose)

    # 计算新的加速度(包括校准，坐标系转换)
    new_acc = np.matmul(tfm.T, np.array([(row_acc.y - acc_biases['y']) * acc_scale['y'],
                                        (row_acc.x - acc_biases['x']) * acc_scale['x'],
                                        (row_acc.z - acc_biases['z']) * acc_scale['z']])).squeeze()
    acceleration = np.vstack((acceleration, new_acc))   #np.vstack在垂直(行方向)堆叠数组，构建一个包含所有加速度测量值的二维数组

    # 计算新的速度，最近2次加速度测量的平均值来估计速度的变化，axis=0确保按列计算平均值，即分别计算x,y,z轴的加速度平均，再乘以采样周期，则得到速度变化量
    new_v = velocity[-1] + np.mean([acceleration[-2], acceleration[-1]], axis=0) * dt_acc
    velocity = np.vstack((velocity, new_v)) #垂直堆叠，保存所有时刻的速度

    # 计算新的位置，最近2次速度测量的平均值来估计位置的变化，axis=0确保按列计算平均值，即分别计算x,y,z轴的速度平均，再乘以采样周期，则得到位置变化量
    new_loc = location[-1] + np.mean([velocity[-1], velocity[-2]], axis=0) * dt_acc
    location = np.vstack((location, new_loc))

# 输出速度信息
print("Velocity:")
print(velocity)

# 设置转弯检测阈值
turning_threshold = 0.5  # 需要根据实际数据进行调整

#初始化用于收集每个时间窗口的平均速度数据的列表
average_velocities = []
time_windows = []

# 遍历陀螺仪数据，以检测转弯
for start_time in np.arange(gyroscope_data['seconds_elapsed'].min(), gyroscope_data['seconds_elapsed'].max(), 5):
    end_time = start_time + 5
    # 筛选当前窗口的陀螺仪数据
    current_window_gyro = gyroscope_data[(gyroscope_data['seconds_elapsed'] >= start_time) & (gyroscope_data['seconds_elapsed'] < end_time)]
  
 # 检查窗口内是否有任何Z轴数据点超过阈值
    if not any(abs(current_window_gyro['z']) > turning_threshold):
        # 如果没有转弯，计算并输出这5秒内的平均速度
        current_window_acc = acceleration_data[(acceleration_data['seconds_elapsed'] >= start_time) & (acceleration_data['seconds_elapsed'] < end_time)]
        average_velocity = current_window_acc[['x', 'y', 'z']].mean().values
        if not current_window_acc.empty:
            average_velocity = current_window_acc[['x', 'y', 'z']].mean().values
            average_velocities.append(np.linalg.norm(average_velocity))  # 计算速度向量的模并添加到列表中
            time_windows.append(start_time)  # 记录当前时间窗口的开始时间
# 绘制每5秒内的平均速度变化
fig = go.Figure()
fig.add_trace(go.Scatter(x=time_windows, y=average_velocities, mode='lines+markers', name='Average Velocity'))
fig.update_layout(title='Average Velocity Over Time', xaxis_title='Time (s)', yaxis_title='Average Velocity (units/s)', xaxis=dict(tickmode='linear', dtick=5))
fig.show()



# 假设加速度数据的CSV文件路径
filepath = 'data\walking\inhand-20-steps-Cxh\Accelerometer.csv'

# 加载数据
acceleration_data = pd.read_csv(filepath)

# 假设加速度数据包含时间戳（秒）和加速度计在x, y, z轴上的读数
# 并且数据已经以一定的频率（例如100Hz）采集，即每0.01秒采集一次

# 计算每个点的加速度模（大小）
acceleration_data['magnitude'] = np.sqrt(acceleration_data['x']**2 + acceleration_data['y']**2 + acceleration_data['z']**2)

# 滑动窗口大小，单位为数据点的数量
# 假设采样率为100Hz，即每秒100个数据点，5秒内有500个数据点
window_size = 500  # 对应5秒

# 使用滑动窗口计算平均加速度的模
acceleration_data['avg_magnitude'] = acceleration_data['magnitude'].rolling(window=window_size, min_periods=1).mean()

# 假设加速度计采样间隔
dt = 0.01  # 100Hz采样率对应的时间间隔

# 计算每个滑动窗口的平均速度变化量（考虑到加速度可能会在正负之间变化，这里简化计算为速度的大小变化）
# 平均速度变化量 = 平均加速度的模 * 时间窗口长度
# 注意：实际应用中还需考虑初速度等因素，这里简化处理
acceleration_data['avg_velocity_change'] = acceleration_data['avg_magnitude'] * (window_size * dt)

# 显示结果
print(acceleration_data[['seconds_elapsed', 'avg_velocity_change']])

# 可以选择绘图查看结果变化
import matplotlib.pyplot as plt

plt.plot(acceleration_data['seconds_elapsed'], acceleration_data['avg_velocity_change'])
plt.xlabel('Time (s)')
plt.ylabel('Average Velocity Change (units/s)')
plt.title('Average Velocity Change Over Time')
plt.show()


# time_threshold = gyroscope_data['seconds_elapsed'].iloc[-1] - 5
# recent_gyro_data = gyroscope_data[gyroscope_data['seconds_elapsed'] > time_threshold]
# z_axis_change = np.abs(recent_gyro_data['z']).mean()  # 计算Z轴的平均变化
# turning_threshold = 0.5  # 定义转弯的阈值，需要根据实际情况调整

# if z_axis_change < turning_threshold:
#     # 直行状态，计算5秒内的平均速度
#     print("Straight movement detected, calculating average velocity.")
#     # 使用加速度数据计算5秒内的平均速度，这里简化为直接计算加速度数据的平均值
#     recent_acc_data = acceleration_data[acceleration_data['seconds_elapsed'] > time_threshold]
#     average_velocity = recent_acc_data[['x', 'y', 'z']].mean().values  # 计算平均加速度，假设为恒定加速度
#     print(f"Average Velocity in the last 5 seconds: {average_velocity}")
# else:
#     print("Turning detected, not calculating average velocity.")


# 绘制加速度数据的图表
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=list(range(len(velocity))), y=velocity[:, 0], mode='lines', name='Velocity x'))  # 绘制速度 x 轴方向
# fig.add_trace(go.Scatter(x=list(range(len(velocity))), y=velocity[:, 1], mode='lines', name='Velocity y'))  # 绘制速度 y 轴方向
# fig.add_trace(go.Scatter(x=list(range(len(velocity))), y=velocity[:, 2], mode='lines', name='Velocity z'))  # 绘制速度 z 轴方向
# fig.update_layout(title='Velocity Information', xaxis_title='Time', yaxis_title='Velocity')
# fig.show()

# # 绘制陀螺仪数据的图表
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=list(range(len(gyroscope_data))), y=gyroscope_data['x'], mode='lines', name='Gyroscope x'))  # 绘制陀螺仪 x 轴方向
# fig.add_trace(go.Scatter(x=list(range(len(gyroscope_data))), y=gyroscope_data['y'], mode='lines', name='Gyroscope y'))  # 绘制陀螺仪 y 轴方向    
# fig.add_trace(go.Scatter(x=list(range(len(gyroscope_data))), y=gyroscope_data['z'], mode='lines', name='Gyroscope z'))  # 绘制陀螺仪 z 轴方向
# fig.update_layout(title='Gyroscope Information', xaxis_title='Time', yaxis_title='Angular Velocity')
# fig.show()

# # 更新图表布局
# fig.update_layout(title='Velocity and Gyroscope Information', xaxis_title='Time', yaxis_title='Value')








