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

# 检查窗口内是否有任何Z轴数据点超过阈值
def detect_turn(gyro_data, threshold=0.5):
    return np.abs(gyro_data['z']).max() > threshold  

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

# 确定起始和结束位置点
start_loc = location[0]
end_loc = location[-1]

# 计算总位移
total_displacement = np.linalg.norm(end_loc - start_loc)

# 计算总时间
total_time = gyroscope_data['seconds_elapsed'].iloc[-1] - gyroscope_data['seconds_elapsed'].iloc[0]

# 计算平均速度
average_speed = total_displacement / total_time

print(f'Average speed calculated from total displacement: {average_speed:.2f} m/s')

# 计算速度向量的模（即速率）的平均值
average_speed = np.mean(np.linalg.norm(velocity, axis=1))

print(f'Average speed calculated from velocity time series: {average_speed:.2f} m/s')



# 设置转弯检测阈值
turning_threshold = 0.5  # 需要根据实际数据进行调整

#初始化用于收集每个时间窗口的平均速度数据的列表
average_velocities = []
time_windows = []

# 遍历陀螺仪数据，以检测转弯
for start_time in np.arange(gyroscope_data['seconds_elapsed'].min(), gyroscope_data['seconds_elapsed'].max(), 5):
    end_time = start_time + 5
    # 筛选当前窗口的陀螺仪数据,加速度数据
    current_window_gyro = gyroscope_data[(gyroscope_data['seconds_elapsed'] >= start_time) & (gyroscope_data['seconds_elapsed'] < end_time)]
    current_window_acc = acceleration_data[(acceleration_data['seconds_elapsed'] >= start_time) & (acceleration_data['seconds_elapsed'] < end_time)]

    #检测是否转弯
    turning=detect_turn(current_window_gyro)

    if turning:
        print(f"Turning detected in the time window from {start_time} to {end_time}.")
    else:
        for row_acc in current_window_acc.itertuples():     #遍历加速度数据，计算速度和位置信息 
            new_acc = np.array([row_acc.x, row_acc.y, row_acc.z])  # 直接使用滤波后的加速度数据
            new_v = velocity[-1] + new_acc * dt_acc
            new_loc = location[-1] + new_v * dt_acc
            
            # 更新速度和位置数组
            velocity = np.vstack((velocity, new_v))
            location = np.vstack((location, new_loc))
            # print(f"Velocity: {new_v}, Location: {new_loc}")

 # 检查窗口内是否有任何Z轴数据点超过阈值
    if not any(abs(current_window_gyro['z']) > turning_threshold):
        print(f"No turning detected in the time window from {start_time} to {end_time}.")
        # 如果没有转弯，计算并输出这5秒内的平均速度
        current_window_acc = acceleration_data[(acceleration_data['seconds_elapsed'] >= start_time) & (acceleration_data['seconds_elapsed'] < end_time)]
        average_velocity = current_window_acc[['x', 'y', 'z']].mean().values
        if not current_window_acc.empty:
            # 计算当前窗口内的平均加速度的模
            avg_acc_magnitude = np.linalg.norm(current_window_acc[['x', 'y', 'z']].mean().values)
            # 将平均加速度转换为平均速度变化（这里假设加速度恒定）
            avg_velocity_change = avg_acc_magnitude * 5  # 5秒时间窗口
            # 存储平均速度变化和时间窗口的开始时间
            average_velocities.append(avg_velocity_change)  
            time_windows.append(start_time)


# 每5秒更新一次速度的列表
velocity_updates = []
time_stamps = []

# 每5秒计算一次速度
window_size = 5  # 5秒窗口
for start_time in np.arange(gyroscope_data['seconds_elapsed'].min(), gyroscope_data['seconds_elapsed'].max() - window_size, window_size):
    end_time = start_time + window_size
    # 筛选当前窗口的加速度数据
    current_window_acc = acceleration_data[(acceleration_data['seconds_elapsed'] >= start_time) & (acceleration_data['seconds_elapsed'] < end_time)]
    
    if not current_window_acc.empty:
        # 计算当前窗口内的平均加速度
        avg_acc = current_window_acc[['x', 'y', 'z']].mean().values
        # 将平均加速度近似为瞬时加速度，乘以时间窗口得到速度变化
        velocity_change = avg_acc * window_size
        velocity_updates.append(np.linalg.norm(velocity_change))  # 存储速度变化的模
        time_stamps.append(start_time)

overall_average_velocity = np.mean(average_velocities)

print(f'Overall average speed across time windows: {overall_average_velocity:.2f} m/s')

# 假设 velocity 是一个 Nx3 的数组，其中 N 是样本数量，3 是 x, y, z 速度分量
def calculate_average_velocity(velocity, fs):
    window_size = 5 * fs  # 5秒内的样本数量，fs 是采样频率
    num_windows = len(velocity) // window_size
    average_velocities = []

    for i in range(num_windows):
        start_index = i * window_size
        end_index = start_index + window_size
        window = velocity[start_index:end_index]
        window_average = np.mean(np.linalg.norm(window, axis=1))  # 计算每个窗口的速度大小平均值
        average_velocities.append(window_average)

    return average_velocities

# 使用示例
fs = 100  # 采样频率为 100 Hz
average_velocities = calculate_average_velocity(velocity, fs)
print(f'Average speed across 5-second windows: {np.mean(average_velocities):.2f} m/s')



# 绘制速度更新图表
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=time_stamps, y=velocity_updates, mode='lines+markers', name='Velocity Update'))
# fig.update_layout(title='Velocity Updates Over Time', xaxis_title='Time (s)', yaxis_title='Velocity Update (units/s)')
# fig.show()




# # 绘制每5秒内的平均速度变化
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=time_windows, y=average_velocities, mode='lines+markers', name='Average Velocity'))
# fig.update_layout(title='Average Velocity Over Time', xaxis_title='Time (s)', yaxis_title='Average Velocity (units/s)', xaxis=dict(tickmode='linear', dtick=5))
# fig.show()


# # 绘制加速度数据的图表
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







