import numpy as np
import pandas as pd
import transforms3d
import plotly.graph_objects as go
from navigation_utils import *
from scipy.signal import butter, filtfilt



def dead_reckon(dir_name, remove_bias=False, title='', sma=0):

    pose, steps, _, name = dir_name.split('-')

    data, fs = load_session(dir_name)

    gyro = data['Gyroscope']
    dt_gyro = 1 / fs['Gyroscope']

    acc = data['Accelerometer']
    dt_acc = 1 / fs['Accelerometer']

    gravity = data['Gravity']
    phi_0, theta_0 = get_initial_alignment(gravity)

    if remove_bias:
        gyro_biases, acc_biases, acc_scale, acc_bias_per_axis = get_calibration_values('calibration', user=name)
        gyro[['x', 'y', 'z']] = gyro[['x', 'y', 'z']] - gyro_biases[['x', 'y', 'z']].to_numpy()
    else:
        acc_biases = pd.DataFrame({'x': [0], 'y': [0], 'z': [0]})
        acc_scale = pd.DataFrame({'x': [1], 'y': [1], 'z': [1]})
        acc_bias_per_axis = acc_biases
        gyro_biases = acc_biases

    if sma:  # simple-moving-average
        acc['x'] = acc['x'].rolling(window=sma, min_periods=1).mean()
        acc['y'] = acc['y'].rolling(window=sma, min_periods=1).mean()
        acc['z'] = acc['z'].rolling(window=sma, min_periods=1).mean()

    # -------------------- Init arrays: ------------------------------------------------------------------------------

    start_pose = np.array([phi_0, theta_0, 0])  # pose (attitude) at t=0
    start_loc = np.array([0, 0, 0])  # location at t=0
    start_velocity = np.array([0, 0, 0])  # velocity at t=0
    start_acc = np.array([0, 0, 0])  # acceleration (in ref frame) at t=0

    pose = np.array([start_pose])  # roll (phi), pitch (theta), yaw (psy)
    velocity = np.array([start_velocity])
    location = np.array([start_loc])
    acceleration = np.array([start_acc])

    # -------------------- Calc position and orientation: ------------------------------------------------------------

    for row_gyro, row_acc in zip(gyro.itertuples(), acc.itertuples()):
        angular_velocity = np.array([row_gyro.y-gyro_biases['y'], row_gyro.x-gyro_biases['x'], row_gyro.z-gyro_biases['z']])
        new_pose = pose[-1, :] + np.matmul(raw2euler(pose[-1, :]), angular_velocity) * dt_gyro
        pose = np.vstack((pose, new_pose))

        # in our case the gravity vector is already subtracted from the acceleration but we have other problems...
        tfm = euler2tfm(pose[-1, :])

        # as bias errors (in our sensor) are not constant per axis per orientation (weird non-linearity)
        # we calculate a mean bias term for each axis in the three main orientations (x,y,z)

        new_acc = np.matmul(tfm.T, np.array([(row_acc.y - acc_biases['y']) * acc_scale['y'],
                                           (row_acc.x - acc_biases['x']) * acc_scale['x'],
                                           (row_acc.z - acc_biases['z']) * acc_scale['z']])).squeeze()

        acceleration = np.vstack((acceleration, new_acc))

        new_v = velocity[-1, :] + np.mean([acceleration[-2, :], acceleration[-1, :]], axis=0) * dt_acc
        velocity = np.vstack((velocity, new_v))

        new_loc = location[-1, :] + np.mean([velocity[-1, :], velocity[-2, :]], axis=0) * dt_acc
        location = np.vstack((location, new_loc))

        print(location)
        return location

# 数据所在的目录名
dir_name = 'walking/inhand-20-steps-Cxh'

# 从 dead_reckon 函数获取位置数据
location_data = dead_reckon(dir_name, remove_bias=False, title='', sma=0)

# 计算总距离，假设最后一个位置为最终位置，第一个位置为起始位置
total_distance = np.linalg.norm(location_data[-1] - location_data[0])

# 获取加速度计的采样频率来计算总时间
_, fs = load_session(dir_name)  # 获取传感器的采样频率数据
total_time = len(location_data) / fs['Accelerometer']  # 计算总时间

# 计算平均速度
average_speed = total_distance / total_time

print(f'Average Speed: {average_speed:.2f} m/s')

