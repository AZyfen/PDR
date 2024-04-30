import pandas as pd
import numpy as np
from scipy.signal import find_peaks, gaussian, convolve
import matplotlib.pyplot as plt

# 加载数据
def load_data(filename, chunk_size=200):
    chunks = []
    for chunk in pd.read_csv(filename, chunksize=chunk_size):
        chunk['time'] = pd.to_numeric(chunk['time'], errors='coerce')
        chunk['time'] /= 1e9  # 从纳秒转换为秒
        chunks.append(chunk)
    return chunks

# 平滑数据
def smooth_data(data, window_size=10, use_gaussian=True):
    if use_gaussian:
        gaussian_window = gaussian(window_size, std=2)
        gaussian_window /= np.sum(gaussian_window)
        smoothed_data = convolve(data, gaussian_window, mode='same')
    else:
        smoothed_data = data.rolling(window=window_size, min_periods=1, center=True).mean()
    return smoothed_data

# 检测波峰
def detect_peaks(smoothed_data, distance_threshold):
    peaks, _ = find_peaks(smoothed_data, distance=distance_threshold)
    return peaks

# 计算平均速度
def calculate_velocity(peaks, time, step_length=0.345):
    if len(peaks) == 0:
        return 0
    total_distance = len(peaks) * step_length
    total_time = time[peaks[-1]] - time[peaks[0]]
    if total_time > 0:
        return total_distance / total_time
    else:
        return 0

# 模拟实时读取数据并绘制图表
def simulate_realtime(chunk_size, step_length, groundtruth_velocity, csv_file_path):
    # 用于存储时间和速度的列表
    timestamps = []
    velocities = []
    current_time = 0

    # 加载CSV文件
    data = pd.read_csv(csv_file_path)
    data['l2_norm'] = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)

    # 模拟实时读取数据
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size]
        smoothed_data = smooth_data(chunk['l2_norm'].values, window_size=10, use_gaussian=True)
        peaks = detect_peaks(smoothed_data, distance_threshold=100 * 0.5)
        velocity = calculate_velocity(peaks, chunk['time'].values, step_length)

        timestamps.append(current_time)
        velocities.append(velocity)

        current_time += 1  # 每次读取模拟2秒的时间间隔

    # 绘制速度随时间变化的图表
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, velocities, label='Calculated Velocity')
    plt.axhline(groundtruth_velocity, color='r', linestyle='--', label='Groundtruth Velocity')
    plt.title('Velocity over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.grid(True)
    plt.show()

# 运行模拟实时读取数据的函数
simulate_realtime(chunk_size=200, step_length=0.345, groundtruth_velocity=0.33, csv_file_path='Accelerometer.csv')