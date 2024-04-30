import pandas as pd
import numpy as np
from scipy.signal import find_peaks, gaussian, convolve
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def load_data(filename):
    data = pd.read_csv(filename)
    # 转换时间戳从纳秒到秒
    data['time'] = pd.to_numeric(data['time'], errors='coerce')
    data['time'] /= 1e9  # 从纳秒转换为秒
    data['l2_norm'] = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)
    return data

def smooth_data(data, window_size=10, use_gaussian=True):
    if use_gaussian:
        gaussian_window = gaussian(window_size, std=2)
        gaussian_window /= np.sum(gaussian_window)
        smoothed_data = convolve(data, gaussian_window, mode='same')
    else:#否则进行移动平均
        smoothed_data = data.rolling(window=window_size, min_periods=1, center=True).mean()
    return smoothed_data

def detect_and_plot_peaks(data, smoothed_data):
    # 假设采样率为100Hz，步间最小时间间隔0.5秒
    distance_threshold = int(100 * 0.5)  
    height_threshold = 0.65  # 根据实际数据调整此阈值

    # 在平滑数据上检测峰值
    peaks, _ = find_peaks(smoothed_data, distance=distance_threshold, height=height_threshold)
    
    # 计算步行的总时间（秒）
    total_time = data['time'].iloc[peaks].max() - data['time'].iloc[peaks].min()
    
    total_steps = len(peaks)
    average_velocity = 0
    if total_time > 0:  # 避免除以零的错误
        step_length = 0.34  # 假设每步0.60米，可以根据实际情况调整
        total_distance = total_steps * step_length
        average_velocity = total_distance / total_time  # 平均速度，单位为 m/s

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(data['time'], data['l2_norm'], label='Original L2 Norm', alpha=0.5)
    plt.plot(data['time'], smoothed_data, label='Smoothed Data', linewidth=2)
    plt.plot(data['time'].values[peaks], smoothed_data[peaks], 'x', label='Detected Peaks', color='red')
    plt.title('Step Detection in Accelerometer Data')
    plt.xlabel('Time (s)')
    plt.ylabel('L2 Norm of Acceleration')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return total_steps, average_velocity

def main():
    data = load_data('accelerometer.csv')
    window_size = 10  # 窗口大小
    smoothed_data = smooth_data(data['l2_norm'], window_size, use_gaussian=True)
    num_peaks, average_velocity = detect_and_plot_peaks(data, smoothed_data)
    print(f"Detected {num_peaks} steps with smoothing.")
    print(f"Average velocity: {average_velocity:.2f} m/s")

if __name__ == "__main__":
    main()


