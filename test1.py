import pandas as pd
import numpy as np
from scipy.signal import find_peaks, gaussian, convolve
import matplotlib.pyplot as plt

def load_data(filename):
    data = pd.read_csv(filename)
    data['l2_norm'] = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)
    return data

def process_data(data, start, end):
    sub_data = data[(data['seconds_elapsed'] >= start) & (data['seconds_elapsed'] <= end)]
    if len(sub_data) == 0:
        return 0, 0  # 如果没有数据，返回速度和步数都为0
    
    window_size = 10
    gaussian_window = gaussian(window_size, std=2)
    gaussian_window /= np.sum(gaussian_window)
    smoothed_data = convolve(sub_data['l2_norm'], gaussian_window, mode='same')

    # 步长和波峰检测参数
    step_length = 0.35  # 每步假设的距离（单位：米）
    distance_threshold = int(100 * 0.5)  # 最小步间间隔时间（假设采样率为100Hz，间隔0.5秒）
    height_threshold = 0.50  # 波峰的最小高度
    peaks, _ = find_peaks(smoothed_data, distance=distance_threshold, height=height_threshold)
    
    total_steps = len(peaks)
    total_distance = total_steps * step_length  # 总距离 = 步数 × 步长
    average_velocity = total_distance / (end - start)  # 平均速度 = 总距离 / 时间间隔

    return average_velocity, total_steps

def main():
    filename = 'Accelerometer.csv'
    data = load_data(filename)
    total_time = data['seconds_elapsed'].max()
    current_time = 0
    interval = 1
    
    velocities = []
    times = []

    while current_time <= total_time:  
        end_time = current_time + interval
        velocity, _ = process_data(data, current_time, end_time)
        velocities.append(velocity)
        times.append(end_time)
        
        # 更新图表
        plt.clf()
        plt.plot(times, velocities, label='Measured Velocity')
        plt.axhline(y=0.319, color='r', linestyle='--', label='Groundtruth Velocity')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.legend()
        plt.pause(0.05)
        
        current_time += interval
    
    plt.show()

if __name__ == "__main__":
    main()
